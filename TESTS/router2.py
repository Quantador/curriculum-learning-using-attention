# train_with_attention_router.py
import os
import math
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# ======== Config (tweak here or via env vars) ========
SEED = int(os.environ.get("SEED", 1337))
DATASET = os.environ.get("DATASET", "wikitext-2-raw-v1")  # or "wikitext-103-raw-v1"
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", 512))

# Router settings
POOL_MULT = int(os.environ.get("POOL_MULT", 8))        # candidate pool = POOL_MULT × per_device_train_batch_size
SELECT_FRAC = float(os.environ.get("SELECT_FRAC", 0.5))# select top-k = SELECT_FRAC × pool
ROUTER_HIDDEN = int(os.environ.get("ROUTER_HIDDEN", 256))
ROUTER_HEADS = int(os.environ.get("ROUTER_HEADS", 4))
TEMP = float(os.environ.get("TEMP", 1.5))              # softmax temperature
ENT_LAMBDA = float(os.environ.get("ENT_LAMBDA", 1e-3)) # entropy regularizer (add sum p log p)
USE_GUMBEL = os.environ.get("USE_GUMBEL", "1") != "0"  # Gumbel-ST vs plain ST
GUMBEL_TAU = float(os.environ.get("GUMBEL_TAU", 0.8))  # Gumbel temperature

# Logging & Hub
USE_WANDB = os.environ.get("WANDB_DISABLED", "false").lower() not in ("true", "1", "yes")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "curriculum-moe")
RUN_NAME = os.environ.get("RUN_NAME", "router-attn-wt2")
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "0") == "1"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "router-attn-wt2")

# ======== Utils ========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def chunk_concat(examples, block_size: int):
    ids = [tok for seq in examples["input_ids"] for tok in seq]
    total_len = (len(ids) // block_size) * block_size
    ids = ids[:total_len]
    chunks = [ids[i:i+block_size] for i in range(0, total_len, block_size)]
    return {
        "input_ids": chunks,
        "labels": chunks.copy(),
        "attention_mask": [[1]*block_size]*len(chunks)
    }

def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u.clamp_min(1e-20)))

# ======== Attention-based Router ========
class AttentionRouter(nn.Module):
    """
    Router using a self-attention block to contextualize samples in the candidate pool
    before scoring each one.
    """
    def __init__(self, emb_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)  # score per sample

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [M, D]
        returns logits: [M]
        """
        x = self.input_proj(feats).unsqueeze(0)  # [1, M, H]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        logits = self.out_proj(x).squeeze(0).squeeze(-1)  # [M]
        return logits

# ======== Custom Trainer ========
class RouterTrainer(Trainer):
    """
    Trainer that:
      - Builds a candidate pool by enlarging the train batch (via BatchSampler)
      - Scores candidates with an attention-based router
      - Selects top-k (hard forward), uses soft probs for entropy regularization
      - Trains LM only on selected subset per step
    We let HF/Accelerate handle AMP scaling, backward, clipping, stepping, and scheduler.
    """

    def __init__(self, *args, router: nn.Module, temp: float, ent_lambda: float,
                 pool_mult: int, select_frac: float, use_gumbel: bool, gumbel_tau: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = router.to(self.args.device)
        self.temp = temp
        self.ent_lambda = ent_lambda
        self.pool_mult = pool_mult
        self.select_frac = select_frac
        self.use_gumbel = use_gumbel
        self.gumbel_tau = gumbel_tau

    # include BOTH model + router params; let HF create scheduler & AMP scaler
    def create_optimizer(self):
        if self.optimizer is None:
            decay, no_decay = [], []
            for n, p in list(self.model.named_parameters()) + list(self.router.named_parameters()):
                if not p.requires_grad:
                    continue
                name = n.lower()
                if any(nd in name for nd in ["bias", "layernorm", "layer_norm", "ln", "norm"]):
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups = [
                {"params": decay, "weight_decay": self.args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            self.optimizer = torch.optim.AdamW(groups, lr=self.args.learning_rate, betas=(0.9, 0.95))

    # enlarge the per-step batch to form the candidate pool
    def get_train_dataloader(self) -> DataLoader:
        base_bs = self.args.per_device_train_batch_size
        M = base_bs * self.pool_mult
        sampler = RandomSampler(self.train_dataset)
        batch_sampler = BatchSampler(sampler, batch_size=M, drop_last=True)
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )

    def _sample_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Cheap per-sample features: mean over token embeddings (no grad).
        """
        with torch.no_grad():
            emb = self.model.transformer.wte(batch["input_ids"])  # [M, T, D]
            mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            feats = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom  # [M, D]
        return feats

    def _select_indices(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          hard_mask: [M] one-hot(ish) mask for selected samples
          st_mask:   [M] straight-through soft mask (not used in loss here, but available)
          probs:     [M] softmax over logits (temp-scaled)
          soft:      [M] gumbel-softmax (if enabled) or probs
        """
        probs = F.softmax(logits / self.temp, dim=0)
        if self.use_gumbel:
            g = gumbel_noise_like(logits)
            gscores = (torch.log(probs.clamp_min(1e-20)) + g) / self.gumbel_tau
            soft = F.softmax(gscores, dim=0)
        else:
            soft = probs

        M = logits.shape[0]
        k = max(1, int(self.select_frac * M))
        topk = torch.topk(soft, k=k, dim=0).indices
        hard_mask = torch.zeros_like(soft)
        hard_mask[topk] = 1.0

        st_mask = (hard_mask - soft).detach() + soft
        return hard_mask, st_mask, probs, soft

    # IMPORTANT: match HF signature and RETURN loss; no manual backward/step here.
    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        model.train()
        self.router.train()

        for k in inputs:
            inputs[k] = inputs[k].to(self.args.device)

        # 1) router features + logits
        feats = self._sample_features(inputs)            # [M, D]
        logits = self.router(feats)                      # [M]
        hard_mask, st_mask, probs, soft = self._select_indices(logits)

        # 2) build selected sub-batch (hard forward for compute)
        sel_idx = (hard_mask > 0).nonzero(as_tuple=False).squeeze(-1)
        sel_batch = {k: v.index_select(0, sel_idx) for k, v in inputs.items()}

        # 3) LM forward on selected only
        outputs = model(**sel_batch)
        lm_loss = outputs.loss

        # 4) entropy regularization for router (maximize entropy -> add sum p log p)
        ent = torch.sum(probs * torch.log(probs.clamp_min(1e-12)))  # <= 0
        loss = lm_loss + self.ent_lambda * ent

        # lightweight logging (HF advances step after optimizer)
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lm_loss": lm_loss.item(),
                        "router/entropy": (-ent).item(),
                        "router/max_prob": probs.max().item(),
                        "router/frac_selected": sel_idx.numel() / inputs["input_ids"].size(0),
                        "lr": self._get_learning_rate(),
                    })
            except Exception:
                pass

        return loss

# ======== Main ========
def main():
    set_seed(SEED)

    # --- Optional W&B ---
    if USE_WANDB:
        try:
            import wandb
            wandb.init(
                project=WANDB_PROJECT,
                name=RUN_NAME,
                config={
                    "dataset": DATASET,
                    "block_size": BLOCK_SIZE,
                    "pool_mult": POOL_MULT,
                    "select_frac": SELECT_FRAC,
                    "router_hidden": ROUTER_HIDDEN,
                    "router_heads": ROUTER_HEADS,
                    "temp": TEMP,
                    "ent_lambda": ENT_LAMBDA,
                    "use_gumbel": USE_GUMBEL,
                    "gumbel_tau": GUMBEL_TAU,
                },
                reinit=True,
            )
        except Exception:
            pass

    # --- Data ---
    ds = load_dataset("wikitext", DATASET)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=BLOCK_SIZE,
            return_overflowing_tokens=False,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    lm_ds = tokenized.map(lambda ex: chunk_concat(ex, BLOCK_SIZE), batched=True)

    # --- Model ---
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE, n_ctx=BLOCK_SIZE,
        n_layer=12, n_head=8, n_embd=512, n_inner=2048,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Router ---
    router = AttentionRouter(
        emb_dim=config.n_embd,
        hidden_dim=ROUTER_HIDDEN,
        num_heads=ROUTER_HEADS
    )

    # --- TrainingArgs ---
    # choose bf16 if supported; else fp16
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="runs/router_attn_wt2",
        run_name=RUN_NAME,
        report_to=["wandb"] if USE_WANDB else [],
        per_device_train_batch_size=8,      # base batch; actual pool = × POOL_MULT
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,      # we already scale batch via pooling
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=2_000,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        eval_steps=1000,
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=2,
        fp16=not bf16_supported,            # use FP16 scaler if bf16 not available
        bf16=bf16_supported,                # prefer bf16 on Ampere+
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HF_REPO_ID if PUSH_TO_HUB else None,
        hub_private_repo=True if PUSH_TO_HUB else None,
        hub_strategy="every_save" if PUSH_TO_HUB else "end",
    )

    trainer = RouterTrainer(
        model=model,
        args=args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        router=router,
        temp=TEMP,
        ent_lambda=ENT_LAMBDA,
        pool_mult=POOL_MULT,
        select_frac=SELECT_FRAC,
        use_gumbel=USE_GUMBEL,
        gumbel_tau=GUMBEL_TAU,
    )

    trainer.train()
    out = trainer.evaluate()
    eval_loss = out["eval_loss"]
    eval_ppl = math.exp(eval_loss)
    print(f"Validation loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

    # push final
    if PUSH_TO_HUB:
        trainer.push_to_hub(commit_message="Final model (attention router)")

    # log to W&B
    if USE_WANDB:
        try:
            import wandb
            wandb.log({"eval/loss": eval_loss, "eval/ppl": eval_ppl})
        except Exception:
            pass


if __name__ == "__main__":
    main()
