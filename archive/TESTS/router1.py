import os, math, random, time, argparse
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, GPT2Config, GPT2LMHeadModel,
    DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup
)

# --------------------------
# 1) Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def calc_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * (probs.clamp_min(1e-12)).log()).sum()

# --------------------------
# 2) Feature Extractor
#    (mean-pool token embeddings; cheap + effective)
# --------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, embed: nn.Embedding, out_dim: int):
        super().__init__()
        self.embed = embed
        self.out_dim = out_dim

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [M, L], attention_mask: [M, L]
        # Embed tokens only (no transformer), then mean-pool with mask
        E = self.embed(input_ids)                    # [M, L, d_model]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [M, L, 1]
            summed = (E * mask).sum(dim=1)               # [M, d_model]
            count = mask.sum(dim=1).clamp_min(1.0)       # [M, 1]
            feat = summed / count
        else:
            feat = E.mean(dim=1)                         # [M, d_model]
        return feat                                     # [M, d_model]

# --------------------------
# 3) Router MLP
# --------------------------
class Router(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # score per sample
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [M, in_dim] -> logits: [M]
        return self.net(feats).squeeze(-1)

# --------------------------
# 4) Top-k selector (ST)
# --------------------------
@dataclass
class SelectionOut:
    indices: torch.Tensor      # [k]
    mask_hard: torch.Tensor    # [M] {0,1}
    probs: torch.Tensor        # [M] softmax over logits
    mask_st: torch.Tensor      # [M] straight-through mask (hard fwd, soft bwd)

def topk_st(logits: torch.Tensor, k: int, temperature: float = 1.0) -> SelectionOut:
    M = logits.shape[0]
    probs = torch.softmax(logits / temperature, dim=0)        # [M]
    topk = torch.topk(probs, k=k, dim=0)
    idx = topk.indices
    mask_hard = torch.zeros_like(probs)
    mask_hard[idx] = 1.0
    # straight-through trick
    mask_st = (mask_hard - probs).detach() + probs
    return SelectionOut(indices=idx, mask_hard=mask_hard, probs=probs, mask_st=mask_st)

# --------------------------
# 5) Training step (manual loop)
# --------------------------
def train_one_epoch(
    model: GPT2LMHeadModel,
    router: Router,
    feat_extractor: FeatureExtractor,
    optimizer: torch.optim.Optimizer,
    router_optimizer: torch.optim.Optimizer,
    lr_sched,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    step_start: int,
) -> int:
    model.train(); router.train()
    global_step = step_start
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    for batch in dataloader:
        # batch is a candidate pool (size M)
        input_ids = batch["input_ids"].to(device)             # [M, L]
        labels     = batch["labels"].to(device)               # [M, L]
        attn_mask  = batch["attention_mask"].to(device)       # [M, L]
        M, L = input_ids.shape

        # 1) Features (no grad)
        feats = feat_extractor(input_ids, attn_mask)          # [M, d_model]

        # 2) Router scores + selection
        logits_r = router(feats)                              # [M]
        sel = topk_st(logits_r, k=args.select_k, temperature=args.temperature)
        idx = sel.indices

        # 3) Run LM only on selected samples
        sel_input  = input_ids[idx]
        sel_labels = labels[idx]
        sel_attn   = attn_mask[idx]

        out = model(input_ids=sel_input, attention_mask=sel_attn, labels=sel_labels)
        lm_loss = out.loss                                    # scalar

        # 4) Entropy regularization on full pool
        ent = calc_entropy(sel.probs)
        loss = lm_loss + args.lambda_ent * ent

        # 5) Backprop: update LM + Router
        optimizer.zero_grad(set_to_none=True)
        router_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(router.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_sched.step()
        router_optimizer.step()

        # 6) Logging
        if (global_step % args.log_steps) == 0:
            with torch.no_grad():
                topk_mean_prob = sel.probs[idx].mean().item()
                wandb.log({
                    "train/step": global_step,
                    "train/loss": float(lm_loss.item()),
                    "router/entropy": float(ent.item()),
                    "router/topk_mean_prob": topk_mean_prob,
                    "router/temperature": args.temperature,
                    "tokens/step": int(sel_input.numel()),
                })
        global_step += 1

        # 7) Temperature anneal (optional)
        if args.temp_min < args.temperature and args.temp_decay > 0:
            args.temperature = max(args.temp_min, args.temperature * args.temp_decay)

    return global_step

# --------------------------
# 6) Main
# --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=str, default="curriculum-moe")
    p.add_argument("--run_name", type=str, default="curriculum-wt2-gpt2-12x512")
    p.add_argument("--hf_repo", type=str, default="curriculum-wt2-gpt2-12x512")

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_pool", type=int, default=64, help="M: candidate pool size per step")
    p.add_argument("--select_k", type=int, default=32, help="k: selected samples per step")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--temp_min", type=float, default=0.7)
    p.add_argument("--temp_decay", type=float, default=0.9995)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lambda_ent", type=float, default=1e-3)

    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1000)

    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--output_dir", type=str, default="runs/curriculum_wt2")

    args = p.parse_args()
    set_seed(args.seed)

    # --- Init W&B
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    BLOCK = args.block_size

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=BLOCK)

    def group_texts(examples):
        ids = [i for seq in examples["input_ids"] for i in seq]
        total = (len(ids) // BLOCK) * BLOCK
        ids = ids[:total]
        chunks = [ids[i:i+BLOCK] for i in range(0, total, BLOCK)]
        return {
            "input_ids": chunks,
            "labels": chunks.copy(),
            "attention_mask": [[1]*BLOCK]*len(chunks)
        }

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
    lm_datasets = tokenized.map(group_texts, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # NOTE: DataLoader with batch_size = M yields a "candidate pool" each step
    train_loader = DataLoader(
        lm_datasets["train"],
        batch_size=args.batch_pool,
        shuffle=True,
        num_workers=2,
        collate_fn=collator
    )
    val_loader = DataLoader(
        lm_datasets["validation"],
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collator
    )

    # --- Model
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK, n_ctx=BLOCK,
        n_layer=12, n_head=8, n_embd=512, n_inner=2048,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # --- Feature extractor uses ONLY embeddings (cheap)
    embed = model.transformer.wte
    feat_extractor = FeatureExtractor(embed, out_dim=config.n_embd).to(device)

    # --- Router
    router = Router(in_dim=config.n_embd, hidden=256, dropout=0.1).to(device)

    # --- Optimizers & schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    total_steps = args.epochs * (len(lm_datasets["train"]) // args.batch_pool)
    lr_sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    router_optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=0.0)

    # --- Training
    global_step = 0
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train(); router.train()
        global_step = train_one_epoch(
            model, router, feat_extractor,
            optimizer, router_optimizer, lr_sched,
            train_loader, device, args, step_start=global_step
        )

        # --- Evaluation (dense, no routing) for clean validation PPL
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels     = batch["labels"].to(device)
                attn_mask  = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                losses.append(out.loss.detach())
        eval_loss = torch.stack(losses).mean().item()
        eval_ppl = math.exp(eval_loss)
        wandb.log({"eval/loss": eval_loss, "eval/perplexity": eval_ppl, "epoch": epoch})
        print(f"[epoch {epoch}] eval loss={eval_loss:.4f} | ppl={eval_ppl:.2f}")

        # Track best
        if eval_loss < best_val:
            best_val = eval_loss
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(router.state_dict(), os.path.join(args.output_dir, "router.pt"))

    # --- Push to Hub (optional)
    if args.push_to_hub:
        # Write a simple model card
        card = f"""---
library_name: transformers
tags: [language-modeling, curriculum, router, wikitext-2]
model-index:
- name: {args.run_name}
  results:
  - task:
      type: text-generation
    metrics:
    - name: perplexity
      type: perplexity
      value: {math.exp(best_val):.2f}
---
# {args.run_name}
Baseline LM + attention-based sample router (Top-k ST) on WikiText-2.
"""
        with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)

        # Push LM (router is saved separately as router.pt)
        model.push_to_hub(args.hf_repo, private=args.hub_private)
        tokenizer.push_to_hub(args.hf_repo, private=args.hub_private)
        # You can also upload router.pt as an artifact:
        # from huggingface_hub import HfApi
        # HfApi().upload_file(path_or_fileobj=os.path.join(args.output_dir,"router.pt"),
        #                     path_in_repo="router.pt", repo_id=args.hf_repo, repo_type="model")

    wandb.finish()

if __name__ == "__main__":
    main()
