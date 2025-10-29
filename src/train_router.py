import os, math, argparse, yaml
import torch
import torch.nn.functional as F
from transformers import (
    Trainer, TrainingArguments, set_seed, DataCollatorForLanguageModeling
)
from src.tokenizer import get_tokenizer
from src.data import build_tokenized_dataset
from src.lm import build_model
from src.pool_collator import PoolCollator
from src.router import RouterMLP, softmax_with_temp, entropy_from_probs
from src.selector import topk_straight_through, gumbel_topk_st, soft_dense_weights

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

class RouterTrainer(Trainer):
    def __init__(self, router_cfg, router, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_cfg = router_cfg
        self.router = router.to(self.args.device)

    @torch.no_grad()
    def _sample_features(self, inputs):
        """
        Compute φ(x): mean of token embeddings (no attention).
        inputs: dict with input_ids [M, L], attention_mask [M, L]
        returns feats: [M, d_model]
        """
        model = self.model
        input_ids = inputs["input_ids"]  # [M, L]
        # embed tokens only
        emb = model.transformer.wte(input_ids)  # [M, L, d_model]
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1)  # [M, L, 1]
            emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1).clamp_min(1))
        else:
            emb = emb.mean(dim=1)
        return emb  # [M, d_model]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        1) Build features φ(x) for M candidates
        2) Get router probs p over M
        3) Select k via ST or Gumbel-ST (hard forward)
        4) Run LM on selected k only; compute LM loss
        5) Add entropy reg on router probs
        """
        device = self.args.device
        M = inputs["input_ids"].size(0)
        k = self.router_cfg["k"]
        tau = self.router_cfg["temperature"]
        mode = self.router_cfg["mode"]
        ent_lambda = float(self.router_cfg["ent_lambda"])

        # 1) features (no grad into LM embeddings, keeps it cheap)
        with torch.no_grad():
            feats = self._sample_features(inputs).to(device)  # [M, d_model]

        # 2) router logits -> probs
        logits = self.router(feats)               # [M]
        probs = softmax_with_temp(logits, tau)    # [M], sum=1

        # 3) select indices/mask
        if mode == "st":
            hard, st = topk_straight_through(probs, k)
            sel_mask = hard
            p_for_grad = st
        elif mode == "gumbel_st":
            hard, st, _soft = gumbel_topk_st(probs, k, tau)
            sel_mask = hard
            p_for_grad = st
        elif mode == "soft_dense":
            # dense relaxation (compute all) – for ablation only
            sel_mask = None
            p_for_grad = probs
        else:
            raise ValueError(f"Unknown router mode: {mode}")

        # 4) LM loss on selected k (or all for dense)
        if sel_mask is not None:
            sel_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(-1)
            # Subset inputs to selected samples only
            sub_inputs = {k_: v.index_select(0, sel_idx) for k_, v in inputs.items()}
            outputs = model(**sub_inputs)
            lm_loss = outputs.loss
        else:
            # compute dense weighted loss: sum_j p_j * L(x_j)
            # run all M forward (compute-heavy)
            outputs = model(**inputs, output_hidden_states=False, return_dict=True)
            # per-token loss → mean per-sample loss
            # HF's loss is already averaged over batch; to do weighted, recompute:
            logits = outputs.logits  # [M, L, V]
            labels = inputs["labels"]  # [M, L]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none"
            ).view(M, -1)
            loss_per_sample = loss_per_token.mean(dim=1)   # [M]
            lm_loss = (p_for_grad.detach() * loss_per_sample).sum()

        # 5) entropy reg on router probs (use probs, not p_for_grad)
        ent = entropy_from_probs(probs)
        loss = lm_loss + ent_lambda * (-ent)  # add penalty = +λ * sum p log p

        return (loss, outputs) if return_outputs else loss

def main(args):
    cfg_ds   = load_yaml(args.dataset_cfg)
    cfg_model= load_yaml(args.model_cfg)
    cfg_train= load_yaml(args.train_cfg)
    cfg_rout = load_yaml(args.router_cfg)

    set_seed(42)
    tokenizer = get_tokenizer()
    train_ds, val_ds, _ = build_tokenized_dataset(cfg_ds, tokenizer, num_proc=cfg_ds.get("num_proc", 4))

    model = build_model(cfg_model, vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id)

    # Pool collator: IMPORTANT — set per_device_train_batch_size == M in train.yaml
    collator = PoolCollator(tokenizer, M=cfg_rout["M"])

    os.makedirs(cfg_train["output_dir"], exist_ok=True)
    args_tr = TrainingArguments(
        output_dir=cfg_train["output_dir"],
        per_device_train_batch_size=cfg_rout["M"],       # pool size
        per_device_eval_batch_size=cfg_train["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg_train["gradient_accumulation_steps"],
        learning_rate=cfg_train["learning_rate"],
        weight_decay=cfg_train["weight_decay"],
        warmup_steps=cfg_train["warmup_steps"],
        num_train_epochs=cfg_train["num_train_epochs"],
        lr_scheduler_type=cfg_train["lr_scheduler_type"],
        logging_steps=cfg_train["logging_steps"],
        eval_steps=cfg_train["eval_steps"],
        save_steps=cfg_train["save_steps"],
        save_total_limit=2,
        report_to="none",
        fp16=cfg_train.get("fp16", False),
        bf16=cfg_train.get("bf16", False),
        dataloader_num_workers=cfg_ds.get("num_proc", 4),
    )

    # Router input dim = model embedding dim
    in_dim = model.transformer.wte.weight.size(1)
    router = RouterMLP(in_dim, hidden=cfg_rout["router_hidden"], dropout=cfg_rout["router_dropout"])

    trainer = RouterTrainer(
        router_cfg=cfg_rout,
        router=router,
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate()
    ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
    print(f"[ROUTER] Validation perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", default="configs/dataset.yaml")
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    parser.add_argument("--router_cfg", default="configs/router.yaml")
    main(parser.parse_args())
