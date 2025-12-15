# run_ablation.py
from __future__ import annotations

import copy
import torch

from config import Config
from data import get_tokenizer, make_mixed_chunks, MixedLMDataset
from model import TinyGPT
from metrics import MetricsTracker, DiversityTracker
from ablation_training import train_router_ablation


def set_seed(seed: int):
    import random
    import numpy as np

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


ABLATIONS = [
    # Baseline router config (for reference)
    {
        "name": "router_full_attention_improvement_topk",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    # Group 1: feature ablations
    {
        "name": "feat_no_hidden",
        "feature_mode": "no_hidden",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    {
        "name": "feat_no_stats",
        "feature_mode": "no_stats",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    {
        "name": "feat_no_hier",
        "feature_mode": "no_hier",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    {
        "name": "feat_random",
        "feature_mode": "random",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    # Group 2: architecture ablations
    {
        "name": "arch_linear",
        "feature_mode": "full",
        "arch": "linear",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    {
        "name": "arch_mlp",
        "feature_mode": "full",
        "arch": "mlp",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    {
        "name": "arch_random",
        "feature_mode": "full",
        "arch": "random",
        "reward_type": "improvement",
        "selection_type": "topk",
    },
    # Group 3: reward ablations
    {
        "name": "reward_constant",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "constant",
        "selection_type": "topk",
    },
    {
        "name": "reward_loss_before",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "loss_before",
        "selection_type": "topk",
    },
    {
        "name": "reward_minus_loss_after",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "minus_loss_after",
        "selection_type": "topk",
    },
    {
        "name": "reward_improvement_no_baseline",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "improvement_no_baseline",
        "selection_type": "topk",
    },
    # Group 4: selection ablations
    {
        "name": "sel_sample",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "sample",
    },
    {
        "name": "sel_gumbel_topk",
        "feature_mode": "full",
        "arch": "attention",
        "reward_type": "improvement",
        "selection_type": "gumbel_topk",
    },
]


def run_single_ablation(cfg: Config, ablation: dict):
    from pathlib import Path
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project="curriculum-learning-ablations",
            entity=cfg.wandb_entity,
            config={**vars(cfg), **ablation},
            name=ablation["name"],
        )

    set_seed(cfg.seed)
    tokenizer = get_tokenizer()

    print(f"\n=== Building datasets for {ablation['name']} ===")
    train_chunks = make_mixed_chunks("train", cfg, tokenizer)
    val_chunks = make_mixed_chunks("validation", cfg, tokenizer)
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)

    model = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)

    metrics = MetricsTracker(ablation["name"], use_wandb=cfg.use_wandb)
    diversity = DiversityTracker(len(train_ds))

    model, _ = train_router_ablation(
        cfg=cfg,
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=metrics,
        diversity=diversity,
        feature_mode=ablation["feature_mode"],
        arch=ablation["arch"],
        reward_type=ablation["reward_type"],
        selection_type=ablation["selection_type"],
    )

    out_dir = Path(cfg.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.save(out_dir / f"{ablation['name']}_metrics.json")

    if cfg.use_wandb:
        import wandb
        wandb.finish()


def main():
    cfg = Config()

    for ablation in ABLATIONS:
        print("\n" + "=" * 80)
        print(f"Running ablation: {ablation['name']}")
        print("=" * 80)
        run_single_ablation(copy.deepcopy(cfg), ablation)


if __name__ == "__main__":
    main()
