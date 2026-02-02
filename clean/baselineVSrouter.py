# run_experiment.py
from __future__ import annotations

import torch

from config import Config
from data import get_tokenizer, make_mixed_chunks, MixedLMDataset
from model import TinyGPT, AttentionRouter
from training import train_baseline, train_router, compare_runs
from metrics import MetricsTracker, DiversityTracker


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment():
    cfg = Config()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()

    print("\n=== Building datasets ===")
    train_chunks = make_mixed_chunks("train", cfg, tokenizer)
    val_chunks = make_mixed_chunks("validation", cfg, tokenizer)

    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)

    print("\n=== Baseline training ===")
    model_base = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    base_metrics = MetricsTracker("baseline", use_wandb=cfg.use_wandb)
    base_div = DiversityTracker(len(train_ds))

    model_base = train_baseline(
        cfg=cfg,
        model=model_base,
        train_ds=train_ds,
        val_ds=val_ds,
        metrics=base_metrics,
        diversity=base_div,
    )

    base_metrics.save(f"{cfg.save_dir}/baseline_metrics.json")

    print("\n=== Router training ===")
    set_seed(cfg.seed)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = AttentionRouter(
        d_input=cfg.n_chunks * cfg.d_model + 4,
        d_k=128,
    )

    router_metrics = MetricsTracker("router", use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=router_metrics,
        diversity=router_div,
    )

    router_metrics.save(f"{cfg.save_dir}/router_metrics.json")

    print("\n=== Final comparison ===")
    compare_runs(base_metrics, router_metrics)


if __name__ == "__main__":
    run_experiment()
