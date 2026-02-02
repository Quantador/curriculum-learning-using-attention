# run_experiment.py
from __future__ import annotations

import torch

from config import ExperimentConfig
from data import get_tokenizer, make_mixed_chunks, MixedLMDataset
from model import TinyGPT, AttentionRouter
from metrics import MetricsTracker, DiversityTracker

from modelExperiments import *
from trainingExperiments import *


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()

    print("\n=== Building datasets ===")
    train_chunks = make_mixed_chunks("train", cfg, tokenizer)
    val_chunks = make_mixed_chunks("validation", cfg, tokenizer)

    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)

    # Read baseline and router metrics from previous runs
    base_metrics = MetricsTracker.load(f"results/baseline_metrics.json")
    router_metrics = MetricsTracker.load(f"results/router_metrics.json")
    
    print(f"\n=== Training experiment: {cfg.experiment_name} ===")
    
    set_seed(cfg.seed)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = build_router(
        d_input=cfg.n_chunks * cfg.d_model + 4,
        arch=cfg.router_architecture,
        d_k=128,
    )

    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router_experiments(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=experiment_metrics,
        diversity=router_div,
    )

    experiment_metrics.save(f"{cfg.save_dir}/{cfg.experiment_name}.json")
    
    
    print("\n=== Comparing runs ===")
    compare_runs_experiments(
        base_metrics,
        router_metrics,
        experiment_metrics,
    )

if __name__ == "__main__":
    run_experiment()
