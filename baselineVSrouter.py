# baselineVSrouter.py
from __future__ import annotations

import torch

from config import ExperimentConfig
from data import get_tokenizer, make_mixed_chunks, make_single_chunks, MixedLMDataset
from model import TinyGPT
from modelExperiments import build_router, get_router_feature_dim
from training import train_baseline, train_router, compare_runs
from trainingExperiments import train_aux_baseline
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
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()

    # --- Dataset loading ---
    print("\n=== Building datasets ===")
    if cfg.use_single_dataset:
        train_chunks, val_chunks, train_embs, val_embs = make_single_chunks(cfg, tokenizer)
        train_ds = MixedLMDataset(train_chunks, embeddings=train_embs)
        val_ds   = MixedLMDataset(val_chunks,   embeddings=val_embs)
    else:
        train_chunks = make_mixed_chunks("train",      cfg, tokenizer)
        val_chunks   = make_mixed_chunks("validation", cfg, tokenizer)
        train_ds = MixedLMDataset(train_chunks)
        val_ds   = MixedLMDataset(val_chunks)

    # Feature dimensionality for the router (accounts for all enabled feature groups)
    d_input = get_router_feature_dim(cfg)

    # --- Baseline training ---
    print("\n=== Baseline training ===")
    model_base  = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    base_metrics = MetricsTracker("baseline", use_wandb=cfg.use_wandb)
    base_div     = DiversityTracker(len(train_ds))

    model_base = train_baseline(
        cfg=cfg,
        model=model_base,
        train_ds=train_ds,
        val_ds=val_ds,
        metrics=base_metrics,
        diversity=base_div,
    )
    base_metrics.save(f"{cfg.save_dir}/baseline_metrics.json")

    # --- Router training ---
    print("\n=== Router training ===")
    set_seed(cfg.seed)

    model_router  = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = build_router(
        d_input=d_input,
        arch=cfg.router_architecture,
        n_heads=cfg.router_n_heads,
    )

    router_metrics = MetricsTracker("router", use_wandb=cfg.use_wandb)
    router_div     = DiversityTracker(len(train_ds))

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

    # --- Auxiliary network baseline (optional) ---
    aux_metrics = None
    if cfg.run_aux_baseline:
        print("\n=== Auxiliary network baseline training ===")
        set_seed(cfg.seed)

        model_aux = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
        aux_net   = build_router(
            d_input=d_input,
            arch="auxnet",
            d_hidden=cfg.aux_net_hidden,
        )

        aux_metrics = MetricsTracker("aux_baseline", use_wandb=cfg.use_wandb)
        aux_div     = DiversityTracker(len(train_ds))

        model_aux, aux_net = train_aux_baseline(
            cfg=cfg,
            model=model_aux,
            aux_net=aux_net,
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            metrics=aux_metrics,
            diversity=aux_div,
        )
        aux_metrics.save(f"{cfg.save_dir}/aux_baseline_metrics.json")

    # --- Final comparison ---
    print("\n=== Final comparison ===")
    compare_runs(base_metrics, router_metrics, aux_metrics)


if __name__ == "__main__":
    run_experiment()
