# run_experiment.py
from __future__ import annotations

import torch

from config import Config
from data import get_tokenizer, make_mixed_chunks, MixedLMDataset
from model import TinyGPT, build_router, AttentionRouter
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
        
def router_input_dim(cfg: Config) -> int:
    # Match extract_router_features() behavior
    if cfg.feature_mode in {"full", "random"}:
        # hierarchical hidden (n_chunks * d_model) + 4 stats
        return cfg.n_chunks * cfg.d_model + 4
    if cfg.feature_mode == "no_hidden":
        # stats only
        return 4
    if cfg.feature_mode == "no_hier":
        # mean hidden (d_model) + 4 stats
        return cfg.d_model + 4
    if cfg.feature_mode == "no_stats":
        # hierarchical hidden only
        return cfg.n_chunks * cfg.d_model
    raise ValueError(f"Unknown feature_mode: {cfg.feature_mode}")



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
    
    
def run_ablation_suite():
    cfg = Config()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()
    train_chunks = make_mixed_chunks("train", cfg, tokenizer)
    train_ds = MixedLMDataset(train_chunks)
    val_chunks = make_mixed_chunks("validation", cfg, tokenizer)
    val_ds = MixedLMDataset(val_chunks)

    # --- Define ablations ---
    ABLATIONS = {
        "A0_baseline": dict(kind="baseline"),
        "A1_full": dict(kind="router", router_arch="attention", feature_mode="full", reward_mode="progress"),
        "A2_text_only": dict(kind="router", router_arch="attention", feature_mode="no_hidden", reward_mode="progress"),
        "A3_hidden_only": dict(kind="router", router_arch="attention", feature_mode="no_stats", reward_mode="progress"),
        "A4_no_progress_reward": dict(kind="router", router_arch="attention", feature_mode="full", reward_mode="neg_loss"),
        "A5_linear_router": dict(kind="router", router_arch="linear", feature_mode="full", reward_mode="progress"),
        "A6_mlp_router": dict(kind="router", router_arch="mlp", feature_mode="full", reward_mode="progress"),
        "A7_random_features": dict(kind="router", router_arch="attention", feature_mode="random", reward_mode="progress"),
    }

    for name, spec in ABLATIONS.items():
        print(f"\n\n==============================")
        print(f"Running {name}")
        print(f"==============================")

        # fresh seed + fresh LM init for comparability
        set_seed(cfg.seed)

        run_cfg = Config(**vars(cfg))
        for k, v in spec.items():
            if k != "kind":
                setattr(run_cfg, k, v)

        model = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=run_cfg).to(run_cfg.device)

        metrics = MetricsTracker(name, cfg.use_wandb)
        diversity = DiversityTracker(len(train_ds))

        if spec["kind"] == "baseline":
            model = train_baseline(
                cfg=run_cfg,
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                metrics=metrics,
                diversity=diversity,
            )
        else:
            d_in = router_input_dim(run_cfg)
            router = build_router(
                d_input=d_in,
                arch=run_cfg.router_arch,
            )
            if router is not None:
                router = router.to(run_cfg.device)

            model, router = train_router(
                cfg=run_cfg,
                model=model,
                router=router,
                train_ds=train_ds,
                val_ds=val_ds,
                tokenizer=tokenizer,
                metrics=metrics,
                diversity=diversity,
            )

        metrics.save(f"{run_cfg.save_dir}/{name}_metrics.json")



if __name__ == "__main__":
    # run_experiment()
    run_ablation_suite()