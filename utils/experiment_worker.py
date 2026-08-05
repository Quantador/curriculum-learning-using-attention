# experiment_worker.py
"""
Run exactly one experiment given a serialized config, as a standalone
process. Launched by parallel_experiments.py so that many experiments can
run concurrently on the same GPU, each in its own CUDA context.

    python experiment_worker.py --config <yaml> --dataset-cache <path>
"""
from __future__ import annotations

import argparse
import sys
import os
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config_from_yaml
from data import get_tokenizer
from metrics import MetricsTracker
from shared_dataset import load_dataset_cache
from models.router import build_router, get_router_feature_dim
from rl_training import train_router_experiments, train_aux_baseline, compare_runs_experiments
from training import train_baseline
from models.model import TinyGPT
from metrics import MetricsTracker, DiversityTracker
from config import ExperimentConfig, load_config_from_yaml
from general_utils import set_seed

def run_single_experiment(cfg: ExperimentConfig, tokenizer, train_ds, val_ds, base_metrics, router_metrics):
    """Run a single experiment with the given configuration.

    Shared by the bare in-process run_experiment() below and by
    utils/experiment_worker.py, which calls this once per config inside its
    own subprocess when running a sweep via run_scheduler().
    """
    print(f"\n{'='*60}")
    print(f"=== Running experiment: {cfg.experiment_name} ===")
    print(f"{'='*60}")
    print(f"  router_architecture: {cfg.router_architecture}")
    print(f"  enable_text_stat: {cfg.enable_text_stat}")
    print(f"  enable_text_hierarchical: {cfg.enable_text_hierarchical}")
    print(f"  hierarchical_representation: {cfg.hierarchical_representation}")
    print(f"  training_algorithm: {cfg.training_algorithm}")
    print(f"  reward_signal: {cfg.reward_signal}")
    print(f"  selection_strategy: {cfg.selection_strategy}")
    print(f"  baseline_type: {cfg.baseline_type}")
    print(f"  temp_schedule: {cfg.temp_schedule}")
    print(f"  entropy_schedule: {cfg.entropy_schedule}")
    print(f"  run_aux_baseline: {cfg.run_aux_baseline}")
    print(f"  run_random_batch_baseline: {cfg.run_random_batch_baseline}")
    print(f"  run_random_pool_baseline: {cfg.run_random_pool_baseline}")

    set_seed(cfg.seed)

    # Ensure save directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds), domain_names=train_ds.domain_names)

    # Wall-clock time for the whole training call, uniform across all four
    # training loops below -- so runs can be compared on wall-clock time
    # later, not just final ppl (see MetricsTracker.save()'s "total_time_s").
    run_start = time.perf_counter()

    if cfg.run_aux_baseline:
        # Supervised MSE alternative to the policy-gradient router (ablation
        # baseline) — same feature pipeline and top-k selection, different
        # training objective. See rl_training.train_aux_baseline().
        aux_net = build_router(
            d_input=get_router_feature_dim(cfg, model_router.block),
            arch="auxnet",
            d_hidden=cfg.aux_net_hidden,
        )
        model_router, _ = train_aux_baseline(
            cfg=cfg,
            model=model_router,
            aux_net=aux_net,
            train_ds=train_ds,
            val_ds=val_ds,
            tokenizer=tokenizer,
            metrics=experiment_metrics,
            diversity=router_div,
        )
    elif cfg.run_random_batch_baseline or cfg.run_random_pool_baseline:
        # Non-learned controls: uniform random selection, no router/aux_net.
        # training.train_baseline() already draws cfg.batch random samples
        # from a cfg.pool-sized window each step, so the "random batch the
        # size of the pool" variant is just that same function with
        # batch widened to pool via replace() -- no new training loop needed.
        # pool_mult=1 too: cfg.pool is a property (pool_mult * batch), so
        # widening batch alone would silently re-inflate pool by another
        # factor of pool_mult, shrinking the window to 1/pool_mult of the
        # dataset and making train_baseline's random.sample(window, batch)
        # discard most of each window instead of training on all of it.
        random_cfg = cfg if cfg.run_random_batch_baseline else replace(cfg, batch=cfg.pool, pool_mult=1)
        model_router = train_baseline(
            cfg=random_cfg,
            model=model_router,
            train_ds=train_ds,
            val_ds=val_ds,
            metrics=experiment_metrics,
            diversity=router_div,
        )
    else:
        router = build_router(
            d_input=get_router_feature_dim(cfg, model_router.block),
            arch=cfg.router_architecture,
            d_k=128,
            n_heads=getattr(cfg, "router_n_heads", 1),
        )
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

    total_time_s = time.perf_counter() - run_start
    experiment_metrics.log(total_time_s=total_time_s)
    print(f"\n=== Total run time: {total_time_s:.1f}s ({total_time_s / 3600:.2f}h) ===")

    # The training loops above intentionally leave their wandb run open so
    # total_time_s lands in it too; this closes it once everything's logged.
    if cfg.use_wandb and cfg.rank == 0:
        import wandb
        wandb.finish()

    print("\n=== Comparing runs ===")
    compare_runs_experiments(
        base_metrics,
        router_metrics,
        experiment_metrics,
    )

    return experiment_metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-cache", required=True)
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    set_seed(cfg.seed)

    tokenizer = get_tokenizer(cfg.tokenizer_name)
    train_ds, val_ds = load_dataset_cache(args.dataset_cache, cfg)

    base_metrics = MetricsTracker.load("results/baseline_metrics.json")
    router_metrics = MetricsTracker.load("results/router_metrics.json")

    run_single_experiment(
        cfg=cfg,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        base_metrics=base_metrics,
        router_metrics=router_metrics,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
