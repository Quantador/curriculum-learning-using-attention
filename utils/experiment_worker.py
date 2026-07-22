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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config_from_yaml
from data import get_tokenizer
from utils.metrics import MetricsTracker
from experiments import run_single_experiment, set_seed
from utils.shared_dataset import load_dataset_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-cache", required=True)
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()
    train_ds, val_ds = load_dataset_cache(args.dataset_cache)

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
