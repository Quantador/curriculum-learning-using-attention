# build_dataset_cache.py
"""
Standalone entrypoint to pre-tokenize and cache a config's dataset chunks.

Run this once, on a GPU-less high-CPU csub.py job, before submitting a real
GPU sweep (parallel_experiments.py) -- so the sweep's inline
get_or_build_dataset_cache() call hits a warm cache instead of blocking on
tokenization on the GPU-scheduling node.

Usage:
    python build_dataset_cache.py --config configs/test_doge_setup.yaml --num-workers 64
"""
from __future__ import annotations

import argparse
import os

from config import load_config_from_yaml
from data import get_tokenizer
from utils.shared_dataset import dataset_signature, get_or_build_dataset_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML file with ExperimentConfig field overrides")
    parser.add_argument("--num-workers", type=int, default=0, help="Tokenization worker processes (default: os.cpu_count())")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    # No model_name arg: must match every other caller in this codebase
    # (parallel_experiments.py, utils/shared_dataset.py), which always use
    # get_tokenizer() with no argument regardless of cfg.hf_model_name --
    # dataset_signature() deliberately excludes tokenizer choice from the
    # cache key on that assumption.
    tokenizer = get_tokenizer()

    path = get_or_build_dataset_cache(
        cfg, tokenizer, num_workers=args.num_workers or os.cpu_count() or 1
    )
    print(f"Dataset cache ready at: {path}")
    print(f"Signature: {dataset_signature(cfg)}")


if __name__ == "__main__":
    main()
