# build_dataset_cache.py
"""
Standalone entrypoint to tokenize a config's dataset(s) into the shared cache.

This is the ONLY thing that tokenizes. Run it once, on a GPU-less high-CPU
csub.py job, before submitting a GPU sweep -- parallel_experiments.py will
refuse to run (with a pointer back to this command) if the cache is missing,
rather than blocking a GPU-scheduling node on tokenization.

Output goes to results/dataset_cache/<signature hash>/, one folder per domain
per split; see tokenization.py for the layout and utils/shared_dataset.py for
what the signature covers.

Usage:
    python build_dataset_cache.py --config configs/test_doge_setup.yaml
    python build_dataset_cache.py --config configs/test_doge_setup.yaml --workers 32 --tasks 128
    python build_dataset_cache.py --config configs/test_doge_setup.yaml --overwrite
"""
from __future__ import annotations

import argparse
import json
import os

from config import load_config_from_yaml
from consts import DATASET_CACHE_DIR
from tokenization import DEFAULT_TASKS, build_tokenized_cache, read_manifest
from utils.shared_dataset import cache_entry_dir, dataset_signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML file with ExperimentConfig field overrides")
    parser.add_argument("--workers", type=int, default=0, help="Concurrent datatrove tasks (default: os.cpu_count())")
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS, help=f"Total datatrove tasks, capped at the split's shard count (default {DEFAULT_TASKS})")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Dataset splits to tokenize (default: train validation)")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing cache entry and rebuild from scratch")
    parser.add_argument("--check", action="store_true", help="Only report whether the cache for this config exists")
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    sig = dataset_signature(cfg)
    entry_dir = cache_entry_dir(cfg)

    print(f"Config:    {args.config}")
    print(f"Signature: {json.dumps(sig, sort_keys=True)}")
    print(f"Cache dir: {entry_dir}")

    manifest = read_manifest(entry_dir)
    if args.check:
        print("Status:    " + ("BUILT" if manifest else "MISSING"))
        raise SystemExit(0 if manifest else 1)

    if manifest and not args.overwrite:
        print("Status:    already built -- nothing to do (pass --overwrite to rebuild).")
        return

    workers = args.workers or os.cpu_count() or 1 
    print(f"Workers: {workers}")
    print(f"Tasks: {args.tasks}")
    build_tokenized_cache(
        cfg,
        entry_dir,
        sig,
        tasks=args.tasks,
        workers=workers,
        splits=args.splits,
        overwrite=args.overwrite,
    )
    print(f"\nDataset cache ready at: {entry_dir}")


if __name__ == "__main__":
    main()
