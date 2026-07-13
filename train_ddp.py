# train_ddp.py
"""
DDP entry point: trains the baseline (uniform random) and the router
(curriculum learning) models across a single-node PyTorch DDP process
group, one process per GPU.

Structurally mirrors compare.py's two-phase (baseline, then router) run,
but:
  - dataset chunks are built once on rank 0 and cached to disk, then every
    rank loads the same cache and takes a disjoint strided shard
    (distributed_utils.build_and_cache_chunks / shard_and_truncate)
  - TinyGPT and the router are each wrapped in DistributedDataParallel
  - wandb / MetricsTracker logging happens only on rank 0 (see the
    cfg.rank == 0 guards inside training.py / rl_training.py)

Launch:
    torchrun --standalone --nproc_per_node=<N> train_ddp.py [overrides]

Falls back to a single-process run (no torch.distributed calls at all) if
launched as plain `python train_ddp.py` -- useful for comparing against
compare.py's output on one GPU.

The aux-net baseline (compare.py's optional third model) is not supported
here -- rl_training.train_aux_baseline was intentionally left out of the
DDP rollout (see the plan's Scope section); cfg.run_aux_baseline is forced
off with a warning if set.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import replace

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from config import ExperimentConfig
from data import get_tokenizer, MixedLMDataset
from distributed_utils import (
    build_and_cache_chunks,
    cleanup_distributed,
    setup_distributed,
    shard_and_truncate,
)
from model import TinyGPT
from router import build_router, get_router_feature_dim
from training import train_baseline, compare_runs
from rl_training import train_router_experiments
from metrics import MetricsTracker, DiversityTracker


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--single-dataset-samples", type=int, default=None)
    p.add_argument("--easy-hard-split", action="store_true",
                    help="Train on easy_dataset/hard_dataset (cfg.use_single_dataset=False) "
                         "instead of the default single_dataset.")
    p.add_argument("--easy-samples", type=int, default=None)
    p.add_argument("--hard-samples", type=int, default=None)
    p.add_argument("--no-external-embeddings", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--cache-path", type=str, default="results/ddp_chunk_cache.pt",
                    help="Shared on-disk cache rank 0 writes and other ranks read.")
    return p.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.single_dataset_samples is not None:
        overrides["single_dataset_samples"] = args.single_dataset_samples
    if args.easy_hard_split:
        overrides["use_single_dataset"] = False
    if args.easy_samples is not None:
        overrides["easy_samples"] = args.easy_samples
    if args.hard_samples is not None:
        overrides["hard_samples"] = args.hard_samples
    if args.no_external_embeddings:
        overrides["use_external_embeddings"] = False
    if args.no_wandb:
        overrides["use_wandb"] = False
    return replace(cfg, **overrides) if overrides else cfg


def run() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()

    cfg = ExperimentConfig()
    cfg = apply_overrides(cfg, args)

    if cfg.run_aux_baseline and rank == 0:
        print("Warning: run_aux_baseline is not supported by train_ddp.py; ignoring.")
    cfg = replace(cfg, run_aux_baseline=False)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    cfg = replace(cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)
    if world_size > 1 and cfg.feature_cache_path:
        # A shared path would be both semantically wrong (each rank's cache
        # covers a different local shard) and racy (concurrent read/write
        # from `world_size` processes) -- keep it CPU-RAM-only per rank.
        cfg = replace(cfg, feature_cache_path="")

    set_seed(cfg.seed + rank)

    tokenizer = get_tokenizer()

    if rank == 0:
        print(f"\n=== Building datasets (world_size={world_size}) ===")
    train_chunks, val_chunks, train_embs, val_embs = build_and_cache_chunks(
        cfg, tokenizer, args.cache_path, rank, world_size
    )
    local_train_chunks, local_train_embs = shard_and_truncate(
        train_chunks, train_embs, cfg.pool, world_size, rank
    )
    train_ds = MixedLMDataset(local_train_chunks, embeddings=local_train_embs)
    val_ds = MixedLMDataset(val_chunks, embeddings=val_embs)
    if rank == 0:
        print(
            f"OK Global train chunks: {len(train_chunks)} -> "
            f"{len(local_train_chunks)} on this rank ({world_size} ranks total). "
            f"Val: {len(val_ds)} chunks (shared)."
        )

    d_input = get_router_feature_dim(cfg)
    print(f"Input size: {d_input=}")
    use_ddp = world_size > 1
    ddp_kwargs = {"device_ids": [local_rank]} if torch.cuda.is_available() else {}

    # --- Router training ---
    if rank == 0:
        print("\n=== Router training ===")
    set_seed(cfg.seed + rank)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg).to(cfg.device)
    router = build_router(d_input=d_input, arch=cfg.router_architecture, n_heads=cfg.router_n_heads)
    router = router.to(cfg.device)
    if use_ddp:
        model_router = DDP(model_router, **ddp_kwargs)
        router = DDP(router, **ddp_kwargs)

    router_metrics = MetricsTracker("router", use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router_experiments(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=router_metrics,
        diversity=router_div,
    )

    if rank == 0:
        router_metrics.save(f"{cfg.save_dir}/router_metrics.json")
    
    # --- Baseline training ---
    if rank == 0:
        print("\n=== Baseline training ===")
    set_seed(cfg.seed + rank)
    model_base = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg).to(cfg.device)
    if use_ddp:
        model_base = DDP(model_base, **ddp_kwargs)

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

    if rank == 0:
        base_metrics.save(f"{cfg.save_dir}/baseline_metrics.json")

        print("\n=== Final comparison (rank 0's local view) ===")
        compare_runs(base_metrics, router_metrics)

    cleanup_distributed()


if __name__ == "__main__":
    run()
