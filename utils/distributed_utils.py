# distributed_utils.py
"""
Single-node PyTorch DDP process-group lifecycle for parallel_experiments.py.

setup_distributed() / cleanup_distributed() wrap torch.distributed's process
group lifecycle, falling back to a single-process no-op when launched
without torchrun so every other entry point is unaffected.

Unlike the older DDP prototype this replaces, there is no dataset-chunk
broadcasting here: utils.shared_dataset.load_dataset_cache() already reads a
pre-tokenized, mmap-backed on-disk cache, which every rank can load
independently and identically -- no rank-0-builds-then-broadcasts dance
needed.
"""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize the process group from torchrun's env vars.

    Returns (rank, local_rank, world_size). If WORLD_SIZE is not set (i.e.
    not launched via torchrun), returns (0, 0, 1) without touching
    torch.distributed at all -- the single-process case.
    """
    if "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        # No GPUs visible (e.g. a local dry run, or this module's own CPU
        # smoke tests) -- gloo lets the same control flow be exercised
        # without real GPU hardware.
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
