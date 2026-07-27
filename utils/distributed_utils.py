# distributed_utils.py
"""
Single-node PyTorch DDP setup helpers for train_ddp.py.

setup_distributed() / cleanup_distributed() wrap torch.distributed process
group lifecycle, falling back to a single-process no-op when launched
without torchrun so every other entry point (compare.py, parallel_experiments.py,
smoke_test.py) is unaffected.

build_and_cache_chunks() / shard_and_truncate() implement the dataset
pattern used by train_ddp.py: rank 0 streams + tokenizes the HuggingFace
dataset once and caches it to disk, all ranks then load the identical
cached chunk list and take a disjoint strided slice. This avoids N ranks
redundantly re-streaming the same data, and guarantees every rank's local
shard length is an exact multiple of cfg.pool -- required so that every
rank issues the same number of pool-steps (and therefore the same number
of DDP-synchronizing .backward() calls) per epoch.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from config import Config
from data import make_mixed_chunks, make_single_chunks


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
        # No GPUs visible (e.g. a local dry run) -- gloo lets the same
        # control flow be exercised on CPU before a real NCCL/H100 run.
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def build_and_cache_chunks(
    cfg: Config,
    tokenizer,
    cache_path: str,
    rank: int,
    world_size: int,
) -> Tuple[
    List[Tuple[List[int], int]],
    List[Tuple[List[int], int]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    """
    Build (train_chunks, val_chunks, train_embs, val_embs) once on rank 0
    and cache to cache_path; every other rank loads the same file.

    Uses the same make_single_chunks()/make_mixed_chunks() calls as
    compare.py -- the tokenization/chunking logic itself is untouched,
    only which process runs it and how the result is shared.
    """
    if rank == 0:
        if cfg.use_single_dataset:
            train_chunks, val_chunks, train_embs, val_embs = make_single_chunks(cfg, tokenizer)
        else:
            train_chunks = make_mixed_chunks("train", cfg, tokenizer)
            val_chunks = make_mixed_chunks("validation", cfg, tokenizer)
            train_embs = val_embs = None

        payload = {
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "train_embs": train_embs,
            "val_embs": val_embs,
        }
        if world_size > 1:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            torch.save(payload, cache_path)

    if world_size > 1:
        dist.barrier()  # ranks != 0 block here until rank 0's torch.save() above completes
        if rank != 0:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)

    return (
        payload["train_chunks"],
        payload["val_chunks"],
        payload["train_embs"],
        payload["val_embs"],
    )


def shard_and_truncate(
    chunks: List[Tuple[List[int], int]],
    embeddings: Optional[List[torch.Tensor]],
    pool: int,
    world_size: int,
    rank: int,
) -> Tuple[List[Tuple[List[int], int]], Optional[List[torch.Tensor]]]:
    """
    Truncate chunks to a multiple of pool * world_size, then return this
    rank's disjoint strided slice (chunks[rank::world_size]).

    The truncation is what guarantees every rank's local shard length is
    itself an exact multiple of pool -- so make_index_loader() yields the
    same number of pool-steps per epoch on every rank. Without this, a
    ragged last pool on some ranks but not others would desync the
    .backward()-triggered DDP all-reduces across ranks and hang NCCL.
    """
    n = len(chunks)
    unit = pool * world_size
    n_trunc = (n // unit) * unit
    if n_trunc == 0:
        raise ValueError(
            f"Not enough chunks ({n}) to give every one of {world_size} ranks "
            f"a full pool of {pool} samples."
        )

    chunks = chunks[:n_trunc]
    embeddings = embeddings[:n_trunc] if embeddings is not None else None

    local_chunks = chunks[rank::world_size]
    local_embeddings = embeddings[rank::world_size] if embeddings is not None else None
    return local_chunks, local_embeddings
