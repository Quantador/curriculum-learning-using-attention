# shared_dataset.py
"""
Build the train/val dataset once and persist it to disk so that the
orchestrator, the GPU memory probe, and every parallel experiment worker
can each load it (via torch.load) instead of re-tokenizing the source
HuggingFace dataset once per process.
"""
from __future__ import annotations

import torch

from config import ExperimentConfig
from data import make_mixed_chunks, make_single_chunks, MixedLMDataset


def build_dataset_cache(cfg: ExperimentConfig, tokenizer, cache_path: str) -> None:
    """Tokenize the configured dataset(s) once and save chunks+embeddings to disk."""
    if cfg.use_single_dataset:
        train_chunks, val_chunks, train_embs, val_embs = make_single_chunks(cfg, tokenizer)
    else:
        train_chunks = make_mixed_chunks("train", cfg, tokenizer)
        val_chunks = make_mixed_chunks("validation", cfg, tokenizer)
        train_embs = val_embs = None

    torch.save(
        {
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "train_embs": train_embs,
            "val_embs": val_embs,
        },
        cache_path,
    )


def load_dataset_cache(cache_path: str, max_train: int | None = None, max_val: int | None = None):
    """
    Load cached chunks and wrap them as MixedLMDataset.

    max_train/max_val optionally truncate the number of chunks used (for the
    GPU memory probe, which only needs a handful of pool-sized batches).
    """
    blob = torch.load(cache_path, weights_only=False)
    train_chunks, val_chunks = blob["train_chunks"], blob["val_chunks"]
    train_embs, val_embs = blob["train_embs"], blob["val_embs"]

    if max_train is not None:
        train_chunks = train_chunks[:max_train]
        train_embs = train_embs[:max_train] if train_embs is not None else None
    if max_val is not None:
        val_chunks = val_chunks[:max_val]
        val_embs = val_embs[:max_val] if val_embs is not None else None

    train_ds = MixedLMDataset(train_chunks, embeddings=train_embs)
    val_ds = MixedLMDataset(val_chunks, embeddings=val_embs)
    return train_ds, val_ds
