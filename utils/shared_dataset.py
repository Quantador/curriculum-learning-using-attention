# shared_dataset.py
"""
Build the train/val dataset once and persist it to disk so that the
orchestrator, the GPU memory probe, and every parallel experiment worker
can each load it (via torch.load) instead of re-tokenizing the source
HuggingFace dataset once per process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from config import ExperimentConfig
from consts import DATASET_CACHE_DIR
from data import make_chunks, MixedLMDataset


def dataset_signature(cfg: ExperimentConfig) -> dict:
    """Fields that actually change the tokenized chunks produced for cfg.

    Configs sharing a signature are assumed to need the same chunks, so
    get_or_build_dataset_cache() only tokenizes once per distinct signature
    instead of once per config. Tokenizer choice is deliberately excluded:
    every caller in this sweep path uses get_tokenizer() with no argument
    (always GPT-2), regardless of cfg.hf_model_name.
    """
    sig = {
        "block": cfg.block,
        "max_chunks": cfg.max_chunks,
        "use_external_embeddings": cfg.use_external_embeddings,
    }
    if cfg.use_external_embeddings:
        sig.update(
            external_embeddings_dataset=cfg.external_embeddings_dataset,
            single_dataset_val_split=cfg.single_dataset_val_split,
        )
    else:
        sig.update(
            split_dataset=cfg.split_dataset,
            split_column=cfg.split_column,
            dataset_list=cfg.dataset_list,
            dataset_proportions=cfg.dataset_proportions,
        )
    return sig


def _signature_hash(sig: dict) -> str:
    blob = json.dumps(sig, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def get_or_build_dataset_cache(
    cfg: ExperimentConfig, tokenizer, cache_root: Path = DATASET_CACHE_DIR
) -> Path:
    """Return the chunks.pt path for cfg's dataset signature, building it
    (and a signature.yaml sidecar for debugging) only on a cache miss."""
    sig = dataset_signature(cfg)
    entry_dir = cache_root / _signature_hash(sig)
    chunks_path = entry_dir / "chunks.pt"

    if chunks_path.exists():
        print(f"[dataset cache] hit  {entry_dir.name} -> reusing {chunks_path}")
        return chunks_path

    print(f"[dataset cache] miss {entry_dir.name} -> tokenizing for signature {sig}")
    entry_dir.mkdir(parents=True, exist_ok=True)
    with (entry_dir / "signature.yaml").open("w") as f:
        yaml.safe_dump(sig, f)
    build_dataset_cache(cfg, tokenizer, str(chunks_path))
    return chunks_path


def build_dataset_cache(cfg: ExperimentConfig, tokenizer, cache_path: str) -> None:
    """Tokenize the configured dataset(s) once and save chunks+embeddings to disk."""

    train_chunks, val_chunks, train_embs, val_embs, train_domain_names, val_domain_names = make_chunks(cfg, tokenizer)

    torch.save(
        {
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "train_embs": train_embs,
            "val_embs": val_embs,
            "train_domain_names": train_domain_names,
            "val_domain_names": val_domain_names,
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
    # .get(): older caches built before domain names were added won't have them.
    train_domain_names = blob.get("train_domain_names")
    val_domain_names = blob.get("val_domain_names")

    if max_train is not None:
        train_chunks = train_chunks[:max_train]
        train_embs = train_embs[:max_train] if train_embs is not None else None
    if max_val is not None:
        val_chunks = val_chunks[:max_val]
        val_embs = val_embs[:max_val] if val_embs is not None else None

    train_ds = MixedLMDataset(train_chunks, embeddings=train_embs, domain_names=train_domain_names)
    val_ds = MixedLMDataset(val_chunks, embeddings=val_embs, domain_names=val_domain_names)
    return train_ds, val_ds
