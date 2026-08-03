# utils/sentence_embedder.py
"""
Precompute per-window sentence-embedder features for router training.

A frozen sentence-transformers model (cfg.sentence_embedder_model) encodes
every window in a TokenizedCorpus once, up front -- analogous to
rl_training.build_feature_cache() for hierarchical features, except this
cache never needs periodic rebuilding: the encoder does not train, so a
window's embedding is identical on epoch 1 and epoch 100 (build_feature_cache
rebuilds every feature_cache_epochs precisely because *its* features depend
on the student model, which is training).

Windows may straddle document/domain boundaries (see data.TokenizedCorpus)
before being decoded back to text here -- the same tradeoff already accepted
by enable_text_hierarchical / enable_text_stat, which also operate directly
on these windows rather than on whole documents.

load_dataset_cache() (utils/shared_dataset.py) calls build_sentence_embeddings()
and assigns the result to TokenizedCorpus.embeddings when
cfg.sentence_embedder_model is set; every training loop that branches on
`train_ds.embeddings is not None` (train_router_experiments, train_aux_baseline
in rl_training.py) picks it up automatically with no further changes.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from config import ExperimentConfig

if TYPE_CHECKING:
    from data import TokenizedCorpus


def build_sentence_embeddings(
    train_ds: "TokenizedCorpus", cfg: ExperimentConfig, tokenizer
) -> torch.Tensor:
    """Encode every window in train_ds with cfg.sentence_embedder_model.

    Returns an [N, cfg.sentence_embedder_dim] fp16 CPU tensor, one row per
    window, in train_ds order (so `embeddings[i]` matches `train_ds[i]`).

    Cache validity mirrors build_feature_cache(): if cfg.sentence_embedder_cache_path
    points at a file whose shape/dtype don't match, it is discarded and rebuilt.
    """
    from sentence_transformers import SentenceTransformer

    n_windows = len(train_ds)
    expected_dim = cfg.sentence_embedder_dim
    cache_path = cfg.sentence_embedder_cache_path

    if cache_path and os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if tuple(cache.shape) == (n_windows, expected_dim) and cache.dtype == torch.float16:
                print(f"[SentenceEmbedder] Loaded from {cache_path}")
                return cache
            print(
                f"[SentenceEmbedder] Shape mismatch ({tuple(cache.shape)} vs "
                f"{(n_windows, expected_dim)}), rebuilding..."
            )
        except Exception as e:
            print(f"[SentenceEmbedder] Could not load ({e}), rebuilding...")

    encoder = SentenceTransformer(cfg.sentence_embedder_model, device=cfg.device)
    actual_dim = encoder.get_sentence_embedding_dimension()
    if actual_dim != expected_dim:
        raise ValueError(
            f"cfg.sentence_embedder_model={cfg.sentence_embedder_model!r} produces "
            f"{actual_dim}-dim embeddings, but cfg.sentence_embedder_dim={expected_dim}. "
            f"Set cfg.sentence_embedder_dim={actual_dim} to match."
        )

    cache = torch.zeros(n_windows, expected_dim, dtype=torch.float16)
    batch_size = cfg.sentence_embedder_batch_size
    for start in tqdm(range(0, n_windows, batch_size), desc="Building sentence-embedder cache"):
        end = min(start + batch_size, n_windows)
        texts = [
            tokenizer.decode(train_ds[i][0], skip_special_tokens=True)
            for i in range(start, end)
        ]
        embs = encoder.encode(
            texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        cache[start:end] = embs.cpu().half()

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"[SentenceEmbedder] Saved to {cache_path}")

    return cache
