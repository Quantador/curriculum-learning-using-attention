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

Under DDP (cfg.world_size > 1): load_dataset_cache() is called identically on
every rank, so without sharding, every rank would redundantly re-encode the
*entire* dataset on its own single GPU -- world_size times the necessary work,
using only one of the world_size available GPUs at a time. Instead, each rank
encodes only its own ~1/world_size contiguous slice of windows (its own GPU,
in parallel with every other rank), then all ranks combine their slices via
dist.all_gather_object so every rank ends up with the identical, complete
[N, dim] tensor -- required since any rank's pool sampling can reference any
window index, not just the ones it personally encoded.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
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
    Identical on every rank under DDP (see module docstring).

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

    # Contiguous per-rank shard: rank i owns windows [i*chunk, (i+1)*chunk),
    # last rank absorbs the remainder (n_windows rarely divides evenly).
    # world_size=1 (the default) reduces to the whole dataset, unchanged.
    if cfg.world_size > 1:
        chunk = n_windows // cfg.world_size
        shard_start = cfg.rank * chunk
        shard_end = n_windows if cfg.rank == cfg.world_size - 1 else shard_start + chunk
    else:
        shard_start, shard_end = 0, n_windows

    local_cache = torch.zeros(shard_end - shard_start, expected_dim, dtype=torch.float16)
    batch_size = cfg.sentence_embedder_batch_size
    for start in tqdm(
        range(shard_start, shard_end, batch_size),
        desc=f"Building sentence-embedder cache (rank {cfg.rank}'s shard)",
        disable=cfg.rank != 0,
    ):
        end = min(start + batch_size, shard_end)
        texts = tokenizer.batch_decode(
            [train_ds[i][0] for i in range(start, end)], skip_special_tokens=True
        )
        embs = encoder.encode(
            texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        local_cache[start - shard_start : end - shard_start] = embs.cpu().half()

    if cfg.world_size > 1:
        gathered: list = [None] * cfg.world_size
        dist.all_gather_object(gathered, local_cache)
        cache = torch.cat(gathered, dim=0)
    else:
        cache = local_cache

    if cache_path and cfg.rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cache, cache_path)
        print(f"[SentenceEmbedder] Saved to {cache_path}")

    return cache
