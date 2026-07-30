# shared_dataset.py
"""
Resolve a config to its pre-tokenized dataset cache entry, and load it.

Tokenization is *not* done here. Sweeps (parallel_experiments.py), the GPU
memory probe, and every experiment worker call require_dataset_cache(), which
raises with the exact build command if the cache is missing — building it is
a separate, explicit step (build_dataset_cache.py), so a GPU-scheduling node
never blocks on tokenizing 6B tokens.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import ExperimentConfig
from consts import DATASET_CACHE_DIR
from data import TokenizedCorpus, discover_domain_files
from tokenization import MANIFEST_NAME, read_manifest


class DatasetCacheMissing(FileNotFoundError):
    """Raised when a config's tokenized dataset has not been built yet."""


def dataset_signature(cfg: ExperimentConfig) -> dict:
    """Fields that change the *tokens written to disk* for cfg.

    Deliberately excluded, because they are applied at read time by
    TokenizedCorpus and so do not require re-tokenizing:
      - cfg.block            (window size over the token stream)
      - cfg.max_chunks       (how many windows are in scope)
      - cfg.dataset_proportions (per-domain rebalancing)

    Included, unlike the old signature: cfg.tokenizer_name. The cache stores
    raw token ids, so a different tokenizer is a genuinely different cache.
    """
    return {
        "tokenizer_name": cfg.tokenizer_name,
        "split_dataset": cfg.split_dataset,
        "split_column": cfg.split_column if cfg.split_dataset else "",
        "dataset_list": list(cfg.dataset_list),
        "max_documents": cfg.max_documents,
    }


def signature_hash(sig: dict) -> str:
    blob = json.dumps(sig, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_entry_dir(cfg: ExperimentConfig, cache_root: Path = DATASET_CACHE_DIR) -> Path:
    return Path(cache_root) / signature_hash(dataset_signature(cfg))


def require_dataset_cache(
    cfg: ExperimentConfig, cache_root: Path = DATASET_CACHE_DIR
) -> Path:
    """Return the cache entry dir for cfg, or raise telling the user to build it."""
    entry_dir = cache_entry_dir(cfg, cache_root)
    manifest = read_manifest(entry_dir)
    if manifest is None:
        sig = dataset_signature(cfg)
        raise DatasetCacheMissing(
            f"No tokenized dataset for this config.\n"
            f"  expected: {entry_dir / MANIFEST_NAME}\n"
            f"  signature: {json.dumps(sig, sort_keys=True)}\n\n"
            f"Build it once (CPU-only job, no GPU needed):\n"
            f"    python build_dataset_cache.py --config <your config>.yaml\n\n"
            f"{'(the directory exists but has no manifest.json -- a previous build was interrupted; re-run the command above to resume)' if entry_dir.exists() else ''}"
        )
    print(f"[dataset cache] hit {entry_dir.name} ({len(manifest['domains'])} domains)")
    return entry_dir


def load_dataset_cache(
    entry_dir: str | Path,
    cfg: ExperimentConfig,
    max_train: int | None = None,
    max_val: int | None = None,
    warm_cache: bool = True,
):
    """Load the cache entry as (train_ds, val_ds) TokenizedCorpus pairs.

    cfg supplies the read-time knobs (block, max_chunks, dataset_proportions,
    seed) that the cache deliberately does not bake in.

    max_train/max_val further truncate the window count (for the GPU memory
    probe, which only needs a handful of pool-sized batches).

    warm_cache sequentially reads every underlying .ds file once up front, so
    the training loop's shuffled per-window reads (real disk I/O only on
    first touch, since TokenizedCorpus mmaps files) hit the OS page cache
    instead of stalling on random access mid-epoch. Pass False when only a
    handful of windows will actually be read (e.g. the GPU memory probe with
    max_train/max_val set) -- there warming would read whole multi-GB domain
    files just to serve a few dozen samples.
    """
    entry_dir = Path(entry_dir)
    manifest = read_manifest(entry_dir)
    if manifest is None:
        raise DatasetCacheMissing(f"No {MANIFEST_NAME} in {entry_dir}")

    domains, folders = manifest["domains"], manifest["folders"]
    token_size = manifest["token_size"]

    # cfg.dataset_proportions is positional in cfg.dataset_list, while domain
    # ids follow the manifest's sorted domain list -- pair them up by name here
    # so the two orders can never be confused downstream.
    proportions = None
    if cfg.dataset_proportions and not cfg.split_dataset:
        if len(cfg.dataset_proportions) != len(cfg.dataset_list):
            raise ValueError(
                f"cfg.dataset_proportions has {len(cfg.dataset_proportions)} entries "
                f"but cfg.dataset_list has {len(cfg.dataset_list)}; they must be the "
                "same length and in the same order."
            )
        proportions = {
            name: float(p) for name, p in zip(cfg.dataset_list, cfg.dataset_proportions)
        }

    def build(split: str, cap: int | None) -> TokenizedCorpus:
        sources = discover_domain_files(entry_dir / split, domains, folders)
        max_chunks = cfg.max_chunks
        if cap is not None:
            max_chunks = cap if max_chunks == -1 else min(max_chunks, cap)
        return TokenizedCorpus(
            sources,
            block=cfg.block,
            token_size=token_size,
            domain_names=domains,
            max_chunks=max_chunks,
            # Proportions rebalance a hand-picked dataset_list; in
            # split_dataset mode the domains are auto-discovered and keep
            # their natural proportions, as before. Validation is never
            # rebalanced -- it should reflect the real distribution.
            proportions=proportions if split == "train" else None,
            seed=cfg.seed,
            warm_cache=warm_cache,
        )

    train_ds = build("train", max_train)
    val_ds = build("validation", max_val)
    print(
        f"[dataset cache] train: {len(train_ds):,} windows | "
        f"val: {len(val_ds):,} windows | block={cfg.block} | "
        f"domains: {', '.join(domains)}"
    )
    return train_ds, val_ds
