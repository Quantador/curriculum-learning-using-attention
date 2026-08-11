# data.py
"""
Reading side of the dataset pipeline: turn a datatrove-tokenized cache into a
PyTorch Dataset of (x, y, domain_id) triples.

Nothing here tokenizes. The cache is built ahead of time by
build_dataset_cache.py (see tokenization.py); training and sweeps only ever
read it, and error out if it is missing (utils.shared_dataset).

Layout assumed on disk -- one folder per domain, so mapping a chunk back to
its domain is just "which folder is this .ds file in":

    <entry_dir>/<split>/<domain folder>/*.ds

Key exports:
  get_tokenizer()      — tokenizer matching the student LM (GPT-2 BPE by default)
  TokenizedCorpus      — Dataset over the .ds files, yielding (x, y, domain_id)
  make_pool_loader()   — DataLoader of shuffled (idx, x, y, domain) pools, for
                         the curriculum loops that need every pool candidate's
                         data (rl_training.train_router_experiments/train_aux_baseline)
  make_baseline_loader() — DataLoader of (idx, x, y, domain) batches drawn by
                         PooledBatchSampler, for training.train_baseline()
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerBase:
    """
    Load the tokenizer matching the student LM's vocabulary.

    Defaults to GPT-2 BPE (used by TinyGPT). Pass a HuggingFace model name
    (e.g. "Qwen/Qwen3-1.7B") to get the matching tokenizer instead — required
    whenever the token ids must line up with that model's embedding table, and
    it must match the cfg.tokenizer_name the cache was built with.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def discover_domain_files(
    split_dir: Path, domains: Sequence[str], folders: dict[str, str]
) -> List[Tuple[Path, int]]:
    """(file, domain_id) for every .ds file under split_dir.

    domain_id is the index of the domain in `domains`, which the manifest
    fixes once for all splits so an id means the same thing in train and val.
    Domains with no files (present in one split but not another) contribute
    nothing and are simply skipped.
    """
    sources: List[Tuple[Path, int]] = []
    for domain_id, domain in enumerate(domains):
        folder = split_dir / folders[domain]
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.ds")):
            if path.stat().st_size > 0:
                sources.append((path, domain_id))
    return sources


class TokenizedCorpus(Dataset):
    """Fixed-length windows over datatrove .ds files, labelled by domain.

    Each .ds file is a flat little-endian stream of token ids with documents
    separated by EOS, which is read as consecutive non-overlapping windows of
    block+1 tokens -- x = window[:-1], y = window[1:]. Unlike the old
    per-document chunking, windows pack across document boundaries, so no
    tokens are dropped as a short tail.

    Files are memory-mapped lazily on first access rather than loaded into
    RAM: a full SlimPajama-6B cache is ~12 GB of token ids, and several
    experiment workers run concurrently against the same cache, so letting
    the OS page cache hold it once beats each process holding its own copy.

    Windows are read in shuffled order (see _select()), so consecutive
    __getitem__ calls jump to essentially random offsets across files --
    mmap avoids re-opening a file on every access (the memmap object itself
    is cached in self._mmaps), but each *new* page touched under that access
    pattern is still a real page fault / disk read the first time, and random
    order defeats OS readahead. warm_cache=True (the default) pays for this
    upfront with one sequential read of every file at construction time --
    much cheaper than scattered random reads -- so the random-order reads
    later hit the page cache instead of stalling mid-epoch.

    max_chunks / proportions are applied here, at read time, by choosing which
    windows are in scope -- they are not baked into the cache, so changing
    either does not require re-tokenizing.
    """

    def __init__(
        self,
        sources: Sequence[Tuple[Path, int]],
        block: int,
        token_size: int,
        domain_names: Optional[List[str]] = None,
        max_chunks: int = -1,
        proportions: Optional[dict] = None,
        seed: int = 0,
        warm_cache: bool = True,
    ):
        if not sources:
            raise ValueError("No .ds files found -- the tokenized cache is empty.")
        if token_size not in (2, 4):
            raise ValueError(f"token_size must be 2 or 4, got {token_size}")

        self.window = block + 1
        self.block = block
        # domain_id -> name, for labelling per-domain metrics and plots. Set
        # before _select(), which resolves proportions by domain name.
        self.domain_names = domain_names
        self._dtype = np.uint16 if token_size == 2 else np.uint32
        self._paths = [str(p) for p, _ in sources]
        self._domain_of_file = np.array([d for _, d in sources], dtype=np.int64)
        self._mmaps: List[Optional[np.memmap]] = [None] * len(self._paths)

        # Windows per file, and the running total so a global window id can be
        # resolved back to (file, offset) with one searchsorted.
        counts = [
            Path(p).stat().st_size // token_size // self.window for p in self._paths
        ]
        self._cum = np.cumsum([0] + counts, dtype=np.int64)

        self._sel = self._select(counts, max_chunks, proportions, seed)

        if warm_cache:
            self._warm_cache()

        # Pre-computed per-window embeddings, concatenated onto router
        # features by the training loops when set (they all branch on
        # `train_ds.embeddings is not None`). None by default -- the
        # use_external_embeddings dataset-column path is not supported by the
        # datatrove path (see tokenization.plan_jobs). Populated after
        # construction by utils.shared_dataset.load_dataset_cache() when
        # cfg.sentence_embedder_model is set (see utils.sentence_embedder).
        self.embeddings = None

    def _warm_cache(self) -> None:
        total_bytes = sum(Path(p).stat().st_size for p in self._paths)
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="warming page cache",
            leave=False,
        ) as pbar:
            for path in self._paths:
                with open(path, "rb") as f:
                    while chunk := f.read(1 << 20):  # 1 MiB: sequential, readahead-friendly
                        pbar.update(len(chunk))

    # -- selection ---------------------------------------------------------
    def _windows_by_domain(self, counts: Sequence[int]) -> dict[int, np.ndarray]:
        by_domain: dict[int, list[np.ndarray]] = {}
        for f, n in enumerate(counts):
            if n == 0:
                continue
            domain_id = int(self._domain_of_file[f])
            ids = np.arange(self._cum[f], self._cum[f + 1], dtype=np.int64)
            by_domain.setdefault(domain_id, []).append(ids)
        return {d: np.concatenate(parts) for d, parts in by_domain.items()}

    def _select(
        self,
        counts: Sequence[int],
        max_chunks: int,
        proportions: Optional[dict],
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        by_domain = self._windows_by_domain(counts)
        total = int(self._cum[-1])

        if proportions:
            # Keyed by domain *name*, not position: cfg.dataset_proportions is
            # written in cfg.dataset_list order while domain ids follow the
            # manifest's sorted domain list, so matching by index would
            # silently swap two domains' proportions.
            if self.domain_names is None:
                raise ValueError("proportions require domain_names")
            unknown = set(proportions) - set(self.domain_names)
            if unknown:
                raise ValueError(
                    f"dataset_proportions names domains not in this cache: {sorted(unknown)}; "
                    f"cache holds {list(self.domain_names)}"
                )
            target_total = total if max_chunks == -1 else min(max_chunks, total)
            parts = []
            for domain_id, name in enumerate(self.domain_names):
                ids = by_domain.get(domain_id)
                if ids is None or len(ids) == 0:
                    continue
                target_n = int(target_total * float(proportions.get(name, 0.0)))
                # Oversample a domain that is short of its target, exactly as
                # the old chunk_datasets() rebalancing did.
                replace = target_n > len(ids)
                parts.append(rng.choice(ids, size=target_n, replace=replace))
            selected = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        else:
            selected = np.arange(total, dtype=np.int64)
            if max_chunks != -1 and total > max_chunks:
                # Uniform subsample keeps the domains' natural proportions,
                # which is the point of cfg.split_dataset mode.
                selected = rng.choice(selected, size=max_chunks, replace=False)

        rng.shuffle(selected)
        return selected

    # -- reading -----------------------------------------------------------
    def _mmap(self, file_idx: int) -> np.memmap:
        mm = self._mmaps[file_idx]
        if mm is None:
            mm = np.memmap(self._paths[file_idx], dtype=self._dtype, mode="r")
            self._mmaps[file_idx] = mm
        return mm

    def __getstate__(self):
        # Open memmaps do not survive pickling (DataLoader workers, spawn).
        state = self.__dict__.copy()
        state["_mmaps"] = [None] * len(self._paths)
        return state

    def __len__(self) -> int:
        return len(self._sel)

    def domain_of(self, i: int) -> int:
        gid = int(self._sel[i])
        return int(self._domain_of_file[int(np.searchsorted(self._cum, gid, side="right")) - 1])

    def __getitem__(self, i: int):
        gid = int(self._sel[i])
        file_idx = int(np.searchsorted(self._cum, gid, side="right")) - 1
        offset = (gid - int(self._cum[file_idx])) * self.window
        window = self._mmap(file_idx)[offset : offset + self.window]
        ids = torch.from_numpy(window.astype(np.int64))
        return ids[:-1], ids[1:], int(self._domain_of_file[file_idx])


class _IndexedDataset(Dataset):
    """Wraps a Dataset so each item also carries its own index.

    The curriculum training loops track selected/covered samples by index
    into the underlying dataset (CoverageTracker, DiversityTracker, the
    feature cache in rl_training.build_feature_cache, ...), which a plain
    DataLoader batch has no way to report back -- shuffling happens inside
    the Sampler, invisible to __getitem__.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y, domain = self.base[i]
        return i, x, y, domain


def make_pool_loader(
    ds: Dataset,
    pool_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    DataLoader yielding shuffled, non-overlapping (idx, x, y, domain) pools of
    pool_size candidates -- one per curriculum-learning step in
    rl_training.train_router_experiments/train_aux_baseline, which need every
    pool candidate's data (router feature extraction runs over the whole
    pool, not just the eventually-selected batch).

    With num_workers > 0, the next pool is gathered on a worker process while
    the GPU is still busy with the current one's feature extraction + LM
    forward/backward -- previously this CPU-side gather (mmap reads +
    torch.stack) was fully serialized with GPU compute every step.

    Reusable across epochs: build it once and iterate repeatedly
    (`for epoch in ...: for pool in loader: ...`) rather than rebuilding it
    every epoch -- persistent_workers keeps worker processes alive between
    epochs instead of paying fork/mmap-reopen cost on every one. When
    world_size > 1, call `loader.sampler.set_epoch(epoch)` before each epoch's
    iteration (DistributedSampler reshuffles deterministically from `seed +
    epoch`, otherwise every rank -- and every epoch -- would see the same
    order); the single-process RandomSampler used when world_size == 1
    reshuffles on every fresh `for ... in loader` automatically and has no
    such method.

    drop_last=True: a trailing partial pool (< pool_size) is skipped. At the
    max_chunks scale these configs run at (hundreds of thousands of windows
    vs. a pool_size in the low hundreds), that drops a negligible fraction of
    an epoch. Under DDP, DistributedSampler's own drop_last=True first
    truncates the dataset to a multiple of world_size, so every rank gets
    exactly len(ds) // world_size candidates and therefore the same number of
    pools per epoch -- required so every rank issues the same number of
    DDP-synchronizing .backward() calls (a mismatch would hang NCCL).
    """
    dataset = _IndexedDataset(ds)
    sampler = None 
    shuffle = True 
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=seed,
        )
        shuffle = None 
    
    return DataLoader(
        dataset,
        batch_size=pool_size,
        shuffle=shuffle,
        sampler = sampler, 
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


class PooledBatchSampler(Sampler):
    """
    Batch sampler backing training.train_baseline()'s "pool" semantics:
    partition a shuffled permutation of range(ds_len) into non-overlapping
    pools of pool_size, then yield a uniformly random batch_size-sized
    subsample of each pool -- same distribution as the old
    make_index_loader() + inline random.sample(pool_indices, cfg.batch), but
    as a batch_sampler so DataLoader only ever fetches the batch_size samples
    actually used, not all pool_size candidates (unlike make_pool_loader()
    above, the random baseline never looks at the rest of the pool).

    Exists so the random baseline sees the same per-epoch step count and
    token budget as the router-based training loops (which also update on
    only batch_size out of every pool_size candidates) -- pool_size, not
    batch_size, is what should set "how much of the dataset counts as an
    epoch" for a fair comparison.

    DDP sharding (world_size > 1): every rank must partition the pools
    identically and then take a disjoint slice, or ranks would train on
    overlapping data and/or issue different numbers of DDP-synchronizing
    .backward() calls per epoch (hanging NCCL). So the shuffle uses a local
    random.Random(seed + epoch) instead of the shared global `random` module
    -- every rank computes the exact same shuffled pool partition from that
    seed, then rank-only strides over whole pools (pools[rank::world_size],
    first truncated to a multiple of world_size so every rank gets the same
    count) -- the same strided-sharding principle as a DistributedSampler,
    just at pool granularity so make_pool_loader() and this class agree on
    "how much of the dataset one epoch means." The final per-pool subsample
    uses a separate, rank-dependent RNG since that step doesn't need
    cross-rank agreement. With world_size=1 (the default) this reduces
    exactly to the old single-process behaviour, just with a local RNG
    instead of the global one.

    Call set_epoch(epoch) before each epoch's iteration, mirroring
    DistributedSampler's API, so pools reshuffle across epochs instead of
    repeating -- required (not just nice-to-have) under DDP, since a local
    RNG has no other source of cross-epoch variation the way the old
    global-`random`-module version implicitly had.
    """

    def __init__(
        self,
        ds_len: int,
        pool_size: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ):
        self.ds_len = ds_len
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        shared_rng = random.Random(self.seed + self.epoch)
        order = list(range(self.ds_len))
        shared_rng.shuffle(order)
        pools = [
            order[start : start + self.pool_size]
            for start in range(0, self.ds_len - self.pool_size + 1, self.pool_size)
        ]
        n_usable = (len(pools) // self.world_size) * self.world_size
        local_pools = pools[self.rank : n_usable : self.world_size]

        # Distinct from shared_rng's seed (which every rank must agree on) --
        # offsets chosen simply to avoid collisions between epoch/rank pairs,
        # not for any cryptographic property.
        local_rng = random.Random(self.seed + self.epoch * 1_000_003 + self.rank)
        for pool in local_pools:
            yield local_rng.sample(pool, self.batch_size)

    def __len__(self) -> int:
        n_pools = self.ds_len // self.pool_size
        return n_pools // self.world_size


def make_baseline_loader(
    ds: Dataset,
    pool_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
) -> DataLoader:
    """DataLoader for training.train_baseline(); see PooledBatchSampler.
    Reusable across epochs the same way as make_pool_loader() (persistent
    workers between epochs); call
    `loader.batch_sampler.set_epoch(epoch)` before each epoch's iteration."""
    return DataLoader(
        _IndexedDataset(ds),
        batch_sampler=PooledBatchSampler(
            len(ds), pool_size, batch_size, rank=rank, world_size=world_size, seed=seed,
        ),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
