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
  get_tokenizer()   — tokenizer matching the student LM (GPT-2 BPE by default)
  TokenizedCorpus   — Dataset over the .ds files, yielding (x, y, domain_id)
  make_index_loader — yields shuffled pool-sized index batches
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
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


def make_index_loader(ds_len: int, pool_size: int):
    order = list(range(ds_len))
    random.shuffle(order)
    for i in range(0, ds_len, pool_size):
        yield order[i : i + pool_size]
