# data.py
from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from tqdm import tqdm, tqdm
from config import Config


def get_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def _get_dataset_config(source: str, num_samples: int) -> dict:
    """
    Returns HuggingFace dataset loading arguments for a given source key.
    You can add/adjust these easily.
    """
    source = source.lower()

    if source == "tinystories":
        return dict(path="roneneldan/TinyStories", name=None, split="train[:{}]".format(num_samples))

    if source == "openwebtext2":
        # commonly used: Geralt-Targaryen/openwebtext2
        return dict(path="Geralt-Targaryen/openwebtext2", name=None, split="train[:{}]".format(num_samples))
    if source == "wikipedia":
        # stable fallback: wikipedia 20220301.en
        return dict(path="wikipedia", name="20220301.en", split="train[:{}]".format(num_samples))
    if source == "c4":
        # C4 is huge; for sweeps you may want to use 'en' or a smaller shard
        return dict(path="c4", name="en", split="train[:{}]".format(num_samples))
    if source == "fineweb_edu":
        # FineWeb-edu naming varies by mirror; keep this configurable
        # Try "HuggingFaceFW/fineweb-edu" first; if your env differs, update here.
        return dict(path="HuggingFaceFW/fineweb-edu", name=None, split="train[:{}]".format(num_samples))
    if source == "fineweb":
        return dict(path="HuggingFaceFW/fineweb", name=None, split="train[:{}]".format(num_samples))

    raise ValueError(f"Unknown source: {source}")


def load_text_stream(source: str, num_samples: int):
    cfg = _get_dataset_config(source, num_samples)
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"])
    # Most datasets use 'text'; Wikipedia uses 'text'; some use 'content'
    if "text" in ds.column_names:
        return ds["text"]
    if "content" in ds.column_names:
        return ds["content"]
    # last resort: join all string-like columns (rare)
    cols = [c for c in ds.column_names if ds[c].dtype == "string"]
    if not cols:
        raise ValueError(f"No usable text columns found for source={source}, columns={ds.column_names}")
    return ["\n".join([row[c] for c in cols if isinstance(row[c], str)]) for row in ds]




def load_mixed_dataset(cfg: Config):
    print("\n" + "=" * 70)
    print("Loading Mixed Difficulty Dataset")
    print("=" * 70)

    print(f"Loading TinyStories (easy)... target: {cfg.easy_samples} samples")
    easy_ds = load_dataset("roneneldan/TinyStories", split=f"train[0:{cfg.easy_samples}]")

    print(f"Loading OpenWebText (hard)... target: {cfg.hard_samples} samples")
    hard_ds = load_dataset("Geralt-Targaryen/openwebtext2", split=f"train[0:{cfg.hard_samples}]")

    print(f"✓ Loaded {len(easy_ds)} easy samples")
    print(f"✓ Loaded {len(hard_ds)} hard samples")

    return easy_ds, hard_ds


def tokenize_and_chunk(
    text: str,
    tokenizer: GPT2TokenizerFast,
    max_length: int,
) -> List[List[int]]:
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks: List[List[int]] = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i : i + max_length]
        if len(chunk) == max_length:
            chunks.append(chunk)
    return chunks


def make_mixed_chunks(
    split: str,
    cfg: Config,
    tokenizer: GPT2TokenizerFast,
) -> List[Tuple[List[int], int]]:
    if split == "train":
        easy_ds, hard_ds = load_mixed_dataset(cfg)

        print("\nTokenizing and chunking...")
        easy_chunks: List[List[int]] = []
        for item in tqdm(easy_ds):
            easy_chunks.extend(
                tokenize_and_chunk(item["text"], tokenizer, cfg.block + 1)
            )

        hard_chunks: List[List[int]] = []
        for item in tqdm(hard_ds):
            hard_chunks.extend(
                tokenize_and_chunk(item["text"], tokenizer, cfg.block + 1)
            )

        print(f"Raw chunks - Easy: {len(easy_chunks)}, Hard: {len(hard_chunks)}")

        target_easy_ratio = 0.7
        if len(easy_chunks) > 0 and len(hard_chunks) > 0:
            total_chunks = len(easy_chunks) + len(hard_chunks)
            current_easy_ratio = len(easy_chunks) / total_chunks
            print(
                f"⚠ Current ratio - Easy: {current_easy_ratio:.2f}, "
                f"Hard: {1 - current_easy_ratio:.2f}"
            )

            target_total = min(cfg.max_chunks, total_chunks)
            target_easy = int(target_total * target_easy_ratio)
            target_hard = target_total - target_easy

            if len(easy_chunks) >= target_easy:
                easy_sampled = random.sample(easy_chunks, target_easy)
            else:
                easy_sampled = random.choices(easy_chunks, k=target_easy)

            if len(hard_chunks) >= target_hard:
                hard_sampled = random.sample(hard_chunks, target_hard)
            else:
                hard_sampled = random.choices(hard_chunks, k=target_hard)

            print(
                f"✓ Rebalanced to {len(easy_sampled)} easy, "
                f"{len(hard_sampled)} hard"
            )
            print(
                "✓ New ratio - Easy: "
                f"{len(easy_sampled)/(len(easy_sampled)+len(hard_sampled)):.2f}"
            )
        else:
            easy_sampled = easy_chunks
            hard_sampled = hard_chunks

        easy_labeled = [(chunk, 0) for chunk in easy_sampled]
        hard_labeled = [(chunk, 1) for chunk in hard_sampled]

        all_chunks: List[Tuple[List[int], int]] = easy_labeled + hard_labeled
        random.shuffle(all_chunks)

        print(f"✓ Final dataset: {len(all_chunks)} chunks")
        return all_chunks

    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = tokenizer.eos_token.join(ds["text"])
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        L = (len(ids) // (cfg.block + 1)) * (cfg.block + 1)
        ids = ids[:L]
        chunks = [
            ids[i : i + cfg.block + 1] for i in range(0, L, cfg.block + 1)
        ]
        return [(chunk, -1) for chunk in chunks]
    
    
    
def make_mixed_chunks_gen(cfg, tokenizer):
    """
    Builds a mixed set of token chunks of length cfg.block+1 from:
      - cfg.easy_source (default tinystories)
      - cfg.hard_source (default openwebtext2)
    Mixed according to cfg.easy_ratio.
    """
    blockp1 = cfg.block + 1

    easy_text = load_text_stream(cfg.easy_source, cfg.easy_samples)
    hard_text = load_text_stream(cfg.hard_source, cfg.hard_samples)

    def tokenize_and_chunk(text_list, max_chunks=None):
        chunks = []
        for t in text_list:
            if not isinstance(t, str) or len(t) == 0:
                continue
            ids = tokenizer(t, add_special_tokens=False)["input_ids"]
            # chunk into block+1 sequences
            for i in range(0, len(ids) - blockp1 + 1, blockp1):
                chunks.append(ids[i:i+blockp1])
                if max_chunks is not None and len(chunks) >= max_chunks:
                    return chunks
        return chunks

    easy_chunks = tokenize_and_chunk(easy_text, max_chunks=cfg.max_easy_chunks)
    hard_chunks = tokenize_and_chunk(hard_text, max_chunks=cfg.max_hard_chunks)

    if cfg.max_chunks is not None:
        # cap total chunks if you already use cfg.max_chunks elsewhere
        max_total = cfg.max_chunks
    else:
        max_total = None

    # determine how many to take from each source
    if max_total is None:
        # take as much as possible while respecting ratio based on available hard/easy
        # choose total based on limiting source
        # total_easy = r*T, total_hard = (1-r)*T -> T <= easy/r and T <= hard/(1-r)
        r = float(cfg.easy_ratio)
        T1 = int(len(easy_chunks) / max(r, 1e-9))
        T2 = int(len(hard_chunks) / max(1.0 - r, 1e-9))
        T = min(T1, T2)
    else:
        T = max_total

    n_easy = int(round(cfg.easy_ratio * T))
    n_hard = T - n_easy

    # safety: clip to availability
    n_easy = min(n_easy, len(easy_chunks))
    n_hard = min(n_hard, len(hard_chunks))

    mixed = easy_chunks[:n_easy] + hard_chunks[:n_hard]
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(mixed)

    return mixed



class MixedLMDataset(Dataset):
    def __init__(self, labeled_chunks: List[Tuple[List[int], int]]):
        self.x = [
            torch.tensor(c[:-1], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.y = [
            torch.tensor(c[1:], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.difficulty = [d for _, d in labeled_chunks]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.difficulty[i]


def make_index_loader(ds_len: int, pool_size: int):
    order = list(range(ds_len))
    random.shuffle(order)
    for i in range(0, ds_len, pool_size):
        yield order[i : i + pool_size]