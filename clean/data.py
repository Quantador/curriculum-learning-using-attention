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