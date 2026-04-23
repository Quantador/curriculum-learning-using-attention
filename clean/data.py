# data.py
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from tqdm import tqdm
from config import Config


# Registry mapping HuggingFace dataset names to their split/text-column metadata.
# Easier datasets produce simpler, shorter text; harder ones contain dense or
# domain-specific language.
#
# Easy:   roneneldan/TinyStories, ajibawa-2023/Children-Stories-Collection,
#         Salesforce/wikitext
# Medium: Geralt-Targaryen/openwebtext2, HuggingFaceFW/fineweb-edu, allenai/c4
# Hard:   armanc/scientific_papers, CShorten/ML-ArXiv-Papers
# Unstructured (single-dataset): HuggingFaceFW/fineweb
DATASET_REGISTRY: dict[str, dict] = {
    "roneneldan/TinyStories":                   {"split": "train", "text_col": "text"},
    "ajibawa-2023/Children-Stories-Collection":  {"split": "train", "text_col": "text"},
    "Salesforce/wikitext":                      {"split": "train", "name": "wikitext-103-raw-v1", "text_col": "text"},
    "Geralt-Targaryen/openwebtext2":             {"split": "train", "text_col": "text"},
    "armanc/scientific_papers":                 {"split": "train", "text_col": "abstract"},
    "CShorten/ML-ArXiv-Papers":                 {"split": "train", "text_col": "abstract"},
    "HuggingFaceFW/fineweb-edu":                {"split": "train", "text_col": "text"},
    "HuggingFaceFW/fineweb":                    {"split": "train", "name": "sample-10BT", "text_col": "text"},
    "allenai/c4":                               {"split": "train", "name": "en", "text_col": "text"},
}


def load_dataset_by_name(name: str, n_samples: int):
    """Load n_samples from a registered HuggingFace dataset.

    Uses streaming=True so only the requested records are fetched - no need to
    download every shard just to slice the first N rows.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Add it to DATASET_REGISTRY in data.py.")
    meta = DATASET_REGISTRY[name]
    kwargs: dict = {"split": meta["split"], "streaming": True}
    if "name" in meta:
        kwargs["name"] = meta["name"]
    ds = load_dataset(name, **kwargs)
    return ds.take(n_samples), meta["text_col"]


def get_tokenizer() -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_mixed_dataset(cfg: Config):
    print("\n" + "=" * 70)
    print("Loading Mixed Difficulty Dataset")
    print("=" * 70)

    easy_name = getattr(cfg, "easy_dataset", "roneneldan/TinyStories")
    hard_name = getattr(cfg, "hard_dataset", "Geralt-Targaryen/openwebtext2")

    print(f"Loading {easy_name} (easy)... target: {cfg.easy_samples} samples")
    easy_ds, easy_col = load_dataset_by_name(easy_name, cfg.easy_samples)

    print(f"Loading {hard_name} (hard)... target: {cfg.hard_samples} samples")
    hard_ds, hard_col = load_dataset_by_name(hard_name, cfg.hard_samples)

    print(f"OK Requested {cfg.easy_samples} easy samples (streaming)")
    print(f"OK Requested {cfg.hard_samples} hard samples (streaming)")

    return easy_ds, easy_col, hard_ds, hard_col


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
        easy_ds, easy_col, hard_ds, hard_col = load_mixed_dataset(cfg)

        print("\nTokenizing and chunking...")
        easy_chunks: List[List[int]] = []
        for item in tqdm(easy_ds):
            easy_chunks.extend(
                tokenize_and_chunk(item[easy_col], tokenizer, cfg.block + 1)
            )

        hard_chunks: List[List[int]] = []
        for item in tqdm(hard_ds):
            hard_chunks.extend(
                tokenize_and_chunk(item[hard_col], tokenizer, cfg.block + 1)
            )

        print(f"Raw chunks - Easy: {len(easy_chunks)}, Hard: {len(hard_chunks)}")

        target_easy_ratio = getattr(cfg, "easy_proportion", 0.7)
        if len(easy_chunks) > 0 and len(hard_chunks) > 0:
            total_chunks = len(easy_chunks) + len(hard_chunks)
            current_easy_ratio = len(easy_chunks) / total_chunks
            print(
                f"WARNING: Current ratio - Easy: {current_easy_ratio:.2f}, "
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
                f"OK Rebalanced to {len(easy_sampled)} easy, "
                f"{len(hard_sampled)} hard"
            )
            print(
                "OK New ratio - Easy: "
                f"{len(easy_sampled)/(len(easy_sampled)+len(hard_sampled)):.2f}"
            )
        else:
            easy_sampled = easy_chunks
            hard_sampled = hard_chunks

        easy_labeled = [(chunk, 0) for chunk in easy_sampled]
        hard_labeled = [(chunk, 1) for chunk in hard_sampled]

        all_chunks: List[Tuple[List[int], int]] = easy_labeled + hard_labeled
        random.shuffle(all_chunks)

        print(f"OK Final dataset: {len(all_chunks)} chunks")
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


def load_dataset_with_embeddings(
    dataset_name: str,
    n_samples: int,
    text_col: str = "text",
    embedding_col: str = "embeddings",
) -> Tuple[List[str], List[torch.Tensor]]:
    """Load a HuggingFace dataset that has a pre-computed embedding column."""
    ds = load_dataset(dataset_name, split="train", streaming=True).take(n_samples)
    texts = []
    embeddings = []
    for row in ds:
        texts.append(row[text_col])
        emb = torch.tensor(row[embedding_col], dtype=torch.float32)
        # Some datasets (e.g. epfml/FineWeb-HQ) store one vector per sub-chunk,
        # yielding shape [n_chunks, dim]. Mean-pool to a single document vector.
        if emb.dim() == 2:
            emb = emb.mean(dim=0)
        embeddings.append(emb)
    return texts, embeddings


def make_single_chunks(
    cfg: Config,
    tokenizer: GPT2TokenizerFast,
) -> Tuple[
    List[Tuple[List[int], int]],
    List[Tuple[List[int], int]],
    Optional[List[torch.Tensor]],
    Optional[List[torch.Tensor]],
]:
    """
    Load a single dataset (no easy/hard split), tokenize, and split into
    train/val. All training difficulty labels are 0 (unlabeled).

    When cfg.use_external_embeddings is True, loads text + embeddings from
    cfg.external_embeddings_dataset (a HuggingFace dataset with 'text' and
    'embedding' columns). Document embeddings are replicated to all chunks
    produced from that document.

    Returns (train_chunks, val_chunks, train_embeddings, val_embeddings).
    Embedding lists are None when use_external_embeddings is False.
    """
    use_ext = getattr(cfg, "use_external_embeddings", False)
    n_samples = getattr(cfg, "single_dataset_samples", 120_000)
    val_split = getattr(cfg, "single_dataset_val_split", 0.05)

    print("\n" + "=" * 70)

    if use_ext:
        ext_dataset = getattr(cfg, "external_embeddings_dataset", "")
        if not ext_dataset:
            raise ValueError("cfg.external_embeddings_dataset must be set when use_external_embeddings=True")
        print(f"Loading Single Dataset with External Embeddings: {ext_dataset}")
        print("=" * 70)
        texts, doc_embeddings = load_dataset_with_embeddings(ext_dataset, n_samples)
        print(f"OK Loaded {len(texts)} documents")

        print("\nTokenizing and chunking...")
        all_chunks: List[List[int]] = []
        all_embeddings: List[torch.Tensor] = []
        for text, emb in tqdm(zip(texts, doc_embeddings), total=len(texts)):
            doc_chunks = tokenize_and_chunk(text, tokenizer, cfg.block + 1)
            for chunk in doc_chunks:
                all_chunks.append(chunk)
                all_embeddings.append(emb)
    else:
        single_name = getattr(cfg, "single_dataset", "HuggingFaceFW/fineweb")
        print(f"Loading Single Dataset: {single_name}")
        print("=" * 70)
        ds, text_col = load_dataset_by_name(single_name, n_samples)
        print(f"OK Streaming up to {n_samples} samples")

        print("\nTokenizing and chunking...")
        all_chunks = []
        for item in tqdm(ds):
            all_chunks.extend(tokenize_and_chunk(item[text_col], tokenizer, cfg.block + 1))

    combined = list(zip(all_chunks, all_embeddings)) if use_ext else [(c, None) for c in all_chunks]
    random.shuffle(combined)

    n_val = int(len(combined) * val_split)
    val_pairs   = combined[:n_val]
    train_pairs = combined[n_val:]

    val_chunks   = [(c, -1) for c, _ in val_pairs]
    train_chunks = [(c,  0) for c, _ in train_pairs]

    if use_ext:
        val_embs   = [e for _, e in val_pairs]
        train_embs = [e for _, e in train_pairs]
    else:
        val_embs = train_embs = None

    print(f"OK Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks")
    return train_chunks, val_chunks, train_embs, val_embs


class MixedLMDataset(Dataset):
    def __init__(
        self,
        labeled_chunks: List[Tuple[List[int], int]],
        embeddings: Optional[List[torch.Tensor]] = None,
    ):
        self.x = [
            torch.tensor(c[:-1], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.y = [
            torch.tensor(c[1:], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.difficulty = [d for _, d in labeled_chunks]
        # Pre-computed external embeddings aligned with each chunk (or None).
        # Access via dataset.embeddings[i] rather than __getitem__ so that
        # existing training loops unpacking (x, y, diff) don't break.
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.difficulty[i]


def make_index_loader(ds_len: int, pool_size: int):
    order = list(range(ds_len))
    random.shuffle(order)
    for i in range(0, ds_len, pool_size):
        yield order[i : i + pool_size]
