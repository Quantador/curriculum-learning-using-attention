# data.py
"""
Dataset loading, tokenisation, and chunking for curriculum learning.

Two dataset modes:
  - Mixed-difficulty: load an easy + a hard HuggingFace dataset, tokenise,
    chunk into (block+1)-token sequences, label them 0 (easy) / 1 (hard).
    Use make_mixed_chunks().
  - Single-dataset: one source, no difficulty split (all labels 0). Supports
    optional pre-computed external embeddings (e.g. epfml/FineWeb-HQ).
    Use make_single_chunks().

Difficulty label convention: 0 = easy, 1 = hard, -1 = validation (no label).

Validation always uses WikiText-2 regardless of training dataset config,
keeping the eval set fixed across all experiments for fair comparison.

Key exports:
  get_tokenizer()       — tokeniser matching the student LM (GPT-2 BPE by
                          default; pass a HF model name for other models)
  make_mixed_chunks()   — builds labelled train/val chunks for mixed mode
  make_single_chunks()  — builds chunks for single-dataset mode
  MixedLMDataset        — PyTorch Dataset yielding (x, y, difficulty) triples
  make_index_loader()   — yields shuffled pool-sized index batches
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2TokenizerFast, PreTrainedTokenizerBase

from tqdm import tqdm
from config import Config, ExperimentConfig


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
    "ajibawa-2023/Children-Stories-Collection": {"split": "train", "text_col": "text"},
    "Salesforce/wikitext":                      {"split": "train", "name": "wikitext-103-raw-v1", "text_col": "text"},
    "Geralt-Targaryen/openwebtext2":            {"split": "train", "text_col": "text"},
    "armanc/scientific_papers":                 {"split": "train", "text_col": "abstract"},
    "CShorten/ML-ArXiv-Papers":                 {"split": "train", "text_col": "abstract"},
    "HuggingFaceFW/fineweb-edu":                {"split": "train", "text_col": "text"},
    "HuggingFaceFW/fineweb":                    {"split": "train", "name": "sample-10BT", "text_col": "text"},
    "allenai/c4":                               {"split": "train", "name": "en", "text_col": "text"},
    "DKYoon/SlimPajama-6B":                     {"split": "train", "text_col": "text"}
}

def get_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerBase:
    """
    Load the tokenizer matching the student LM's vocabulary.

    Defaults to GPT-2 BPE (used by TinyGPT). Pass a HuggingFace model name
    (e.g. "Qwen/Qwen3-1.7B") to get the matching tokenizer instead — required
    whenever the token ids must line up with that model's embedding table.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

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

def load_dataset_with_embeddings(
    dataset_name: str,
    n_samples: int,
    text_col: str = "text",
    embedding_col: str = "embeddings",
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Load a HuggingFace dataset that has a pre-computed embedding column.

    Some datasets (e.g. epfml/FineWeb-HQ) store one embedding vector per
    sub-chunk of the document, yielding shape [n_sub_chunks, dim]. These
    are mean-pooled to a single document-level vector before being attached
    to each token chunk produced from that document.

    Returns (texts, embeddings) where embeddings[i] is a 1-D float32 tensor
    aligned with texts[i]. n_samples=-1 means no cap (stream the whole split).
    """
    ds = load_dataset(dataset_name, split="train", streaming=True)
    if n_samples != -1:
        ds = ds.take(n_samples)
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

def load_hf_datasets(cfg: ExperimentConfig, split = "train") -> list: # A list of (domain_name, texts, embeddings_or_None)
    if cfg.use_external_embeddings:
        # Backward-compat with the old make_single_chunks() external-embeddings
        # path: embeddings datasets like epfml/FineWeb-HQ only expose a 'train'
        # split, so there's no real "validation" split to request from HF.
        # Instead -- exactly like make_single_chunks() used to -- we stream the
        # whole (capped) dataset and carve off cfg.single_dataset_val_split of
        # it for validation. Since load_hf_datasets() has no state shared
        # between the separate "train" and "validation" calls make_chunks()
        # makes, the full stream is re-fetched on each call; a fixed (not
        # shuffled) split point keeps the two calls' outputs disjoint.
        assert len(cfg.dataset_list) <= 1, (
            "use_external_embeddings only supports a single dataset "
            "(cfg.external_embeddings_dataset), not cfg.dataset_list"
        )
        n_samples = cfg.max_chunks
        texts, embeddings = load_dataset_with_embeddings(cfg.external_embeddings_dataset, n_samples)
        n_val = int(len(texts) * cfg.single_dataset_val_split)
        if split == "train":
            texts, embeddings = texts[n_val:], embeddings[n_val:]
        else:
            texts, embeddings = texts[:n_val], embeddings[:n_val]

        print(f"OK Loaded {len(texts)} documents ({split}) from {cfg.external_embeddings_dataset}")
        return [(cfg.external_embeddings_dataset, texts, embeddings)]

    buckets: dict[str, list[str]] = {}
    total = 0

    if cfg.split_dataset:
        assert (len(cfg.dataset_list) == 1), "If you want to split a dataset, make sure it is just one"
        assert cfg.split_column, "cfg.split_column must be set when cfg.split_dataset=True"
        name = cfg.dataset_list[0]

        text_col = DATASET_REGISTRY.get(name).get("text_col")
        ds = load_dataset(name, split = split,  streaming = True) # Make sure you load the correct split

        print("\n" + "=" * 70)
        print(f"Loading {name}, splitting by column '{cfg.split_column}'")
        print(f"(auto-discovering domains, up to {cfg.max_chunks} rows total)")
        print("=" * 70)

        for row in tqdm(ds, total=cfg.max_chunks):
            if cfg.max_chunks != -1 and total >= cfg.max_chunks:
                break
            domain = row[cfg.split_column]
            buckets.setdefault(domain, []).append(row[text_col])
            total += 1

        return [(name, texts, None) for name, texts in buckets.items()]
    else:
        for name in cfg.dataset_list:
            text_col = DATASET_REGISTRY.get(name).get("text_col")
            ds = load_dataset(name, split = split,  streaming = True) # Make sure you load the correct split
            for row in tqdm(ds, total=cfg.max_chunks):
                if cfg.max_chunks != -1 and total >= cfg.max_chunks:
                    break
                buckets.setdefault(name, []).append(row[text_col])
                total += 1

    print(
        f"OK Found {len(buckets)} different datasets: "
        + ", ".join(f"{name} ({len(texts)})" for name, texts in buckets.items())
    )

    return [(name, texts, None) for name, texts in buckets.items()]


def chunk_datasets(
    datasets: list,
    cfg: ExperimentConfig,
    tokenizer: GPT2TokenizerFast,
) -> Tuple[List[Tuple[List[int], int]], Optional[List[torch.Tensor]]]:
    """Tokenize/chunk each (domain_name, texts, embeddings) triple from
    load_hf_datasets() and label every resulting chunk with its domain's
    index (0..N-1, in the order domains appear in `datasets`) -- the same
    (chunk, label) shape make_mixed_chunks() used to produce, so the result
    is ready to hand straight to MixedLMDataset(chunks, embeddings=embs).

    When cfg.split_dataset is False, each domain's chunk count is rebalanced
    to match cfg.dataset_proportions (oversampling/undersampling exactly like
    the old easy/hard rebalancing), capped at cfg.max_chunks total chunks
    (cfg.max_chunks=-1 means no cap). When cfg.split_dataset is True, no
    rebalancing happens -- domains already keep their natural proportions
    from load_hf_datasets() (see its docstring), since dataset_proportions
    doesn't apply to an auto-discovered domain list.

    When a domain carries per-text embeddings (cfg.use_external_embeddings),
    each chunk produced from a text inherits that text's embedding -- a
    document that splits into multiple chunks replicates its embedding
    across all of them, exactly like make_single_chunks() used to. Returns
    (chunks, embeddings) where embeddings is None unless at least one domain
    had embeddings attached.
    """
    print("\nTokenizing and chunking...")
    per_domain_chunks: List[List[List[int]]] = []
    per_domain_embs: List[Optional[List[torch.Tensor]]] = []
    for name, texts, embeddings in datasets:
        chunks: List[List[int]] = []
        embs: Optional[List[torch.Tensor]] = [] if embeddings is not None else None
        text_embs = zip(texts, embeddings) if embeddings is not None else zip(texts, [None] * len(texts))
        for text, emb in tqdm(text_embs, total=len(texts), desc=name):
            doc_chunks = tokenize_and_chunk(text, tokenizer, cfg.block + 1)
            chunks.extend(doc_chunks)
            if embs is not None:
                embs.extend([emb] * len(doc_chunks))
        per_domain_chunks.append(chunks)
        per_domain_embs.append(embs)

    print(
        "Raw chunks - "
        + ", ".join(f"{name}: {len(c)}" for (name, _, _), c in zip(datasets, per_domain_chunks))
    )

    if not cfg.split_dataset and cfg.dataset_proportions:
        if len(cfg.dataset_proportions) != len(datasets):
            raise ValueError(
                f"cfg.dataset_proportions has {len(cfg.dataset_proportions)} entries "
                f"but {len(datasets)} datasets were loaded from cfg.dataset_list; "
                "they must be the same length and in the same order."
            )
        total_chunks = sum(len(c) for c in per_domain_chunks)
        target_total = total_chunks if cfg.max_chunks == -1 else min(cfg.max_chunks, total_chunks)

        sampled_chunks = []
        sampled_embs = []
        for (name, _, _), chunks, embs, proportion in zip(
            datasets, per_domain_chunks, per_domain_embs, cfg.dataset_proportions
        ):
            target_n = int(target_total * float(proportion))
            # Sample by index (not by value) so a domain's embeddings, if any,
            # can be subsampled in lockstep with its chunks.
            if len(chunks) >= target_n:
                idx = random.sample(range(len(chunks)), target_n)
            else:
                idx = random.choices(range(len(chunks)), k=target_n)
            sampled_chunks.append([chunks[i] for i in idx])
            sampled_embs.append([embs[i] for i in idx] if embs is not None else None)
            print(f"OK Rebalanced {name} to {target_n} chunks (target {target_n})")
    else:
        sampled_chunks = per_domain_chunks
        sampled_embs = per_domain_embs

    has_embeddings = any(embs is not None for embs in sampled_embs)

    all_chunks: List[Tuple[List[int], int]] = []
    all_embs: Optional[List[torch.Tensor]] = [] if has_embeddings else None
    for domain_id, chunks in enumerate(sampled_chunks):
        embs = sampled_embs[domain_id]
        for i, chunk in enumerate(chunks):
            all_chunks.append((chunk, domain_id))
            if has_embeddings:
                # Domains without embeddings (mixed with an embedded domain)
                # contribute None placeholders so all_chunks/all_embs stay
                # aligned 1:1.
                all_embs.append(embs[i] if embs is not None else None)

    if has_embeddings:
        combined = list(zip(all_chunks, all_embs))
        random.shuffle(combined)
        all_chunks = [c for c, _ in combined]
        all_embs = [e for _, e in combined]
    else:
        random.shuffle(all_chunks)

    print(f"OK Final dataset: {len(all_chunks)} chunks across {len(datasets)} domains")
    return all_chunks, all_embs


def make_chunks(cfg: ExperimentConfig,
    tokenizer: GPT2TokenizerFast):

    train_datasets = load_hf_datasets(cfg, "train")
    validation_datasets = load_hf_datasets(cfg, "validation")

    train_chunks, train_embs = chunk_datasets(train_datasets, cfg, tokenizer)
    validation_chunks, val_embs = chunk_datasets(validation_datasets, cfg, tokenizer)

    # domain_id -> name, in the same order chunk_datasets() assigned ids --
    # used to label per-domain metrics/plots with real names instead of bare
    # ids. Tracked separately for train/val: in cfg.split_dataset mode,
    # domains are auto-discovered per stream, so the validation split isn't
    # guaranteed to discover the same domains in the same order as train.
    train_domain_names = [name for name, _, _ in train_datasets]
    val_domain_names = [name for name, _, _ in validation_datasets]

    return train_chunks, validation_chunks, train_embs, val_embs, train_domain_names, val_domain_names

class MixedLMDataset(Dataset):
    def __init__(
        self,
        labeled_chunks: List[Tuple[List[int], int]],
        embeddings: Optional[List[torch.Tensor]] = None,
        domain_names: Optional[List[str]] = None,
    ):
        self.x = [
            torch.tensor(c[:-1], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.y = [
            torch.tensor(c[1:], dtype=torch.long) for c, _ in labeled_chunks
        ]
        self.domains = [d for _, d in labeled_chunks]
        self.embeddings = embeddings
        # domain_id -> name (e.g. domain_names[0] == "wikipedia"), for
        # labeling per-domain metrics with real names. None when unknown to
        # the caller (e.g. validation sets, or older cached datasets).
        self.domain_names = domain_names

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i], self.domains[i] # In certain cases the difficulty can match the domain


def make_index_loader(ds_len: int, pool_size: int):
    order = list(range(ds_len))
    random.shuffle(order)
    for i in range(0, ds_len, pool_size):
        yield order[i : i + pool_size]
