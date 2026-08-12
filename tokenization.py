# tokenization.py
"""
Datatrove-based tokenization of a config's dataset(s) into a per-domain
on-disk cache.

This is the *build* half of the dataset pipeline; the *read* half lives in
data.TokenizedCorpus / utils.shared_dataset. The two halves are deliberately
separate processes: training never tokenizes. A sweep that finds no cache for
its config errors out and tells you to run build_dataset_cache.py first (see
utils.shared_dataset.require_dataset_cache).

Layout produced under <cache_root>/<signature hash>/:

    manifest.json                       # written last -- its presence means "complete"
    train/<domain folder>/*.ds          # + .ds.index, .ds.metadata (datatrove)
    validation/<domain folder>/*.ds
    logs/<split>/<domain>/              # datatrove executor logs + completions

One folder per domain is the whole trick: the domain -> chunk mapping that
data.py used to carry around in memory as (chunk, domain_id) tuples is now
just "which folder did this .ds file come from", resolved once at load time.

Two modes, mirroring the old data.load_hf_datasets():

  cfg.split_dataset=False -- one domain per entry in cfg.dataset_list. Each
    dataset gets its own executor run reading only that dataset.

  cfg.split_dataset=True -- a single dataset split by cfg.split_column (e.g.
    SlimPajama's "meta" -> redpajama_set_name). Domains are auto-discovered by
    streaming a sample of the split first, then one executor run per domain
    filters the stream down to that domain, exactly as testing_datatrove.py
    did with its hardcoded KNOWN_DOMAINS.

Note on cost: the split_dataset path re-streams the source once per domain
(~7x for SlimPajama). It is the simplest thing that reuses a stock
DocumentTokenizer unmodified. If the redundant reads ever become the
bottleneck, the alternative is a single pass writing per-domain JSONL via a
templated JsonlWriter output_filename, then tokenizing those local shards.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable

from datasets import load_dataset
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.utils.tokenization import load_tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

from config import ExperimentConfig
from consts import DATASET_REGISTRY

# datatrove shards work by task; each task reads one shard of the streaming
# dataset, so more tasks than the dataset has shards just yields idle tasks
# (see HuggingFaceDatasetReader._get_dataset_shard). build_tokenized_cache()
# caps tasks at the split's n_shards for that reason.
DEFAULT_TASKS = 128
DEFAULT_WORKERS = 16

# Rows streamed when auto-discovering domains for cfg.split_dataset mode.
# Only needs to be large enough to see every domain at least once.
DOMAIN_DISCOVERY_ROWS = 100_000

MANIFEST_NAME = "manifest.json"

# Validation always uses WikiText-2, not the configured training dataset(s) --
# keeps the eval signal identical across all experiment variants regardless
# of cfg.dataset_list / cfg.split_dataset.
VALIDATION_PATH = "Salesforce/wikitext"
VALIDATION_DATASET_NAME = "wikitext-2-raw-v1"


# --------------------------------------------------------------------------
# dataset / domain plumbing
# --------------------------------------------------------------------------
def registry_entry(path: str) -> dict:
    entry = DATASET_REGISTRY.get(path)
    if entry is None:
        raise KeyError(
            f"{path!r} is not in DATASET_REGISTRY (consts.py). Add an entry with "
            f"its 'text_col' (and 'name' if the dataset needs a config name) first."
        )
    return entry


def dataset_options(path: str, split: str) -> dict:
    """load_dataset kwargs for `path`, as datatrove's reader wants them."""
    entry = registry_entry(path)
    options = {"split": split}
    if entry.get("name"):
        options["name"] = entry["name"]
    return options


def domain_folder(name: str) -> str:
    """Filesystem-safe folder name for a domain.

    Dataset paths carry slashes ("DKYoon/SlimPajama-6B") and discovered domain
    values can carry anything, so everything outside [A-Za-z0-9._-] collapses
    to '_'. build_tokenized_cache() asserts the mapping stays injective, and
    the manifest keeps the real name for labelling metrics.
    """
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._-")
    return safe or "unnamed"


def extract_domain(value) -> str:
    """Normalise a split_column cell to a domain string.

    SlimPajama's "meta" column is a dict ({"redpajama_set_name": ...}) rather
    than a bare string; other datasets use a plain string column.
    """
    if isinstance(value, dict):
        for key in ("redpajama_set_name", "domain", "source", "subset"):
            if key in value:
                return str(value[key])
        raise ValueError(
            f"split_column holds a dict with no recognised domain key: {sorted(value)}"
        )
    return str(value)


def discover_domains(
    path: str, split: str, split_column: str, max_rows: int = DOMAIN_DISCOVERY_ROWS
) -> list[str]:
    """Stream `max_rows` of the split and return the distinct domains, sorted.

    Sorted (not first-seen) order so domain ids are stable no matter how many
    rows the discovery pass happens to look at.
    """
    ds = load_dataset(registry_entry(path).get("repo_id", path), split=split, streaming=True, **{
        k: v for k, v in dataset_options(path, split).items() if k != "split"
    })
    domains: set[str] = set()
    for i, row in enumerate(tqdm(ds, total=max_rows, desc=f"discovering domains ({split})")):
        if max_rows != -1 and i >= max_rows:
            break
        domains.add(extract_domain(row[split_column]))
    return sorted(domains)


def n_shards_for(path: str, split: str) -> int:
    ds = load_dataset(registry_entry(path).get("repo_id", path), streaming=True, **dataset_options(path, split))
    return max(1, ds.n_shards)


def make_adapter(text_col: str, split_column: str | None) -> Callable:
    """Reader adapter mapping a source row to a datatrove Document dict.

    Pulls `text_col` into "text" and, in split mode, the domain into
    metadata["domain"] so a LambdaFilter downstream can select on it.
    """

    def adapter(self, data: dict, path: str, id_in_file):
        metadata = {}
        if split_column:
            metadata["domain"] = extract_domain(data.get(split_column))
        return {
            "text": data.get(text_col) or "",
            "id": f"{path}/{id_in_file}",
            "metadata": metadata,
        }

    return adapter


def make_domain_filter(domain: str) -> LambdaFilter:
    return LambdaFilter(lambda doc, _d=domain: doc.metadata.get("domain") == _d)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
def plan_jobs(cfg: ExperimentConfig, split: str) -> list[dict]:
    """One tokenization job per domain for `split`.

    Each job is {domain, path, registry_key, options, text_col, split_column}
    -- everything build_tokenized_cache() needs to assemble an executor
    pipeline. `path` is the resolved repo id (what load_dataset()/
    HuggingFaceDatasetReader() actually need); `registry_key` is the
    DATASET_REGISTRY key it came from (what n_shards_for()/registry_entry()
    need) -- the two differ whenever a registry entry sets "repo_id" to point
    a friendly key at a different HF config/subset of the same repo.
    """
    if cfg.use_external_embeddings:
        raise NotImplementedError(
            "cfg.use_external_embeddings is not supported by the datatrove "
            "tokenization path: DocumentTokenizer writes token ids only, so a "
            "per-document embedding column has nowhere to go."
        )
    if split == "validation":
        return [
            {
                "domain": VALIDATION_PATH,
                "path": VALIDATION_PATH,
                "registry_key": VALIDATION_PATH,
                "options": {"split": split, "name": VALIDATION_DATASET_NAME},
                "text_col": "text",
                "split_column": None,
            }
        ]

    if not cfg.dataset_list:
        raise ValueError("cfg.dataset_list is empty -- nothing to tokenize.")

    if cfg.split_dataset:
        if len(cfg.dataset_list) != 1:
            raise ValueError(
                "cfg.split_dataset=True splits a single source dataset, but "
                f"cfg.dataset_list has {len(cfg.dataset_list)} entries."
            )
        if not cfg.split_column:
            raise ValueError("cfg.split_column must be set when cfg.split_dataset=True")

        path = cfg.dataset_list[0]
        text_col = registry_entry(path)["text_col"]
        domains = discover_domains(path, split, cfg.split_column, cfg.domain_discovery_rows)
        if not domains:
            raise RuntimeError(f"No domains discovered in {path} ({split}).")
        print(f"[{split}] discovered {len(domains)} domains: {', '.join(domains)}")
        repo_id = registry_entry(path).get("repo_id", path)
        return [
            {
                "domain": domain,
                "path": repo_id,
                "registry_key": path,
                "options": dataset_options(path, split),
                "text_col": text_col,
                "split_column": cfg.split_column,
            }
            for domain in domains
        ]

    return [
        {
            "domain": path,
            "path": registry_entry(path).get("repo_id", path),
            "registry_key": path,
            "options": dataset_options(path, split),
            "text_col": registry_entry(path)["text_col"],
            "split_column": None,
        }
        for path in cfg.dataset_list
    ]


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def token_size_for(tokenizer_name: str) -> int:
    """Bytes per token id on disk -- must match what DocumentTokenizer wrote.

    Same rule as datatrove's PipelineStepWithTokenizer.token_size: uint16 for
    vocabularies that fit, uint32 otherwise (GPT-2 -> 2, Qwen -> 4).
    """
    return 4 if load_tokenizer(tokenizer_name).get_vocab_size() > 65535 else 2


def eos_token_for(tokenizer_name: str) -> str:
    eos = AutoTokenizer.from_pretrained(tokenizer_name).eos_token
    if eos is None:
        raise ValueError(
            f"Tokenizer {tokenizer_name!r} has no eos_token; DocumentTokenizer "
            "needs one to separate documents in the token stream."
        )
    return eos


def _domain_stats(folder: Path, token_size: int) -> dict:
    files = sorted(folder.glob("*.ds")) if folder.is_dir() else []
    return {
        "folder": folder.name,
        "files": len(files),
        "tokens": sum(f.stat().st_size for f in files) // token_size,
    }


def build_tokenized_cache(
    cfg: ExperimentConfig,
    entry_dir: Path,
    signature: dict,
    *,
    tasks: int = DEFAULT_TASKS,
    workers: int = DEFAULT_WORKERS,
    splits: Iterable[str] = ("train", "validation"),
    overwrite: bool = False,
) -> Path:
    """Tokenize cfg's dataset(s) into entry_dir, one folder per domain.

    Returns entry_dir. Writes manifest.json last, so a half-finished build is
    never mistaken for a cache hit; re-running resumes, since datatrove's
    skip_completed reuses the per-task completion markers under logs/.
    """
    entry_dir = Path(entry_dir)
    if overwrite and entry_dir.exists():
        print(f"[tokenize] --overwrite: removing {entry_dir}")
        shutil.rmtree(entry_dir)
    entry_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_name = cfg.tokenizer_name
    token_size = token_size_for(tokenizer_name)
    eos_token = eos_token_for(tokenizer_name)
    print(
        f"[tokenize] {tokenizer_name} (token_size={token_size}, eos={eos_token!r}) "
        f"-> {entry_dir}"
    )

    jobs_by_split = {split: plan_jobs(cfg, split) for split in splits}

    # One domain id space shared by every split: in split_dataset mode each
    # split is discovered independently and need not surface the same domains
    # in the same order, but per-domain metrics are only comparable if a
    # domain id means the same thing in train and validation.
    all_domains = sorted({job["domain"] for jobs in jobs_by_split.values() for job in jobs})
    folders = {d: domain_folder(d) for d in all_domains}
    if len(set(folders.values())) != len(folders):
        raise ValueError(f"Domain names collide after sanitising to folders: {folders}")

    manifest_splits: dict[str, dict] = {}
    for split, jobs in jobs_by_split.items():
        print(f"\n[{split}] {len(jobs)} domain(s) to tokenize")

        # Each dataset has its own shard count, so tasks are resolved per
        # registry key (not job["path"], which is the underlying repo id --
        # two registry keys can share a repo id with different HF config
        # names/subsets, e.g. fineweb's 10BT vs 100BT samples, and must not
        # collide here).
        shards_of_path = {
            key: n_shards_for(key, split) for key in {j["registry_key"] for j in jobs}
        }

        for job in jobs:
            domain = job["domain"]
            split_tasks = min(tasks, shards_of_path[job["registry_key"]])
            # reader `limit` is per task, so spread the row budget across them.
            limit = (
                -1
                if cfg.max_documents == -1
                else max(1, math.ceil(cfg.max_documents / split_tasks))
            )
            out_folder = entry_dir / split / folders[domain]
            print(
                f"\n=== [{split}] tokenizing domain: {domain} -> {out_folder} "
                f"(tasks={split_tasks}, per-task row limit={limit}) ==="
            )

            pipeline = [
                HuggingFaceDatasetReader(
                    job["path"],
                    dataset_options=job["options"],
                    streaming=True,
                    limit=limit,
                    adapter=make_adapter(job["text_col"], job["split_column"]),
                )
            ]
            if job["split_column"]:
                pipeline.append(make_domain_filter(domain))
            pipeline.append(
                DocumentTokenizer(
                    output_folder=str(out_folder),
                    tokenizer_name_or_path=tokenizer_name,
                    eos_token=eos_token,
                    save_filename=folders[domain],
                )
            )

            LocalPipelineExecutor(
                pipeline=pipeline,
                tasks=split_tasks,
                workers=workers,
                logging_dir=str(entry_dir / "logs" / split / folders[domain]),
            ).run()

        manifest_splits[split] = {
            domain: _domain_stats(entry_dir / split / folders[domain], token_size)
            for domain in all_domains
        }

    manifest = {
        "signature": signature,
        "tokenizer": tokenizer_name,
        "token_size": token_size,
        "eos_token": eos_token,
        # domain_id == index into this list, for every split.
        "domains": all_domains,
        "folders": folders,
        "splits": manifest_splits,
    }
    tmp = entry_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(entry_dir / MANIFEST_NAME)

    print("\n[tokenize] done:")
    for split, stats in manifest_splits.items():
        total = sum(s["tokens"] for s in stats.values())
        print(f"  {split}: {total:,} tokens")
        for domain, s in stats.items():
            if s["tokens"] == 0:
                print(f"    ! {domain}: EMPTY (domain absent from this split?)")
            else:
                print(f"      {domain}: {s['tokens']:,} tokens in {s['files']} file(s)")
    return entry_dir


def read_manifest(entry_dir: Path) -> dict | None:
    path = Path(entry_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())
