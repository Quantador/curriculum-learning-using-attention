# parallel_experiments.py
"""
Ablation study orchestration for curriculum learning experiments, run either
as a single in-process experiment (bare invocation, no flags) or as a sweep
with many experiments training concurrently on one GPU.

Three sweep modes (same as before the merge with the old experiments.py):
  1. One-factor-at-a-time ablation (--all flag, default when a sweep is
     requested): For each field in EXPERIMENTAL_FIELDS, generates one config
     per alternative value, isolating the effect of each design choice.
  2. Full grid search (--combinations flag): all combinations of all field
     values. Grows exponentially — use only for a small subset of fields.
  3. Predefined profiles (--profile flag): curated subsets defined in
     EXPERIMENT_PROFILES for focused runs (e.g. 'final_presentation').

A bare `python parallel_experiments.py` (no --all/--field/--combinations/
--profile) runs a single experiment in-process from the default (or
--config-supplied) ExperimentConfig — no subprocess, no GPU probing.

Once a sweep of configs is generated, this script schedules them across one
GPU instead of experiments.py's old one-at-a-time loop:

  1. Tokenizes the dataset once and caches it to disk (shared_dataset.py) so
     every worker process loads it instead of re-tokenizing.
  2. Groups the generated configs by "memory signature" (the fields that
     actually affect GPU memory: model size, batch/pool, router, training
     algorithm, feature caching, ...) and probes each distinct signature once
     by really running it for a couple of steps in an isolated subprocess
     (gpu_memory_probe.py), reading back torch.cuda.max_memory_allocated().
  3. Greedily launches experiment_worker.py subprocesses, tracking committed
     GPU bytes against the free-memory budget, refilling slots as workers
     finish, until the queue is exhausted. A failed worker is logged and the
     rest of the sweep continues (its slot is freed immediately).

Every config keeps the experiment_name/save_dir that
generate_experiment_configs()/generate_combination_configs() already assign,
and wandb_project is left untouched — so every run in the sweep lands in the
same wandb project and shows up on the same graph.

CLI:
    python parallel_experiments.py                        # single run, in-process
    python parallel_experiments.py --all
    python parallel_experiments.py --field training_algorithm --field reward_signal
    python parallel_experiments.py --profile final_presentation
    python parallel_experiments.py --combinations --field router_architecture --field training_algorithm
    python parallel_experiments.py --list ...            # just print what would run

Scheduling knobs (sweep modes only):
    --max-parallel N     Skip GPU probing; always keep exactly N workers running.
    --safety-margin F    Fraction of free GPU memory treated as usable (default 0.85).
    --poll-interval S    Seconds between checks on running workers (default 5).
    --gpu INDEX          Physical GPU index to target (default 0).
See EXPERIMENTS.md for the full CLI reference and field descriptions.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

from config import ExperimentConfig, load_config_from_yaml
from data import get_tokenizer, make_mixed_chunks, make_single_chunks, MixedLMDataset
from models.model import TinyGPT
from models.router import build_router, get_router_feature_dim
from rl_training import train_router_experiments, compare_runs_experiments
from utils.metrics import MetricsTracker, DiversityTracker
from utils.shared_dataset import build_dataset_cache

# Maps each experimental dimension to (baseline_value, [alternative_values]).
# The baseline_value is used in the control experiment (experiment_name="baseline").
# Each alternative generates one experiment that changes only this single field.
# This one-factor-at-a-time design lets us isolate the effect of each choice.
EXPERIMENTAL_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Router architecture
    "router_architecture": ("attention", ["mlp", "linear"]),

    # Router features
    "enable_text_stat": (True, [False]),
    "enable_text_hierarchical": (True, [False]),
    "hierarchical_representation": ("full", ["embedder"]),

    # Training algorithm
    "training_algorithm": ("ppo", ["grpo", "reinforce"]),
    "reward_signal": ("loss_improvement", [
        "neg_loss",
        "relative_improvement",
        "difficulty_weighted",
        "uncertainty_reduction",
        "gradient_norm",
        "gradient_alignment",
        "combined",
    ]),

    # Selection strategy
    "selection_strategy": ("topk", ["sample", "epsilon_greedy"]),

    # Baseline for variance reduction
    "baseline_type": ("batch_mean", ["moving_avg", "none"]),

    # Temperature schedule
    "temp_schedule": ("fixed", ["linear_decay", "cosine_decay"]),

    # Entropy schedule
    "entropy_schedule": ("fixed", ["linear_decay", "cosine_decay", "exponential_decay", "cyclic"]),

    # Entropy formulation
    "entropy_type": ("shannon", ["renyi", "tsallis", "kl_uniform"]),

    # Entropy targeting (SAC-style automatic adjustment)
    "use_entropy_targeting": (False, [True]),

    # Coverage regularization
    "use_coverage_regularization": (False, [True]),
    "coverage_type": ("count", ["recency", "uncertainty"]),

    # Dataset combinations (easy_dataset, hard_dataset)
    # Easy options: TinyStories, Children-Stories, SimpleWikipedia, WikiText
    # Hard options: OpenWebText2, ArXiv, Code, FineWeb-Edu
    "easy_dataset": ("roneneldan/TinyStories", [
        "ajibawa-2023/Children-Stories-Collection",
        "Salesforce/wikitext",
    ]),
    "hard_dataset": ("Geralt-Targaryen/openwebtext2", [
        "armanc/scientific_papers",
        "CShorten/ML-ArXiv-Papers",
        "HuggingFaceFW/fineweb-edu",
    ]),

    # Feature caching
    "feature_cache_epochs": (0, [1, 2]),
}

# Predefined experiment profiles (subset of ablations)
FINAL_PRESENTATION_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Neg loss, gradient magnitude
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["neg_loss", "gradient_norm", "greats_score"]),
    # GRPO, PPO
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo", "reinforce"]),
    # Shannon fixed vs Shannon with decay (linear)
    "entropy_schedule": (EXPERIMENTAL_FIELDS["entropy_schedule"][0], ["linear_decay"]),
    # Coverage bonus
    "use_coverage_regularization": (EXPERIMENTAL_FIELDS["use_coverage_regularization"][0], [True]),
    # Top-k (baseline), sampling, sigma-greedy (epsilon_greedy)
    "selection_strategy": (EXPERIMENTAL_FIELDS["selection_strategy"][0], ["sample", "epsilon_greedy"]),
}

COMPARE_GRPO_VS_GREATS: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["greats_score"]),
    # GRPO, PPO
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo"]),
}
COMPARE_REWARD_SIGNALS: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["neg_loss", "gradient_norm", "greats_score"])
}
FEATURE_CACHE_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # baseline=0 (no cache) is the reference; 2 is the experiment
    "feature_cache_epochs": (0, [2]),
}

# New experiments added for NeurIPS:
#   1. Multi-head attention router (n_heads = 2 and 4)
#   2. Harder easy datasets (WikiText, Children-Stories)
#   3. Harder hard datasets (scientific papers, ML-ArXiv, FineWeb-Edu)
# Note: single-dataset mode and aux-net baseline use a different training loop
# and must be run via compare.py with use_single_dataset / run_aux_baseline.
ADDITIONAL_EXPERIMENTS_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    "router_n_heads": (1, [2, 4]),
    "easy_dataset": (
        "roneneldan/TinyStories",
        ["ajibawa-2023/Children-Stories-Collection", "Salesforce/wikitext"],
    ),
    "hard_dataset": (
        "Geralt-Targaryen/openwebtext2",
        ["armanc/scientific_papers", "CShorten/ML-ArXiv-Papers", "HuggingFaceFW/fineweb-edu"],
    ),
}

EXPERIMENT_PROFILES: Dict[str, Dict[str, tuple[Any, List[Any]]]] = {
    "final_presentation": FINAL_PRESENTATION_FIELDS,
    "final-presentation": FINAL_PRESENTATION_FIELDS,  # alias
    "feature_cache": FEATURE_CACHE_FIELDS,
    "additional_experiments": ADDITIONAL_EXPERIMENTS_FIELDS,
    "additional-experiments": ADDITIONAL_EXPERIMENTS_FIELDS,  # alias
    "grpo_vs_greats": COMPARE_GRPO_VS_GREATS,
    "compare_reward_signals": COMPARE_REWARD_SIGNALS
}


def get_profile_fields(profile: str | None) -> Dict[str, tuple[Any, List[Any]]] | None:
    if not profile:
        return None
    if profile in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[profile]
    normalized = profile.replace("-", "_").replace(" ", "_")
    if normalized in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[normalized]
    raise ValueError(
        f"Unknown profile '{profile}'. Available: {', '.join(sorted(set(EXPERIMENT_PROFILES.keys())))}"
    )


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(cfg: ExperimentConfig, tokenizer, train_ds, val_ds, base_metrics, router_metrics):
    """Run a single experiment with the given configuration.

    Shared by the bare in-process run_experiment() below and by
    utils/experiment_worker.py, which calls this once per config inside its
    own subprocess when running a sweep via run_scheduler().
    """
    print(f"\n{'='*60}")
    print(f"=== Running experiment: {cfg.experiment_name} ===")
    print(f"{'='*60}")
    print(f"  router_architecture: {cfg.router_architecture}")
    print(f"  enable_text_stat: {cfg.enable_text_stat}")
    print(f"  enable_text_hierarchical: {cfg.enable_text_hierarchical}")
    print(f"  hierarchical_representation: {cfg.hierarchical_representation}")
    print(f"  training_algorithm: {cfg.training_algorithm}")
    print(f"  reward_signal: {cfg.reward_signal}")
    print(f"  selection_strategy: {cfg.selection_strategy}")
    print(f"  baseline_type: {cfg.baseline_type}")
    print(f"  temp_schedule: {cfg.temp_schedule}")
    print(f"  entropy_schedule: {cfg.entropy_schedule}")

    set_seed(cfg.seed)

    # Ensure save directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = build_router(
        d_input=get_router_feature_dim(cfg, model_router.block),
        arch=cfg.router_architecture,
        d_k=128,
        n_heads=getattr(cfg, "router_n_heads", 1),
    )

    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router_experiments(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=experiment_metrics,
        diversity=router_div,
    )

    experiment_metrics.save(f"{cfg.save_dir}/{cfg.experiment_name}.json")

    print("\n=== Comparing runs ===")
    compare_runs_experiments(
        base_metrics,
        router_metrics,
        experiment_metrics,
    )

    return experiment_metrics


def run_experiment(cfg: ExperimentConfig | None = None):
    """Run a single experiment in-process (no subprocess, no GPU probing).

    Used for a bare `python parallel_experiments.py` invocation — builds the
    dataset itself, then delegates the actual training run to
    run_single_experiment(), same as every sweep worker does.
    """
    if cfg is None:
        cfg = ExperimentConfig()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()

    print("\n=== Building datasets ===")
    if cfg.use_single_dataset:
        train_chunks, val_chunks, train_embs, val_embs = make_single_chunks(cfg, tokenizer)
        train_ds = MixedLMDataset(train_chunks, embeddings=train_embs)
        val_ds   = MixedLMDataset(val_chunks,   embeddings=val_embs)
    else:
        train_chunks = make_mixed_chunks("train", cfg, tokenizer)
        val_chunks   = make_mixed_chunks("validation", cfg, tokenizer)
        train_ds = MixedLMDataset(train_chunks)
        val_ds   = MixedLMDataset(val_chunks)

    base_metrics = MetricsTracker.load("results/baseline_metrics.json")
    router_metrics = MetricsTracker.load("results/router_metrics.json")

    return run_single_experiment(
        cfg=cfg,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        base_metrics=base_metrics,
        router_metrics=router_metrics,
    )


def get_baseline_config() -> Dict[str, Any]:
    """Get the baseline values for all experimental fields."""
    return {field: values[0] for field, values in EXPERIMENTAL_FIELDS.items()}


def generate_experiment_configs(
    base_cfg: ExperimentConfig | None = None,
    experimental_fields: Dict[str, tuple[Any, List[Any]]] | None = None,
    include_baseline: bool = True,
) -> List[ExperimentConfig]:
    """
    Generate ablation (one-factor-at-a-time) experiment configurations.

    For each field in experimental_fields, generates one ExperimentConfig per
    alternative value. Each config is identical to the baseline except for
    exactly ONE field — this isolates each design choice cleanly.

    Example: baseline uses (ppo, loss_improvement, topk). Varying
    training_algorithm yields two configs: one with 'grpo', one with
    'reinforce', both with all other fields at baseline values.

    Args:
        base_cfg:            Starting config (defaults to ExperimentConfig()).
        experimental_fields: {field: (baseline, [alternatives])} mapping.
        include_baseline:    Whether to prepend the all-baseline config first.

    Returns a list of ExperimentConfig with descriptive experiment_name fields.
    """
    if base_cfg is None:
        base_cfg = ExperimentConfig()

    if experimental_fields is None:
        experimental_fields = EXPERIMENTAL_FIELDS

    configs = []

    # Get baseline values
    baseline_values = {field: values[0] for field, values in experimental_fields.items()}

    # Optionally add baseline experiment
    if include_baseline:
        baseline_cfg = replace(
            base_cfg,
            experiment_name="baseline",
            save_dir="results/baseline",
            **baseline_values
        )
        configs.append(baseline_cfg)

    # Generate one experiment per alternative value (one-factor-at-a-time)
    for field_name, (_, alternatives) in experimental_fields.items():
        for alt_value in alternatives:
            # Start from baseline, change only this one field
            overrides = baseline_values.copy()
            overrides[field_name] = alt_value

            # Generate descriptive experiment name
            experiment_name = f"{field_name}={alt_value}"

            new_cfg = replace(
                base_cfg,
                experiment_name=experiment_name,
                save_dir=f"results/{experiment_name}",
                **overrides
            )
            configs.append(new_cfg)

    return configs


def generate_combination_configs(
    base_cfg: ExperimentConfig | None = None,
    experimental_fields: Dict[str, List[Any]] | None = None,
) -> List[ExperimentConfig]:
    """
    Generate all combinations of experimental field values (full grid search).

    Produces the Cartesian product of all field value lists. Grows exponentially:
    3 fields × 3 values each = 27 experiments; 10 fields = potentially thousands.
    Only use this for small, targeted subsets of fields.

    Args:
        base_cfg:            Starting config (defaults to ExperimentConfig()).
        experimental_fields: {field: [values]} mapping (flat lists, no baseline tuple).

    Returns a list of ExperimentConfig, one per combination.
    """
    if base_cfg is None:
        base_cfg = ExperimentConfig()

    if experimental_fields is None:
        # Convert EXPERIMENTAL_FIELDS to flat list format
        experimental_fields = {
            field: [baseline] + alts
            for field, (baseline, alts) in EXPERIMENTAL_FIELDS.items()
        }

    field_names = list(experimental_fields.keys())
    field_values = list(experimental_fields.values())

    configs = []
    for combination in product(*field_values):
        overrides = dict(zip(field_names, combination))
        name_parts = [f"{k}={v}" for k, v in overrides.items()]
        experiment_name = "_".join(name_parts)

        new_cfg = replace(
            base_cfg,
            experiment_name=experiment_name,
            save_dir=f"results/{experiment_name}",
            **overrides
        )
        configs.append(new_cfg)

    return configs


SCRATCH_DIR = Path("results/_parallel_run")
CONTEXT_OVERHEAD_BYTES = 400 * 1024 * 1024  # per-process CUDA context overhead
PER_PROC_BUFFER = 1.15  # safety factor over the measured probe peak


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(asdict(cfg), f)


def memory_signature(cfg: ExperimentConfig) -> Tuple[Any, ...]:
    """Fields that plausibly change GPU memory use. Configs sharing a
    signature are assumed to need the same amount of GPU memory, so we only
    probe once per signature instead of once per config."""
    return (
        cfg.model_type, cfg.hf_model_name,
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.n_chunks,
        cfg.batch, cfg.block, cfg.pool_mult,
        cfg.training_algorithm, cfg.ppo_epochs, cfg.grpo_group_size,
        cfg.router_architecture, cfg.router_n_heads,
        cfg.enable_text_hierarchical, cfg.hierarchical_representation, cfg.hierarchical_layer_index,
        cfg.feature_cache_epochs > 0, cfg.feature_cache_batch_size,
    )


def query_free_memory_bytes(gpu_index: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    return int(out.decode().strip()) * 1024 * 1024


def probe_signature(cfg: ExperimentConfig, dataset_cache: Path, scratch_dir: Path, gpu_index: int) -> int:
    cfg_path = scratch_dir / "probe_configs" / f"{safe_name(cfg.experiment_name)}.yaml"
    dump_config(cfg, cfg_path)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    proc = subprocess.run(
        [
            sys.executable, "utils/gpu_memory_probe.py",
            "--config", str(cfg_path),
            "--dataset-cache", str(dataset_cache),
        ],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"GPU memory probe failed for a config like '{cfg.experiment_name}' "
            f"(exit {proc.returncode}). See output above."
        )

    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("PROBE_PEAK_BYTES="):
            return int(line.split("=", 1)[1])

    raise RuntimeError(
        f"GPU memory probe for '{cfg.experiment_name}' did not report PROBE_PEAK_BYTES. "
        f"stdout:\n{proc.stdout}"
    )


def compute_costs(
    configs: List[ExperimentConfig], dataset_cache: Path, scratch_dir: Path, gpu_index: int
) -> Dict[str, int]:
    """Returns {experiment_name: cost_bytes}, probing once per distinct memory signature."""
    signature_of = {cfg.experiment_name: memory_signature(cfg) for cfg in configs}
    representatives: Dict[Tuple[Any, ...], ExperimentConfig] = {}
    for cfg in configs:
        representatives.setdefault(signature_of[cfg.experiment_name], cfg)

    print(f"\n=== Probing GPU memory for {len(representatives)} distinct config signature(s) ===")
    peak_by_signature: Dict[Tuple[Any, ...], int] = {}
    for i, (sig, rep_cfg) in enumerate(representatives.items(), 1):
        peak = probe_signature(rep_cfg, dataset_cache, scratch_dir, gpu_index)
        peak_by_signature[sig] = peak
        print(f"  [{i}/{len(representatives)}] like '{rep_cfg.experiment_name}': {peak / 1e9:.2f} GB peak")

    return {
        cfg.experiment_name: int(peak_by_signature[signature_of[cfg.experiment_name]] * PER_PROC_BUFFER)
        + CONTEXT_OVERHEAD_BYTES
        for cfg in configs
    }


def run_scheduler(
    configs: List[ExperimentConfig],
    cost_by_name: Dict[str, int],
    usable_bytes: int,
    dataset_cache: Path,
    scratch_dir: Path,
    gpu_index: int,
    poll_interval: float,
) -> Dict[str, List[str]]:
    queue: List[Tuple[ExperimentConfig, int]] = [(cfg, cost_by_name[cfg.experiment_name]) for cfg in configs]
    running: Dict[subprocess.Popen, Tuple[ExperimentConfig, int, Any]] = {}
    committed = 0
    results: Dict[str, List[str]] = {"succeeded": [], "failed": []}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    def launch(cfg: ExperimentConfig, cost: int) -> None:
        nonlocal committed
        name = safe_name(cfg.experiment_name)
        cfg_path = scratch_dir / "configs" / f"{name}.yaml"
        dump_config(cfg, cfg_path)
        log_path = scratch_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w")
        popen = subprocess.Popen(
            [
                sys.executable, "utils/experiment_worker.py",
                "--config", str(cfg_path),
                "--dataset-cache", str(dataset_cache),
            ],
            stdout=log_f, stderr=subprocess.STDOUT, env=env,
        )
        running[popen] = (cfg, cost, log_f)
        committed += cost
        print(
            f"[launch] {cfg.experiment_name}  "
            f"(cost {cost / 1e9:.2f} GB, committed {committed / 1e9:.2f}/{usable_bytes / 1e9:.2f} GB, "
            f"{len(running)} running, {len(queue)} queued)"
        )

    while queue or running:
        while queue and committed + queue[0][1] <= usable_bytes:
            cfg, cost = queue.pop(0)
            launch(cfg, cost)

        if not running and queue:
            cfg, cost = queue.pop(0)
            print(
                f"[warn] '{cfg.experiment_name}' needs ~{cost / 1e9:.2f} GB, more than the "
                f"{usable_bytes / 1e9:.2f} GB usable budget alone — running it by itself."
            )
            launch(cfg, cost)

        finished = [p for p in running if p.poll() is not None]
        for p in finished:
            cfg, cost, log_f = running.pop(p)
            log_f.close()
            committed -= cost
            name = safe_name(cfg.experiment_name)
            if p.returncode == 0:
                results["succeeded"].append(cfg.experiment_name)
                print(f"[done]   {cfg.experiment_name}")
            else:
                results["failed"].append(cfg.experiment_name)
                print(
                    f"[FAILED] {cfg.experiment_name} (exit {p.returncode}) "
                    f"— see {scratch_dir}/logs/{name}.log"
                )

        if not finished and (queue or running):
            time.sleep(poll_interval)

    return results


def build_config_list(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.profile and args.field:
        print("Error: --profile and --field cannot be used together.")
        sys.exit(1)

    # base_cfg is what every generated experiment is built from. Fields NOT
    # under ablation (i.e. not in selected_fields below) carry through from
    # it unchanged; fields that ARE under ablation still get forced to their
    # declared (baseline, [alternatives]) values regardless of base_cfg.
    base_cfg = load_config_from_yaml(args.config) if args.config else None

    # The base config's experiment_name sets the shared wandb_project
    # (curriculum-learning-<name>) that every run in the sweep lands in.
    # Without --name it falls back to --config's (or ExperimentConfig()'s
    # "presentation_experiment") experiment_name.
    if args.name:
        # save_dir/wandb_project only re-derive from experiment_name in
        # __post_init__ while still None; base_cfg (if any) already has them
        # resolved to concrete strings, so force both back to None or
        # replace() would silently keep the stale values tied to the old name.
        base_cfg = (
            replace(base_cfg, experiment_name=args.name, save_dir=None, wandb_project=None)
            if base_cfg else ExperimentConfig(experiment_name=args.name)
        )

    try:
        profile_fields = get_profile_fields(args.profile)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.field:
        selected_fields = {k: v for k, v in EXPERIMENTAL_FIELDS.items() if k in args.field}
        if not selected_fields:
            print(f"Error: No valid fields specified. Available: {list(EXPERIMENTAL_FIELDS.keys())}")
            sys.exit(1)
    elif profile_fields is not None:
        selected_fields = profile_fields
    else:
        selected_fields = None

    if args.combinations:
        flat_fields = {
            f: [b] + a for f, (b, a) in (selected_fields or EXPERIMENTAL_FIELDS).items()
        }
        return generate_combination_configs(base_cfg=base_cfg, experimental_fields=flat_fields)
    return generate_experiment_configs(
        base_cfg=base_cfg,
        experimental_fields=selected_fields,
        include_baseline=not args.no_baseline,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curriculum learning experiments in parallel on one GPU")
    parser.add_argument("--all", action="store_true", help="Run all ablation experiments (one-factor-at-a-time)")
    parser.add_argument("--combinations", action="store_true", help="Run full grid search of all combinations")
    parser.add_argument("--field", type=str, action="append", help="Run experiments for specific field(s) only")
    parser.add_argument("--profile", type=str, help="Run a predefined experiment profile")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the baseline experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML file with ExperimentConfig field overrides, used as the base config every experiment is built from (fields under ablation are still forced to their declared values)")
    parser.add_argument("--name", type=str, default=None, help="Sweep name; sets the shared wandb project curriculum-learning-<name> (default: --config's, or presentation_experiment)")
    parser.add_argument("--list", action="store_true", help="List experiments that would run, without running them")
    parser.add_argument("--max-parallel", type=int, default=None, help="Skip GPU probing; always run exactly N workers")
    parser.add_argument("--safety-margin", type=float, default=0.85, help="Fraction of free GPU memory usable (default 0.85)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls of running workers")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to target (default 0)")
    args = parser.parse_args()

    if not (args.all or args.field or args.combinations or args.profile):
        if not args.list:
            # Bare invocation (no sweep flag): run a single experiment
            # in-process, same as the old bare `python experiments.py`.
            base_cfg = load_config_from_yaml(args.config) if args.config else None
            run_experiment(cfg=base_cfg)
            return
        # --list with no sweep flag still lists the full EXPERIMENTAL_FIELDS
        # ablation, so `--list` alone works as a preview of `--all`.

    configs = build_config_list(args)

    if args.list:
        print(f"\n=== {len(configs)} experiments would be run ===\n")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.experiment_name}")
        return

    scratch_dir = SCRATCH_DIR
    scratch_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache = scratch_dir / "dataset_cache.pt"

    base_cfg = ExperimentConfig()
    tokenizer = get_tokenizer()
    print("\n=== Building dataset cache (once, shared by every worker) ===")
    build_dataset_cache(base_cfg, tokenizer, str(dataset_cache))

    if args.max_parallel is not None:
        cost_by_name = {cfg.experiment_name: 1 for cfg in configs}
        usable = args.max_parallel
        print(f"\n=== Skipping GPU probe: fixed at {args.max_parallel} concurrent workers ===")
    else:
        free_bytes = query_free_memory_bytes(args.gpu)
        usable = int(free_bytes * args.safety_margin)
        print(f"\nGPU {args.gpu}: {free_bytes / 1e9:.2f} GB free, {usable / 1e9:.2f} GB usable budget")
        cost_by_name = compute_costs(configs, dataset_cache, scratch_dir, args.gpu)
        avg_cost = sum(cost_by_name.values()) / len(cost_by_name)
        print(f"Estimated max concurrency: ~{max(1, int(usable / avg_cost))} experiments")

    print(f"\n=== Running {len(configs)} experiments (parallel) ===\n")
    results = run_scheduler(
        configs, cost_by_name, usable, dataset_cache, scratch_dir, args.gpu, args.poll_interval
    )

    print(f"\n{'=' * 60}")
    print(f"=== {len(results['succeeded'])} succeeded, {len(results['failed'])} failed ===")
    if results["failed"]:
        print("Failed experiments (see results/_parallel_run/logs/<name>.log):")
        for name in results["failed"]:
            print(f"  - {name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
