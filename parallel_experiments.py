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
import subprocess
import sys
import time

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import ExperimentConfig, load_config_from_yaml
from data import get_tokenizer, make_mixed_chunks, make_single_chunks, MixedLMDataset

from utils.metrics import MetricsTracker
from utils.shared_dataset import build_dataset_cache
from consts import EXPERIMENTAL_FIELDS, SCRATCH_DIR
from utils.general_utils import (get_profile_fields, safe_name, 
                                 query_free_memory_bytes, compute_costs, dump_config)

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
        # Create logging folders 
        name = safe_name(cfg.experiment_name)
        cfg_path = scratch_dir / "configs" / f"{name}.yaml"
        dump_config(cfg, cfg_path)
        log_path = scratch_dir / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w")

        # Launch experiment 
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
