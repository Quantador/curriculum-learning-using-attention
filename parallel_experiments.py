# parallel_experiments.py
"""
Run an ablation sweep with many experiments training concurrently on one GPU,
instead of experiments.py's one-at-a-time loop.

TinyGPT experiments are small enough that a single 40GB card sits mostly
idle running them sequentially. This script:

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
and wandb_project is left untouched — so, same as experiments.py, every run
in the sweep lands in the same wandb project and shows up on the same graph.

Mirrors experiments.py's CLI:
    python parallel_experiments.py --all
    python parallel_experiments.py --field training_algorithm --field reward_signal
    python parallel_experiments.py --profile final_presentation
    python parallel_experiments.py --combinations --field router_architecture --field training_algorithm
    python parallel_experiments.py --list ...            # just print what would run

Scheduling knobs:
    --max-parallel N     Skip GPU probing; always keep exactly N workers running.
    --safety-margin F    Fraction of free GPU memory treated as usable (default 0.85).
    --poll-interval S    Seconds between checks on running workers (default 5).
    --gpu INDEX          Physical GPU index to target (default 0).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from config import ExperimentConfig, load_config_from_yaml
from data import get_tokenizer
from experiments import (
    EXPERIMENTAL_FIELDS,
    generate_experiment_configs,
    generate_combination_configs,
    get_profile_fields,
)
from utils.shared_dataset import build_dataset_cache

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
        print("Error: specify --all, --field, --profile, and/or --combinations (see --help).")
        sys.exit(1)

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
