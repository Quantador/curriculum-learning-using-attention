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

  1. Resolves each config to its pre-tokenized dataset cache entry
     (shared_dataset.py) so every worker process memory-maps the same token
     files. This script never tokenizes: if the cache for a config is
     missing it exits with the build_dataset_cache.py command to run first.
  2. Groups the generated configs by "memory signature" (the fields that
     actually affect GPU memory: model size, batch/pool, router, training
     algorithm, feature caching, ...) and probes each distinct signature once
     by really running it for a couple of steps in an isolated subprocess
     (gpu_memory_probe.py), reading back torch.cuda.max_memory_allocated().
  3. Greedily launches experiment_worker.py subprocesses, tracking committed
     GPU bytes against the free-memory budget, refilling slots as workers
     finish, until the queue is exhausted. A failed worker is logged and the
     rest of the sweep continues (its slot is freed immediately).

Every config keeps the experiment_name that
generate_experiment_configs()/generate_combination_configs() already assigns,
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

DDP mode -- speed up ONE experiment across multiple GPUs instead of packing
several experiments onto one, e.g. when a single config (large model) is the
bottleneck rather than sweep breadth. Launch under torchrun instead of plain
python; detected automatically from torchrun's WORLD_SIZE env var, no new
flag needed:

    torchrun --standalone --nproc-per-node=gpu parallel_experiments.py --profile final_presentation

Every config in the sweep still runs, just sequentially instead of packed --
each one trains across all GPUs in the process group via
DistributedDataParallel (see run_ddp_sweep()), one persistent NCCL process
group for the whole sweep. --max-parallel/--safety-margin/--gpu are ignored
in this mode (no GPU-memory probing, no CUDA_VISIBLE_DEVICES targeting --
GPU selection comes from torchrun's LOCAL_RANK).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time

from datetime import datetime
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist

from config import ExperimentConfig, load_config_from_yaml
from data import get_tokenizer

from utils.shared_dataset import (
    DatasetCacheMissing,
    dataset_signature,
    load_dataset_cache,
    require_dataset_cache,
)
from utils.distributed_utils import cleanup_distributed, setup_distributed
from utils.metrics import MetricsTracker
from utils import memory_snapshot, run_status
from consts import EXPERIMENTAL_FIELDS, SCRATCH_DIR
from utils.general_utils import (get_profile_fields, safe_name,
                                 query_free_memory_bytes, compute_costs, dump_config, tee_stdio)

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

    An alternative may instead be a dict of {field: value} overrides (with a
    required "_name" key for its experiment_name) when a single scalar can't
    express the comparison -- e.g. "sentence embedder alone" needs
    enable_text_hierarchical AND enable_text_stat turned off together with
    sentence_embedder_model set, not just one field changed from baseline.

    Always includes two core reference runs that every sweep should be judged
    against, regardless of include_baseline:
      - 'experiment_baseline':  plain baseline_values, RL router as usual --
                                the sweep's own control condition.
      - 'random_batch_baseline': uniform random cfg.global_batch_size-sized
                                selection, no router/aux_net (training.train_baseline()).
    Additionally includes (unless include_baseline=False) two secondary
    reference runs:
      - 'aux_baseline':         supervised MSE alternative to the router
                                (rl_training.train_aux_baseline()).
      - 'random_pool_baseline':  same as random_batch_baseline, but selecting
                                cfg.pool samples (the router's whole
                                candidate pool, unfiltered).
    None of these are ablated EXPERIMENTAL_FIELDS entries — they're fixed
    reference points that ride along with every sweep instead of only
    appearing when someone happens to select them with --field.

    Args:
        base_cfg:            Starting config (defaults to ExperimentConfig()).
        experimental_fields: {field: (baseline, [alternatives])} mapping.
        include_baseline:    Whether to also include aux_baseline and
                             random_pool_baseline (experiment_baseline and
                             random_batch_baseline are always included).

    Returns a list of ExperimentConfig with descriptive experiment_name fields.
    """
    if base_cfg is None:
        base_cfg = ExperimentConfig()

    if experimental_fields is None:
        experimental_fields = EXPERIMENTAL_FIELDS

    configs = []

    # Get baseline values
    baseline_values = {field: values[0] for field, values in experimental_fields.items()}

    # Reference experiments (router baseline + non-router controls), each
    # identical to baseline_values except for the one flag that switches
    # training loop. experiment_baseline and random_batch_baseline are the
    # two every sweep is judged against, so they ride along unconditionally;
    # aux_baseline and random_pool_baseline are secondary and gated behind
    # include_baseline.
    reference_overrides = {
        "random_pool_baseline": {"run_random_pool_baseline": True},
        "experiment_baseline": {},
        "aux_baseline": {"run_aux_baseline": True},
        "random_batch_baseline": {"run_random_batch_baseline": True},
    }
    unconditional_references = {"experiment_baseline", "random_batch_baseline"}
    for name, overrides in reference_overrides.items():
        if include_baseline or name in unconditional_references:
            configs.append(replace(
                base_cfg,
                experiment_name=name,
                **baseline_values,
                **overrides,
            ))

    # Generate one experiment per alternative value (one-factor-at-a-time)
    for field_name, (_, alternatives) in experimental_fields.items():
        for alt_value in alternatives:
            # Start from baseline, change only this one field -- unless
            # alt_value is a multi-field combo dict (see docstring), which
            # applies all its overrides together under its own "_name".
            overrides = baseline_values.copy()
            if isinstance(alt_value, dict):
                combo = dict(alt_value)
                experiment_name = combo.pop("_name")
                overrides.update(combo)
            else:
                overrides[field_name] = alt_value
                experiment_name = f"{field_name}={alt_value}"

            new_cfg = replace(
                base_cfg,
                experiment_name=experiment_name,
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
            **overrides
        )
        configs.append(new_cfg)

    return configs

def run_scheduler(
    configs: List[ExperimentConfig],
    cost_by_name: Dict[str, int],
    usable_bytes: int,
    dataset_cache_by_name: Dict[str, Path],
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

    status_dir = run_status.status_dir_for(scratch_dir)

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
                "--dataset-cache", str(dataset_cache_by_name[cfg.experiment_name]),
                "--status-dir", str(status_dir),
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
            # From the parent's side, so a worker that died too abruptly to
            # write its own status (SIGKILL / host OOM-killer) still gets one.
            run_status.record_child_exit(status_dir, name, p.returncode)
            if p.returncode == 0:
                results["succeeded"].append(cfg.experiment_name)
                print(f"[done]   {cfg.experiment_name}")
            else:
                results["failed"].append(cfg.experiment_name)
                # Popen reports a signal death as -signum. Spelling it out
                # here matters because the two cases have nothing to do with
                # each other: a positive code is the experiment's own bug, a
                # negative one is something outside it reaching in.
                how = (f"killed by {signal.Signals(-p.returncode).name}"
                       if p.returncode < 0 else f"exit {p.returncode}")
                print(
                    f"[FAILED] {cfg.experiment_name} ({how}) "
                    f"— see {scratch_dir}/logs/{name}.log"
                )

        if not finished and (queue or running):
            time.sleep(poll_interval)

    return results


def run_ddp_sweep(
    configs: List[ExperimentConfig],
    dataset_cache_by_name: Dict[str, Path],
    signature_of_name: Dict[str, Any],
    rank: int,
    local_rank: int,
    world_size: int,
    scratch_dir: Path,
) -> None:
    """
    DDP-mode sweep runner: entered instead of run_scheduler() when launched
    under torchrun (world_size > 1, see main()). No GPU-memory probing, no
    packing multiple experiments onto one GPU -- each config in `configs`
    trains one at a time, using every GPU in this process group via
    DistributedDataParallel (see the DDP-wrapping inside train_baseline /
    train_router_experiments / train_aux_baseline), run entirely in-process
    (not subprocessed) so the NCCL process group setup_distributed() already
    initialized persists across the whole sweep instead of being torn down
    and rebuilt per config.

    utils.shared_dataset.load_dataset_cache() reads a pre-tokenized,
    mmap-backed on-disk cache, so it is safe (and gives an identical result)
    to call on every rank independently -- no rank-0-loads-then-broadcasts
    step needed. It's also loaded once per distinct dataset signature and
    reused across every config that shares it (typically all of them, since
    a sweep's ablated fields rarely touch dataset_signature()'s fields),
    rather than reloaded -- and re-mmap'd/re-shuffled -- per config.
    """
    # Imported here, not at module level: run_single_experiment lives in the
    # same package as this file's other GPU-probing/subprocess-launching
    # machinery, which plain `python parallel_experiments.py` (no torchrun)
    # should never need to import torch.distributed-adjacent DDP code for.
    from utils.experiment_worker import run_single_experiment

    if not configs:
        return

    tokenizer = get_tokenizer(configs[0].tokenizer_name)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Two permanently-empty trackers: compare_runs_experiments() (called at
    # the end of every run_single_experiment()) only ever prints "cannot
    # compare" against these today -- nothing in the active codebase writes
    # results/baseline_metrics.json / router_metrics.json anymore (dead
    # since the old compare.py script was removed), so there's no file to
    # load here either. Matches what MetricsTracker.load() on a missing path
    # already returns.
    base_metrics = MetricsTracker("baseline", use_wandb=False)
    router_metrics = MetricsTracker("router", use_wandb=False)

    loaded_by_signature: Dict[str, Tuple[Any, Any]] = {}

    for i, cfg in enumerate(configs, 1):
        if rank == 0:
            print(f"\n=== [{i}/{len(configs)}] DDP experiment: {cfg.experiment_name} "
                  f"(world_size={world_size}) ===")

        cfg = replace(cfg, rank=rank, world_size=world_size, local_rank=local_rank, device=device)

        sig_key = json.dumps(signature_of_name[cfg.experiment_name], sort_keys=True)
        if sig_key not in loaded_by_signature:
            loaded_by_signature[sig_key] = load_dataset_cache(dataset_cache_by_name[cfg.experiment_name], cfg)
        train_ds, val_ds = loaded_by_signature[sig_key]

        name = safe_name(cfg.experiment_name)
        if rank == 0:
            dump_config(cfg, scratch_dir / "configs" / f"{name}.yaml")

        # Only rank 0 prints anything meaningful (tqdm/wandb/logging are all
        # rank==0-gated in train_baseline/train_router_experiments/
        # train_aux_baseline), so only its stdio needs teeing to line up with
        # what run_scheduler()'s subprocess-per-experiment log files capture
        # on the single-GPU path. The try/except stays INSIDE this block so
        # a failure's traceback lands in the log file too, not just the raw
        # console -- otherwise the one thing you'd actually want to read to
        # debug a failed experiment (e.g. a CUDA OOM message) never makes it
        # into results/_parallel_run/.../logs/<name>.log.
        # Only rank 0 records status, for the same reason only rank 0 tees:
        # every rank runs the identical sequence, so 16 ranks would write 16
        # identical status files racing over one path.
        tracker = (
            run_status.track(run_status.status_dir_for(scratch_dir), name,
                             experiment=cfg.experiment_name, world_size=world_size,
                             wandb_project=cfg.wandb_project)
            if rank == 0 else contextlib.nullcontext()
        )
        with tee_stdio(scratch_dir / "logs" / f"{name}.log") if rank == 0 else contextlib.nullcontext():
            try:
                with tracker:
                    # Recorded on EVERY rank, unlike the status tracking above:
                    # under FSDP the ranks hold different shards, so an OOM on
                    # rank 7 is not visible in rank 0's allocator history.
                    # Filenames carry the rank, so they don't collide.
                    with memory_snapshot.record(scratch_dir / "snapshots", name, rank=rank):
                        run_single_experiment(
                            cfg=cfg, tokenizer=tokenizer, train_ds=train_ds, val_ds=val_ds,
                            base_metrics=base_metrics, router_metrics=router_metrics,
                        )
            except Exception:
                if rank == 0:
                    print(f"[FAILED] {cfg.experiment_name} — see {scratch_dir}/logs/{name}.log")
                import traceback
                traceback.print_exc()
                # Every rank must keep moving through the same config sequence in
                # lockstep (they share one process group) -- letting one rank
                # raise while others continue would desync the next config's
                # collective calls, or hang if others are still mid-.backward().
                # Continuing the loop on every rank keeps them aligned; a bad
                # config just fails identically everywhere instead of hanging.

        if world_size > 1:
            dist.barrier()  # hold every rank at the same config boundary before moving on


JOB_TEMPLATE = """#!/usr/bin/env bash
#SBATCH --account={account}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
#SBATCH --nodes={nodes}
#SBATCH --partition={partition}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --no-requeue

# Generated by parallel_experiments.py --submit. One experiment per job:
# {experiment}
#
# No dataset-cache build here -- --submit refuses to run until the cache
# exists, so every job can go straight to training instead of N jobs each
# re-tokenizing the same corpus.
set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT={master_port}
export WORLD_SIZE=$SLURM_NPROCS

srun --mpi=pmix --environment="{environment}" --network=disable_rdzv_get \\
    bash -c '
        source {venv}/bin/activate
        export RANK=$SLURM_PROCID
        export LOCAL_RANK=$SLURM_LOCALID
        exec python parallel_experiments.py --single --config "{config_path}"
    '
"""


def submit_experiment_jobs(
    configs: List[ExperimentConfig],
    dataset_cache_by_name: Dict[str, Any],
    scratch_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Write one sbatch script per config and submit it.

    Each job runs a single experiment across one node's GPUs via the same
    run_ddp_sweep path a hand-launched run uses, so nothing about how an
    experiment trains changes -- only how many allocations the sweep occupies.
    N short single-node jobs backfill into the scheduler's gaps far better
    than one long multi-node reservation, and a preemption costs one
    experiment rather than the whole sweep.
    """
    job_dir = scratch_dir / "jobs"
    cfg_dir = scratch_dir / "configs"
    log_dir = (scratch_dir / "logs").resolve()
    for d in (job_dir, cfg_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    submitted, failed = [], []
    for i, cfg in enumerate(configs):
        name = safe_name(cfg.experiment_name)
        cfg_path = (cfg_dir / f"{name}.yaml").resolve()
        # cfg.device is "" here (this runs on a GPU-less login node) and stays
        # "" in the YAML; each job resolves it for itself via resolve_device().
        dump_config(cfg, cfg_path)

        script = JOB_TEMPLATE.format(
            account=args.submit_account,
            job_name=name[:60],
            log_dir=log_dir,
            nodes=args.submit_nodes,
            partition=args.submit_partition,
            tasks_per_node=args.submit_gpus_per_node,
            # Whole-node CPU budget split across the node's tasks; nodes are
            # allocated exclusively, so anything less just idles cores.
            cpus_per_task=288 // max(1, args.submit_gpus_per_node),
            time=args.submit_time,
            experiment=cfg.experiment_name,
            # Distinct port per job: two jobs landing on the same node would
            # otherwise collide on the rendezvous socket.
            master_port=29500 + (i % 1000),
            environment=args.submit_environment,
            venv=args.submit_venv,
            config_path=cfg_path,
        )
        script_path = job_dir / f"{name}.sbatch"
        script_path.write_text(script)
        script_path.chmod(0o755)

        if args.submit_dry_run:
            print(f"  [dry-run] {cfg.experiment_name}  ->  {script_path}")
            continue
        try:
            out = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True, check=True, cwd=Path.cwd(),
            ).stdout.strip()
            job_id = out.rsplit(maxsplit=1)[-1] if out else "?"
            submitted.append((job_id, cfg.experiment_name))
            print(f"  [{job_id}] {cfg.experiment_name}")
        except FileNotFoundError:
            print("Error: sbatch not found. --submit must run where the Slurm "
                  "client is available (a login node), not inside the container.")
            sys.exit(1)
        except subprocess.CalledProcessError as exc:
            failed.append((cfg.experiment_name, exc.stderr.strip()))
            print(f"  [FAILED] {cfg.experiment_name}: {exc.stderr.strip()}")

    print(f"\nscripts:  {job_dir}")
    print(f"configs:  {cfg_dir}")
    print(f"logs:     {log_dir}")
    if args.submit_dry_run:
        print(f"\ndry run: {len(configs)} script(s) written, nothing submitted.")
        return
    print(f"\nsubmitted {len(submitted)}/{len(configs)} job(s)")
    if failed:
        print(f"{len(failed)} failed to submit:")
        for name, err in failed:
            print(f"  {name}: {err}")
    if submitted:
        ids = " ".join(j for j, _ in submitted)
        print(f"\n  squeue -j {ids.replace(' ', ',')}")
        print(f"  scancel {ids}")


def build_config_list(args: argparse.Namespace) -> List[ExperimentConfig]:
    if args.profile and args.field:
        print("Error: --profile and --field cannot be used together.")
        sys.exit(1)

    # --single is the leaf case: run the given YAML verbatim, generating
    # nothing. Without it, a bare --config falls through to
    # generate_experiment_configs(experimental_fields=None), which defaults to
    # the FULL EXPERIMENTAL_FIELDS ablation -- so a submitted per-experiment
    # job would re-expand its own config into the whole sweep again.
    if args.single:
        if not args.config:
            print("Error: --single requires --config.")
            sys.exit(1)
        if args.profile or args.field or args.all or args.combinations:
            print("Error: --single cannot be combined with --profile/--field/--all/--combinations.")
            sys.exit(1)
        return [load_config_from_yaml(args.config)]

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
        # wandb_project only re-derives from experiment_name in __post_init__
        # while still None; base_cfg (if any) already has it resolved to a
        # concrete string, so force it back to None or replace() would
        # silently keep the stale value tied to the old name.
        base_cfg = (
            replace(base_cfg, experiment_name=args.name, wandb_project=None)
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
    parser.add_argument("--no-baseline", action="store_true", help="Skip the aux_baseline and random_pool_baseline reference experiments (experiment_baseline and random_batch_baseline always run)")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML file with ExperimentConfig field overrides, used as the base config every experiment is built from (fields under ablation are still forced to their declared values)")
    parser.add_argument("--name", type=str, default=None, help="Sweep name; sets the shared wandb project curriculum-learning-<name> (default: --config's, or presentation_experiment)")
    parser.add_argument("--list", action="store_true", help="List experiments that would run, without running them")
    parser.add_argument("--single", action="store_true", help="Run exactly the config given by --config, with no ablation generation. What each --submit job invokes")
    parser.add_argument("--submit", action="store_true", help="Submit one sbatch job per experiment instead of running them here; each job runs its config DDP across one node's GPUs")
    parser.add_argument("--submit-dry-run", action="store_true", help="With --submit: write the job scripts but do not call sbatch")
    parser.add_argument("--submit-time", type=str, default="04:30:00", help="Wall clock per submitted job (default 03:00:00)")
    parser.add_argument("--submit-partition", type=str, default="normal", help="Partition for submitted jobs (default normal)")
    parser.add_argument("--submit-account", type=str, default="infra01", help="Account for submitted jobs (default infra01)")
    parser.add_argument("--submit-nodes", type=int, default=1, help="Nodes per submitted job (default 1)")
    parser.add_argument("--submit-gpus-per-node", type=int, default=4, help="Tasks (= GPUs) per node for submitted jobs (default 4)")
    parser.add_argument("--submit-environment", type=str, default=str(Path.home() / "curriculum-learning-using-attention" / "clariden.toml"), help="Container .toml passed to srun --environment")
    parser.add_argument("--submit-venv", type=str, default="/iopsstor/scratch/cscs/$USER/curriculum-venv", help="Venv activated inside each job")
    parser.add_argument("--max-parallel", type=int, default=None, help="Skip GPU probing; always run exactly N workers")
    parser.add_argument("--safety-margin", type=float, default=0.6, help="Fraction of free GPU memory usable (default 0.6)")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls of running workers")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index to target (default 0). No-op under DDP (torchrun) -- GPU selection there comes from LOCAL_RANK, not this")
    args = parser.parse_args()

    # (0, 0, 1) untouched if not launched via torchrun (WORLD_SIZE unset) --
    # see utils/distributed_utils.py.
    rank, local_rank, world_size = setup_distributed()

    configs = build_config_list(args)

    if args.list:
        if rank == 0:
            print(f"\n=== {len(configs)} experiments would be run ===\n")
            for i, cfg in enumerate(configs, 1):
                print(f"  {i}. {cfg.experiment_name}")
        cleanup_distributed()
        return

    if rank == 0:
        print("\n=== Resolving pre-tokenized dataset cache per config ===")
    signature_of_name = {cfg.experiment_name: dataset_signature(cfg) for cfg in configs}
    representative_by_signature: Dict[Any, ExperimentConfig] = {}
    for cfg in configs:
        sig_key = json.dumps(signature_of_name[cfg.experiment_name], sort_keys=True)
        representative_by_signature.setdefault(sig_key, cfg)

    # Nothing is tokenized here: a sweep that finds no cache stops immediately
    # rather than tying up a GPU node tokenizing (see build_dataset_cache.py).
    # Safe to resolve identically on every rank under DDP too -- this only
    # reads each cache entry's manifest.json, no dataset loading yet.
    try:
        cache_path_by_signature = {
            sig_key: require_dataset_cache(rep_cfg)
            for sig_key, rep_cfg in representative_by_signature.items()
        }
    except DatasetCacheMissing as exc:
        if rank == 0:
            print(f"\n{exc}")
        cleanup_distributed()
        sys.exit(1)
    dataset_cache_by_name = {
        cfg.experiment_name: cache_path_by_signature[json.dumps(signature_of_name[cfg.experiment_name], sort_keys=True)]
        for cfg in configs
    }
    if rank == 0:
        print(
            f"{len(configs)} config(s) map to {len(representative_by_signature)} "
            f"distinct dataset signature(s)"
        )

    # Scratch dir name is <timestamp>_<label>, and rank 0 both picks it and
    # creates it before telling anyone -- the two properties are related.
    #
    # The label exists because --submit launches N independent jobs: with a
    # timestamp alone, any two starting in the same second picked the same
    # results/_parallel_run/<timestamp>/ and whichever lost the mkdir race
    # died on the exist_ok=False below. Since every submitted job runs a
    # different experiment, the label makes that collision impossible.
    # The numeric suffix then covers what the label cannot: the *same*
    # experiment launched twice within one second (a resubmit, a retry).
    #
    # exist_ok=False is kept deliberately -- it is what makes the loop a real
    # claim on the directory rather than a check-then-use race between two
    # processes both finding it absent.
    if rank == 0:
        label = safe_name(
            configs[0].experiment_name if len(configs) == 1
            else (args.name or "sweep")
        )[:80]
        base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"
        candidate = SCRATCH_DIR / base
        attempt = 1
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                attempt += 1
                candidate = SCRATCH_DIR / f"{base}-{attempt}"
        chosen = [candidate.name]
        print(f"\n=== Scratch dir for this run: {candidate} ===")
    else:
        chosen = [None]
    # Broadcast AFTER the mkdir, not before: the retry above can change the
    # name, so sending the timestamp up front would leave the other ranks
    # writing into a directory rank 0 ended up abandoning.
    if world_size > 1:
        dist.broadcast_object_list(chosen, src=0)
    scratch_dir = SCRATCH_DIR / chosen[0]
    if world_size > 1:
        dist.barrier()  # other ranks wait for rank 0's mkdir before writing into it

    if args.submit:
        # Deliberately before the signal handlers and any GPU work below:
        # this process only writes YAML and calls sbatch, then exits. It runs
        # on a login node (that is where the Slurm client lives), so it must
        # never touch CUDA.
        if world_size > 1:
            print("Error: --submit launches its own jobs; run it directly, not under srun/torchrun.")
            cleanup_distributed()
            sys.exit(1)
        print(f"\n=== Submitting {len(configs)} experiment(s), "
              f"{args.submit_nodes} node(s) x {args.submit_gpus_per_node} GPU(s) each, "
              f"{args.submit_time} per job ===\n")
        submit_experiment_jobs(configs, dataset_cache_by_name, scratch_dir, args)
        cleanup_distributed()
        return

    # From here on the process can live for hours, which is the whole window
    # in which an external kill is both likely and unexplained. Installed on
    # every rank, not just rank 0: under torchrun a preemption SIGTERMs all of
    # them, and knowing whether rank 0 specifically got notice is worth the
    # one small file per rank.
    run_status.install_signal_handlers(
        scratch_dir, role="orchestrator" if rank == 0 else f"rank{rank}"
    )

    if world_size > 1:
        # DDP mode (launched via torchrun): one experiment at a time, using
        # every GPU in this process group via DistributedDataParallel --
        # no packing multiple experiments onto one GPU, so GPU-memory
        # probing and --max-parallel/--safety-margin/--gpu are all no-ops.
        if rank == 0:
            print(
                f"\n=== DDP mode: {world_size} ranks, running {len(configs)} experiments "
                f"sequentially (--max-parallel/--safety-margin/--gpu ignored) ===\n"
            )
        run_ddp_sweep(configs, dataset_cache_by_name, signature_of_name, rank, local_rank, world_size, scratch_dir)
        cleanup_distributed()
        return

    if args.max_parallel is not None:
        cost_by_name = {cfg.experiment_name: 1 for cfg in configs}
        usable = args.max_parallel
        print(f"\n=== Skipping GPU probe: fixed at {args.max_parallel} concurrent workers ===")
    else:
        free_bytes = query_free_memory_bytes(args.gpu)
        usable = int(free_bytes * args.safety_margin)
        print(f"\nGPU {args.gpu}: {free_bytes / 1e9:.2f} GB free, {usable / 1e9:.2f} GB usable budget")
        cost_by_name = compute_costs(configs, dataset_cache_by_name, scratch_dir, args.gpu)
        avg_cost = sum(cost_by_name.values()) / len(cost_by_name)
        print(f"Estimated max concurrency: ~{max(1, int(usable / avg_cost))} experiments")

    print(f"\n=== Running {len(configs)} experiments (parallel) ===\n")
    # Tee'd, unlike everything above it: the scheduler's [launch]/[done]/
    # [FAILED] lines are the only per-experiment verdict this path produces,
    # and on RunAI plain stdout is the pod log -- which is discarded the
    # moment the pod is deleted or preempted. The DDP path needs no
    # equivalent: its [FAILED] print already sits inside run_ddp_sweep()'s
    # per-experiment tee_stdio().
    with tee_stdio(scratch_dir / "logs" / "_orchestrator.log"):
        results = run_scheduler(
            configs, cost_by_name, usable, dataset_cache_by_name, scratch_dir, args.gpu, args.poll_interval
        )

        print(f"\n{'=' * 60}")
        print(f"=== {len(results['succeeded'])} succeeded, {len(results['failed'])} failed ===")
        if results["failed"]:
            print(f"Failed experiments (see {scratch_dir}/logs/<name>.log):")
            for name in results["failed"]:
                print(f"  - {name}")
        print(f"{'=' * 60}")
        print(f"Run fates: python -m utils.run_status {scratch_dir}")


if __name__ == "__main__":
    main()
