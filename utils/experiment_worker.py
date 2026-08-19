# experiment_worker.py
"""
Run exactly one experiment given a serialized config, as a standalone
process. Launched by parallel_experiments.py so that many experiments can
run concurrently on the same GPU, each in its own CUDA context.

    python experiment_worker.py --config <yaml> --dataset-cache <path>
                                [--status-dir <path>]
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import os
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from config import load_config_from_yaml
from data import get_tokenizer
from utils.shared_dataset import load_dataset_cache
from models.router import build_router, build_router_for_cfg, get_router_feature_dim
from rl_training import train_router_experiments, train_aux_baseline, compare_runs_experiments
from training import train_baseline
from models.model import build_model
from utils.metrics import MetricsTracker, DiversityTracker
from config import ExperimentConfig, load_config_from_yaml
from utils.general_utils import resolve_device, safe_name, set_seed
from utils import memory_snapshot, run_status


def _save_checkpoint(cfg, model, router_or_aux, save_dir: Path) -> Path:
    """Save the trained LM (and router/aux_net, if any) to <save_dir>/<name>.pt, per
    cfg.save_model_at_end's docstring in config.py. DDP wraps are unwrapped first so the
    saved state_dict's keys match a plain (unwrapped) model -- DDP otherwise prefixes every
    key with "module.", which a later model.load_state_dict(...) on an unwrapped model
    would reject."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{safe_name(cfg.experiment_name)}.pt"

    raw_model = model.module if isinstance(model, DDP) else model
    state = {"model": raw_model.state_dict()}
    if router_or_aux is not None:
        raw_router = router_or_aux.module if isinstance(router_or_aux, DDP) else router_or_aux
        state["router"] = raw_router.state_dict()

    torch.save(state, path)
    return path


def _end_of_training_eval(cfg, model, tokenizer, experiment_metrics) -> None:
    """Best-effort OPUS-comparable benchmark eval at the end of a training run.

    NEVER raises. The checkpoint written by _save_checkpoint() just before this call is
    already on disk and is a complete, independent recovery path (evaluate_checkpoint.py),
    so a failure here -- a transient HF Hub error, an OOM from running eval on the GPU
    training just finished on, an lm-eval registry skew -- must not mark a multi-day
    training run as failed, nor skip the caller's total_time_s logging.

    Called on rank 0 only (see run_single_experiment); `model` may still be the
    DDP-wrapped object every training loop returns when cfg.world_size > 1.
    """
    try:
        from utils.eval_harness import run_eval_suite, suite_averages
        import json

        # Unwrap DDP *before* asking for .hf: all three training loops return the
        # DDP-wrapped model when cfg.world_size > 1 (the real --submit Slurm path), and DDP
        # defines no __getattr__ passthrough, so hasattr(wrapper, "hf") is always False --
        # the raw DistributedDataParallel object would otherwise be handed to lm-eval's
        # HFLM, which reads self._model.device and raises AttributeError.
        raw_model = model.module if isinstance(model, DDP) else model
        hf_model = getattr(raw_model, "hf", None)
        if hf_model is None:
            # Not an error: run_eval_suite drives lm-eval's HFLM, which needs a real
            # transformers.PreTrainedModel. TinyGPT (the default model_type) has no .hf to
            # hand it, so there is simply nothing here for this harness to evaluate.
            print(
                "[eval] skipping OPUS-comparable eval suite: needs model_type='hf_pretrained' "
                f"(got {cfg.model_type!r})"
            )
            return

        print("\n=== Running OPUS-comparable eval suite ===")
        scores = run_eval_suite(hf_model, tokenizer, cfg)
        averages = suite_averages(scores)
        experiment_metrics.log(**{f"eval/{k}": v for k, v in scores.items()}, **averages)

        # Raw cfg.experiment_name (NOT safe_name): rl_training.py already writes into
        # results/<cfg.experiment_name>/ with the raw name, and both evaluate_checkpoint.py
        # and compare_to_opus.py assume that same directory shape. safe_name() is
        # deliberately scoped to the checkpoint .pt filename only.
        eval_out_dir = Path("results") / cfg.experiment_name
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        (eval_out_dir / "eval_scores.json").write_text(
            json.dumps({**scores, **averages}, indent=2)
        )
        print(json.dumps(averages, indent=2))
    except Exception:
        import traceback

        print(
            "[eval] OPUS-comparable eval suite failed; continuing without it (the checkpoint "
            "is already saved and independently evaluable via evaluate_checkpoint.py)."
        )
        traceback.print_exc()


def run_single_experiment(cfg: ExperimentConfig, tokenizer, train_ds, val_ds, base_metrics, router_metrics, save_dir: Path | None = None):
    """Run a single experiment with the given configuration.

    Shared by the bare in-process run_experiment() below and by
    utils/experiment_worker.py, which calls this once per config inside its
    own subprocess when running a sweep via run_scheduler().
    """
    # Every rank runs this function identically under DDP (cfg.world_size >
    # 1); gate the purely informational prints to rank 0 so a multi-GPU job
    # log doesn't repeat each line once per rank.
    if cfg.rank == 0:
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
        print(f"  run_aux_baseline: {cfg.run_aux_baseline}")
        print(f"  run_random_batch_baseline: {cfg.run_random_batch_baseline}")
        print(f"  run_random_pool_baseline: {cfg.run_random_pool_baseline}")

    # + cfg.rank: harmless (always +0) outside DDP; under DDP, diverges each
    # rank's data-selection/dropout randomness while model init still matches
    # across ranks regardless (DDP's constructor broadcasts rank 0's weights).
    set_seed(cfg.seed + cfg.rank)

    model_router = build_model(tokenizer.vocab_size, cfg)
    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds), domain_names=train_ds.domain_names)

    # Wall-clock time for the whole training call, uniform across all four
    # training loops below -- so runs can be compared on wall-clock time
    # later, not just final ppl (see MetricsTracker.save()'s "total_time_s").
    run_start = time.perf_counter()

    # try/finally so wandb.finish() always runs, even if training raises --
    # otherwise a failed run (e.g. an OOM) leaves its wandb run open, and
    # under run_ddp_sweep() (which runs every config sequentially in the same
    # process, unlike run_scheduler()'s one-subprocess-per-config isolation)
    # the NEXT config's wandb.init() call just reattaches to that still-open
    # run instead of starting its own -- so its metrics silently land under
    # the failed run's name instead of its own.
    try:
        if cfg.run_aux_baseline:
            # Supervised MSE alternative to the policy-gradient router (ablation
            # baseline) — same feature pipeline and top-k selection, different
            # training objective. See rl_training.train_aux_baseline().
            aux_net = build_router(
                d_input=get_router_feature_dim(cfg, model_router.block),
                arch="auxnet",
                d_hidden=cfg.aux_net_hidden,
            )
            model_router, _ = train_aux_baseline(
                cfg=cfg,
                model=model_router,
                aux_net=aux_net,
                train_ds=train_ds,
                val_ds=val_ds,
                tokenizer=tokenizer,
                metrics=experiment_metrics,
                diversity=router_div,
            )
        elif cfg.run_random_batch_baseline or cfg.run_random_pool_baseline:
            # Non-learned controls: uniform random selection, no router/aux_net.
            # training.train_baseline() already draws cfg.global_batch_size random
            # samples (split across ranks) from a cfg.pool-sized window each step,
            # so the "random batch the size of the pool" variant is just that same
            # function with global_batch_size widened to pool via replace() -- no
            # new training loop needed, and the widened pool-sized batch still
            # gets split across ranks like any other global_batch_size.
            # pool_mult=1 too: cfg.pool is a property (pool_mult * global_batch_size),
            # so widening global_batch_size alone would silently re-inflate pool by
            # another factor of pool_mult, shrinking the window to 1/pool_mult of
            # the dataset and making train_baseline's random.sample(window, batch)
            # discard most of each window instead of training on all of it.
            random_cfg = cfg if cfg.run_random_batch_baseline else replace(cfg, global_batch_size=cfg.pool, pool_mult=1)
            model_router = train_baseline(
                cfg=random_cfg,
                model=model_router,
                train_ds=train_ds,
                val_ds=val_ds,
                metrics=experiment_metrics,
                diversity=router_div,
            )
        else:
            router = build_router_for_cfg(
                cfg,
                sequence_size=model_router.block,
                vocab_size=tokenizer.vocab_size,
            )
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

        if cfg.save_model_at_end and save_dir is not None and cfg.rank == 0:
            router_obj = None
            if cfg.run_aux_baseline:
                router_obj = aux_net
            elif not (cfg.run_random_batch_baseline or cfg.run_random_pool_baseline):
                router_obj = router
            ckpt_path = _save_checkpoint(cfg, model_router, router_obj, save_dir)
            print(f"[checkpoint] saved to {ckpt_path}")

            # Best-effort, never raises -- see _end_of_training_eval's docstring. The
            # total_time_s logging below must happen even if the eval suite falls over.
            _end_of_training_eval(cfg, model_router, tokenizer, experiment_metrics)

        total_time_s = time.perf_counter() - run_start
        if cfg.rank == 0:
            experiment_metrics.log(total_time_s=total_time_s)
            print(f"\n=== Total run time: {total_time_s:.1f}s ({total_time_s / 3600:.2f}h) ===")
    finally:
        # The training loops above intentionally leave their wandb run open so
        # total_time_s lands in it too; this closes it once everything's logged
        # -- or, on a raised exception, closes whatever partial run is still
        # open so it doesn't bleed into the next config's wandb.init().
        if cfg.use_wandb and cfg.rank == 0:
            import wandb
            wandb.finish()

    if cfg.rank == 0:
        print("\n=== Comparing runs ===")
        compare_runs_experiments(
            base_metrics,
            router_metrics,
            experiment_metrics,
        )

    return experiment_metrics

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument(
        "--status-dir", default=None,
        help="Where to record how this run ended (see utils/run_status.py). "
             "Omitted when running this worker by hand; parallel_experiments.py "
             "always passes the sweep's status dir.",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to checkpoint the trained model into when cfg.save_model_at_end is "
             "set (omitted when running this worker by hand without --save-model).",
    )
    args = parser.parse_args()

    cfg = load_config_from_yaml(args.config)
    # Before anything touches a GPU: the YAML may have been written by a
    # process that could not see one (parallel_experiments.py --submit runs on
    # a login node), so the device is decided here, on the machine that trains.
    cfg = resolve_device(cfg)
    set_seed(cfg.seed)

    tokenizer = get_tokenizer(cfg.tokenizer_name)
    train_ds, val_ds = load_dataset_cache(args.dataset_cache, cfg)

    base_metrics = MetricsTracker.load("results/baseline_metrics.json")
    router_metrics = MetricsTracker.load("results/router_metrics.json")

    # Installed only once the expensive setup above is done: everything before
    # this point is fast and reproducible, so a kill during it needs no
    # explaining, while a kill during training is exactly what we can't
    # currently account for after the fact.
    if args.status_dir:
        run_status.install_signal_handlers(Path(args.status_dir).parent, role=f"worker.{os.getpid()}")

    name = safe_name(cfg.experiment_name)
    tracker = (
        run_status.track(args.status_dir, name, experiment=cfg.experiment_name,
                         config_path=args.config, wandb_project=cfg.wandb_project)
        if args.status_dir else contextlib.nullcontext()
    )
    # Snapshots go beside the run's other artifacts when the sweep gave us a
    # status dir (its parent is the scratch dir), else the cwd for a
    # hand-launched worker. Inside `tracker`, so on an OOM the allocator dump
    # happens before run_status records the exception -- the inner context
    # exits first, and the snapshot is only meaningful before unwinding.
    snapshot_dir = (Path(args.status_dir).parent / "snapshots") if args.status_dir else Path("snapshots")
    with tracker:
        with memory_snapshot.record(snapshot_dir, name, rank=cfg.rank):
            run_single_experiment(
                cfg=cfg,
                tokenizer=tokenizer,
                train_ds=train_ds,
                val_ds=val_ds,
                base_metrics=base_metrics,
                router_metrics=router_metrics,
                save_dir=Path(args.save_dir) if args.save_dir else None,
            )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
