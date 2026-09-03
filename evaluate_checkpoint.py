"""Standalone entry point: evaluate a saved checkpoint on the OPUS-comparable benchmark suite.

Usage:
    RUN=results/_parallel_run/<timestamp>_<name>
    python evaluate_checkpoint.py --checkpoint $RUN/checkpoints/<name>.pt \\
        --config configs/opus_gpt2xl_muon_fineweb.yaml --out $RUN/eval_scores.json \\
        [--batch-size 16] \\
        [--tasks task1,task2]  # optional: restrict to a task subset (default: all of ALL_TASKS)

Scores belong in the run's own <timestamp>_<name> directory, not a directory named after
the experiment alone: experiment_name repeats across runs (every sweep emits an
"experiment_baseline"), so a name-keyed path is a shared slot that later runs overwrite.

Also used automatically by utils/experiment_worker.py at the end of a training run when
cfg.save_model_at_end is set -- see run_single_experiment() in that file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import load_config_from_yaml
from data import get_tokenizer
from models.model import build_model
from utils.eval_harness import run_eval_suite, suite_averages
from utils.general_utils import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out", default=None,
        help="Default: eval_scores.json next to --checkpoint, i.e. inside that run's "
             "<timestamp>_<name>/checkpoints/ dir. To land it where the end-of-training "
             "eval writes -- and where compare_to_opus.py expects a run directory -- pass "
             "--out results/_parallel_run/<timestamp>_<name>/eval_scores.json (one level "
             "up, beside that run's configs/ and logs/).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--tasks", default=None,
        help="Comma-separated task names to run instead of the full ALL_TASKS suite (e.g. "
             "for fast local dev iteration). Default: every task in utils.eval_harness.ALL_TASKS.",
    )
    args = parser.parse_args()
    tasks = args.tasks.split(",") if args.tasks else None

    # resolve_device: cfg.device defaults to "" (the config may have been written on a
    # GPU-less login node), and neither shipped OPUS config sets it. Without this, the
    # .to() below would silently pin a 1.5B-param model to CPU on a GPU node -- an
    # effective hang for a 22-task eval suite. Same call, same reason, as
    # utils/experiment_worker.py's main().
    cfg = resolve_device(load_config_from_yaml(args.config))
    tokenizer = get_tokenizer(cfg.tokenizer_name)

    model = build_model(vocab_size=tokenizer.vocab_size, cfg=cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    # This .to() is what actually decides where eval runs: lm_eval's HFLM ignores its
    # `device=` argument when handed a pre-built model instance (see run_eval_suite).
    model.to(cfg.device)

    hf_model = model.hf if hasattr(model, "hf") else model
    scores = run_eval_suite(hf_model, tokenizer, cfg, batch_size=args.batch_size, tasks=tasks)
    averages = suite_averages(scores)

    out_path = Path(args.out) if args.out else Path(args.checkpoint).with_name("eval_scores.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**scores, **averages}, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(averages, indent=2))


if __name__ == "__main__":
    main()
