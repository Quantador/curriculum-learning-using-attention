"""Print a comparison table: this project's eval_scores.json result(s) against OPUS's
published Table 3 (GPT-2 XL + Muon + FineWeb, in-domain) and Table 5 (out-of-distribution)
numbers, arXiv:2602.05400.

Usage:
    python compare_to_opus.py results/opus_gpt2xl_muon_fineweb \
        [results/opus_gpt2xl_muon_fineweb_random_baseline]

Scale: eval_scores.json stores fractions in [0, 1] (lm-evaluation-harness's native accuracy
scale -- see utils/eval_harness.py), while the OPUS tables below are percentages on a 0-100
scale. Our rows are therefore multiplied by 100 at the print site so both land in the same
column on the same scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Denominators for the "(N/total) tasks scored" annotation on our own rows. Deliberately
# hardcoded rather than imported from utils.eval_harness: importing that module pulls in
# lm_eval (a heavy, optional-at-report-time dependency), and this script is meant to run
# anywhere the JSON files are, off nothing but the standard library. tests/test_compare_to_opus.py
# asserts these stay equal to len(IN_DOMAIN_TASKS)/len(OOD_TASKS) in utils/eval_harness.py,
# so they cannot silently drift apart.
N_IN_DOMAIN_TASKS = 12
N_OOD_TASKS = 10

# OPUS Table 3, "GPT-2 XL with Muon optimizer on 30B update tokens of FineWeb", Avg. column.
OPUS_IN_DOMAIN_AVG = {
    "Random": 40.29, "PPL": 39.82, "GREATS": 39.23, "QuRating": 40.72, "DSIR": 40.68,
    "DCLM-FastText": 40.60, "FineWeb-Edu": 40.74, "UltraFineweb": 39.64, "OPUS": 41.75,
}
# OPUS Table 5, out-of-distribution average, same GPT-2 XL checkpoints.
OPUS_OOD_AVG = {"Random": 38.09, "OPUS": 40.07}

_NAME_W, _IND_W, _OOD_W = 20, 22, 20


def _load(run_dir: str) -> dict:
    """Read <run_dir>/eval_scores.json, failing with a one-line actionable message rather
    than a raw traceback -- this is the last step of a multi-day pipeline."""
    path = Path(run_dir) / "eval_scores.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        sys.exit(
            f"error: no eval_scores.json in {run_dir!r} (looked for {path}). Run training "
            f"with save_model_at_end, or produce it with:\n"
            f"    python evaluate_checkpoint.py --checkpoint <ckpt.pt> --config <cfg.yaml> "
            f"--out {path}"
        )
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {path} is not valid JSON ({exc}); re-generate it with evaluate_checkpoint.py.")

    missing = [k for k in ("in_domain_avg", "ood_avg") if k not in data]
    if missing:
        sys.exit(
            f"error: {path} is missing {', '.join(missing)} (found keys: {sorted(data)}). "
            f"It should be written by utils.eval_harness.suite_averages via "
            f"evaluate_checkpoint.py or the end-of-training eval; re-generate it."
        )
    return data


def _our_row(label: str, data: dict) -> str:
    """One of our own rows: fractions scaled to OPUS's 0-100 percentage scale, with the
    number of tasks that actually contributed to each average (nan/unavailable tasks are
    skipped by suite_averages, which would otherwise hide the real sample size)."""
    ind = f"{data['in_domain_avg'] * 100:.2f} ({data.get('in_domain_n', '?')}/{N_IN_DOMAIN_TASKS})"
    ood = f"{data['ood_avg'] * 100:.2f} ({data.get('ood_n', '?')}/{N_OOD_TASKS})"
    return f"{label:<{_NAME_W}}{ind:>{_IND_W}}{ood:>{_OOD_W}}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("router_run_dir")
    parser.add_argument("control_run_dir", nargs="?", default=None)
    args = parser.parse_args()

    router = _load(args.router_run_dir)
    control = _load(args.control_run_dir) if args.control_run_dir else None

    print(f"{'Method':<{_NAME_W}}{'In-domain avg':>{_IND_W}}{'OOD avg':>{_OOD_W}}")
    print("-" * (_NAME_W + _IND_W + _OOD_W))
    for name, avg in OPUS_IN_DOMAIN_AVG.items():
        # OPUS's published figures are already percentages; no per-task breakdown is
        # available for them, so no (N/total) count is shown on these rows.
        ood = OPUS_OOD_AVG.get(name, float("nan"))
        print(f"{name:<{_NAME_W}}{avg:>{_IND_W}.2f}{ood:>{_OOD_W}.2f}")
    if control is not None:
        print(_our_row("Our random control", control))
    print(_our_row("Our router", router))
    print(
        f"\n(N/{N_IN_DOMAIN_TASKS}) and (N/{N_OOD_TASKS}) = tasks that actually scored "
        f"on our runs; unavailable tasks score nan and are excluded from the average, so a "
        f"low N means the figure is not a like-for-like average of OPUS's benchmark set."
    )


if __name__ == "__main__":
    main()
