"""Test for compare_to_opus.py using fixture eval_scores.json files (no real run needed).
Run directly: python tests/test_compare_to_opus.py

The fixtures below deliberately use the SCALE THE REAL PIPELINE PRODUCES: fractions in [0, 1]
(lm-evaluation-harness's native accuracy scale -- see utils/eval_harness.run_eval_suite and
tests/test_eval_harness.py's `0.0 <= score <= 1.0` assertion), NOT percentages. An earlier
version of this test hand-authored `"in_domain_avg": 42.0`, a value the real pipeline can
never emit, which is exactly why it failed to notice that compare_to_opus.py printed our
fractions unscaled in the same column as OPUS's 0-100 published figures -- a silent 100x
mismatch in the one artifact this whole comparison exists to produce.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(*run_dirs):
    result = subprocess.run(
        [sys.executable, "compare_to_opus.py", *[str(d) for d in run_dirs]],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result


def test_compare_to_opus_scales_fractions_to_percent():
    """Our rows come from eval_scores.json as fractions and must be printed on OPUS's 0-100
    scale, so the two are actually comparable in the same column."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        router_dir = tmp / "router_run"
        router_dir.mkdir()
        (router_dir / "eval_scores.json").write_text(json.dumps({
            "in_domain_avg": 0.42, "ood_avg": 0.405, "in_domain_n": 12, "ood_n": 1,
        }))

        control_dir = tmp / "control_run"
        control_dir.mkdir()
        (control_dir / "eval_scores.json").write_text(json.dumps({
            "in_domain_avg": 0.40, "ood_avg": 0.38, "in_domain_n": 12, "ood_n": 1,
        }))

        result = _run(router_dir, control_dir)
        assert result.returncode == 0, result.stderr
        out = result.stdout

        # The fix: 0.42 -> "42.00", never "0.42".
        assert "42.00" in out, out
        assert "40.50" in out, out  # router OOD 0.405
        assert "40.00" in out and "38.00" in out, out  # control rows
        assert "0.42" not in out and "0.40" not in out, (
            "raw fractions leaked into the table -- the 100x scale bug is back:\n" + out
        )
        # OPUS's own hardcoded percentages are untouched.
        assert "41.75" in out and "OPUS" in out, out
        print("OK: our fractions are printed on OPUS's 0-100 percentage scale")


def test_compare_to_opus_shows_contributing_task_counts():
    """suite_averages skips nan tasks, so `ood_avg` is often a 1-task average printed next to
    OPUS's genuine 6-benchmark figure. The table must say so."""
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "router_run"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_scores.json").write_text(json.dumps({
            "in_domain_avg": 0.35, "ood_avg": 0.30, "in_domain_n": 11, "ood_n": 1,
        }))

        result = _run(run_dir)
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "(11/12)" in out, out
        assert "(1/10)" in out, out
        print("OK: contributing-task counts are shown on our rows")


def test_task_count_denominators_match_eval_harness():
    """compare_to_opus.py hardcodes the denominators to stay import-light (utils.eval_harness
    pulls in lm_eval); this is the drift guard for that."""
    import compare_to_opus
    from utils.eval_harness import IN_DOMAIN_TASKS, OOD_TASKS

    assert compare_to_opus.N_IN_DOMAIN_TASKS == len(IN_DOMAIN_TASKS)
    assert compare_to_opus.N_OOD_TASKS == len(OOD_TASKS)
    print(f"OK: denominators match eval_harness ({len(IN_DOMAIN_TASKS)}/{len(OOD_TASKS)})")


def test_missing_and_malformed_eval_scores_give_a_clear_error():
    """A raw FileNotFoundError/KeyError traceback is a poor way to end a multi-day pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        missing = tmp / "no_such_run"
        missing.mkdir()
        result = _run(missing)
        assert result.returncode != 0
        assert "eval_scores.json" in result.stderr and "Traceback" not in result.stderr, result.stderr

        malformed = tmp / "malformed_run"
        malformed.mkdir()
        (malformed / "eval_scores.json").write_text("{not json")
        result = _run(malformed)
        assert result.returncode != 0
        assert "not valid JSON" in result.stderr and "Traceback" not in result.stderr, result.stderr

        incomplete = tmp / "incomplete_run"
        incomplete.mkdir()
        (incomplete / "eval_scores.json").write_text(json.dumps({"winogrande": 0.5}))
        result = _run(incomplete)
        assert result.returncode != 0
        assert "in_domain_avg" in result.stderr and "Traceback" not in result.stderr, result.stderr
        print("OK: missing/malformed/incomplete eval_scores.json give one-line errors")


if __name__ == "__main__":
    test_compare_to_opus_scales_fractions_to_percent()
    test_compare_to_opus_shows_contributing_task_counts()
    test_task_count_denominators_match_eval_harness()
    test_missing_and_malformed_eval_scores_give_a_clear_error()
    print("All tests passed.")
