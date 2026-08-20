"""Smoke test for utils/eval_harness.py against a tiny model -- no real checkpoint needed.
Run directly: python tests/test_eval_harness.py
This is slow-ish (downloads small eval datasets on first run) but should complete in a few
minutes; it exists to catch integration breakage (wrong task names, adapter mismatches, the
custom MMLU task YAML) before spending real training/eval compute on Clariden.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_run_eval_suite_completes_on_tiny_model():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from config import Config
    from utils.eval_harness import run_eval_suite

    hf_config = AutoConfig.from_pretrained(
        "openai-community/gpt2", n_layer=2, n_head=2, n_embd=32,
    )
    model = AutoModelForCausalLM.from_config(hf_config)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    cfg = Config(device="cpu")
    scores = run_eval_suite(model, tokenizer, cfg, batch_size=2)

    from utils.eval_harness import ALL_TASKS
    assert set(scores.keys()) == set(ALL_TASKS), (set(scores.keys()), set(ALL_TASKS))
    for task, score in scores.items():
        assert isinstance(score, float), (task, score)
        assert math.isnan(score) or 0.0 <= score <= 1.0, (task, score)
    n_nan = sum(1 for s in scores.values() if math.isnan(s))
    print(f"OK: {len(scores)} tasks scored, {n_nan} unavailable (nan)")


def test_metric_key_prefers_acc_norm_over_acc():
    """Regression test: utils.eval_harness._pick_metric_key must pick acc_norm over acc when a
    task reports both. piqa's metric_list declares acc before acc_norm, so a naive
    first-prefix-match picker (the original bug) silently returns the weaker, length-biased raw
    acc instead of the acc_norm figure OPUS -- and virtually every published leaderboard --
    actually reports for piqa/hellaswag/arc_easy/arc_challenge. Runs lm_eval.simple_evaluate
    directly on a single task with a small `limit`, so this only hits the network for piqa, not
    the full slow suite.
    """
    import lm_eval
    import lm_eval.tasks
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from utils.eval_harness import CUSTOM_TASKS_DIR, _pick_metric_key

    hf_config = AutoConfig.from_pretrained(
        "openai-community/gpt2", n_layer=2, n_head=2, n_embd=32,
    )
    model = AutoModelForCausalLM.from_config(hf_config)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=2, device="cpu")
    task_manager = lm_eval.tasks.TaskManager(include_path=str(CUSTOM_TASKS_DIR))

    results = lm_eval.simple_evaluate(
        model=lm, tasks=["piqa"], num_fewshot=0, task_manager=task_manager, limit=20,
    )
    task_results = results["results"]["piqa"]

    # Confirms this fixture actually exercises the ordering bug: piqa must report both metrics.
    assert "acc,none" in task_results, task_results
    assert "acc_norm,none" in task_results, task_results

    picked_key = _pick_metric_key(task_results)
    assert picked_key == "acc_norm,none", (picked_key, task_results)
    assert float(task_results[picked_key]) == task_results["acc_norm,none"]
    print(f"OK: metric-key picker chose {picked_key!r} for piqa (not 'acc,none')")


def test_suite_averages_reports_contributing_task_counts_and_fraction_scale():
    """Fast unit test (no model, no network): suite_averages skips nan tasks, so the sample
    size behind each average is otherwise invisible -- on a real run most of OOD_TASKS is
    expected to be nan (BBH exceeds GPT-2's context, AX-b/AX-g are unregistered, StoryCloze is
    gated), leaving `ood_avg` a 1-task figure printed next to OPUS's genuine 6-benchmark
    average. in_domain_n/ood_n make that visible; compare_to_opus.py prints them.

    Also pins the SCALE: averages stay fractions in [0, 1], the same scale run_eval_suite
    returns. compare_to_opus.py is the only place that converts to OPUS's 0-100 percentages,
    and it must stay the only place.
    """
    from utils.eval_harness import IN_DOMAIN_TASKS, OOD_TASKS, suite_averages

    scores = {t: float("nan") for t in IN_DOMAIN_TASKS + OOD_TASKS}
    scores["hellaswag"] = 0.30
    scores["piqa"] = 0.40
    scores["race"] = 0.25

    averages = suite_averages(scores)
    assert averages["in_domain_avg"] == 0.35, averages  # mean of the 2 non-nan in-domain tasks
    assert averages["ood_avg"] == 0.25, averages
    assert averages["in_domain_n"] == 2, averages
    assert averages["ood_n"] == 1, averages
    assert 0.0 <= averages["in_domain_avg"] <= 1.0, "averages must stay fractions, not percent"

    # All-nan degrades to nan with a zero count rather than raising or returning 0.0.
    all_nan = suite_averages({t: float("nan") for t in IN_DOMAIN_TASKS + OOD_TASKS})
    assert math.isnan(all_nan["in_domain_avg"]) and all_nan["in_domain_n"] == 0, all_nan
    assert math.isnan(all_nan["ood_avg"]) and all_nan["ood_n"] == 0, all_nan
    print("OK: suite_averages reports contributing task counts on the [0, 1] fraction scale")


def test_evaluate_checkpoint_resolves_the_device_instead_of_pinning_cpu():
    """Regression test: evaluate_checkpoint.py never called resolve_device(), and cfg.device
    defaults to "" (neither shipped OPUS config sets it), so `model.to(cfg.device if
    cfg.device else "cpu")` silently pinned a 1.5B-param model to CPU on a GPU node -- an
    effective hang for a 22-task eval suite.

    torch.cuda.is_available is stubbed True so the GPU-node case can be checked on a CPU dev
    machine; the model is a stub whose .to() only records its argument, so nothing real is
    ever moved to a device that isn't there.
    """
    import tempfile
    from pathlib import Path

    import torch
    import yaml

    import evaluate_checkpoint as ec

    recorded = {}

    class _StubModel:
        hf = "HF-MODEL-SENTINEL"

        def load_state_dict(self, state_dict):
            recorded["loaded"] = True

        def to(self, device):
            recorded["moved_to"] = device
            return self

    def fake_run_eval_suite(model, tokenizer, cfg, batch_size=16, tasks=None):
        recorded["model"] = model
        recorded["cfg_device"] = cfg.device
        return {"winogrande": 0.5}

    originals = (ec.build_model, ec.run_eval_suite, ec.torch.load,
                 torch.cuda.is_available, sys.argv)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_path = tmp / "tiny.yaml"
        cfg_path.write_text(yaml.dump({"model_type": "tiny_gpt", "use_wandb": False}))
        out_path = tmp / "eval_scores.json"
        ckpt_path = tmp / "fake.pt"
        ckpt_path.write_bytes(b"")  # never really read: torch.load is stubbed below

        ec.build_model = lambda vocab_size, cfg: _StubModel()
        ec.run_eval_suite = fake_run_eval_suite
        ec.torch.load = lambda *a, **kw: {"model": {}}
        torch.cuda.is_available = lambda: True
        sys.argv = ["evaluate_checkpoint.py", "--checkpoint", str(ckpt_path),
                    "--config", str(cfg_path), "--out", str(out_path)]
        try:
            ec.main()
        finally:
            (ec.build_model, ec.run_eval_suite, ec.torch.load,
             torch.cuda.is_available, sys.argv) = originals

        assert recorded["moved_to"] == "cuda", (
            f"model was moved to {recorded['moved_to']!r}, not the resolved device -- "
            f"evaluate_checkpoint.py is pinning CPU again"
        )
        assert recorded["cfg_device"] == "cuda", (
            f"cfg.device reached run_eval_suite as {recorded['cfg_device']!r}; "
            f"resolve_device() was not applied"
        )
        assert recorded["model"] == "HF-MODEL-SENTINEL", recorded
        assert out_path.exists()
    print("OK: evaluate_checkpoint.py resolves the device rather than defaulting to CPU")


def test_evaluate_checkpoint_script_writes_eval_scores_json():
    """Exercises evaluate_checkpoint.py's load-then-eval path end-to-end against a tiny
    from-scratch model and checkpoint, so it doesn't need a real trained model."""
    import json
    import signal
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    import torch
    import yaml

    # utils/run_status.py references signal.SIGHUP in a module-level default argument
    # (meant for Linux HPC clusters), which doesn't exist on Windows -- importing
    # utils.experiment_worker below would otherwise raise AttributeError before this
    # test can run. Same shim as tests/test_checkpoint_saving.py; no-op on platforms
    # that already have SIGHUP, changes no runtime behavior under test.
    if not hasattr(signal, "SIGHUP"):
        signal.SIGHUP = signal.SIGTERM

    from config import Config
    from models.model import build_model
    from utils.experiment_worker import _save_checkpoint

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_path = tmp / "tiny.yaml"
        cfg_path.write_text(yaml.dump({
            "model_type": "hf_pretrained",
            "hf_model_name": "openai-community/gpt2",
            "d_model": 32, "n_layers": 2, "n_heads": 2, "block": 16,
            "use_wandb": False,
        }))

        cfg = Config(model_type="hf_pretrained", hf_model_name="openai-community/gpt2",
                     block=16, use_wandb=False)
        model = build_model(vocab_size=50257, cfg=cfg)
        from config import ExperimentConfig
        exp_cfg = ExperimentConfig(experiment_name="tiny", model_type="hf_pretrained",
                                    hf_model_name="openai-community/gpt2", block=16, use_wandb=False)
        ckpt_path = _save_checkpoint(exp_cfg, model, None, tmp)

        out_path = tmp / "eval_scores.json"
        # --tasks restricts this run to a 2-task subset (one real success-path task, one
        # guaranteed-nan task -- super_glue_axb has no registered task in the installed
        # lm-eval-harness version, see the module docstring above) instead of the full
        # ALL_TASKS suite (22 tasks), which takes ~30-90 minutes on a CPU dev machine
        # (confirmed empirically). This exercises the same evaluate_checkpoint.py plumbing
        # (checkpoint load -> build_model -> run_eval_suite -> suite_averages -> JSON write)
        # end-to-end in about a minute; ALL_TASKS' full breadth is already covered by
        # test_run_eval_suite_completes_on_tiny_model above (unchanged by this change) and by
        # a real run on GPU hardware.
        result = subprocess.run(
            [sys.executable, "evaluate_checkpoint.py",
             "--checkpoint", str(ckpt_path), "--config", str(cfg_path), "--out", str(out_path),
             "--tasks", "winogrande,super_glue_axb"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "in_domain_avg" in data and "ood_avg" in data, data
        assert "winogrande" in data and "super_glue_axb" in data, data
        assert math.isnan(data["super_glue_axb"]), data  # unregistered task -> always nan
        print(f"OK: evaluate_checkpoint.py wrote {out_path} (trimmed --tasks subset)")


if __name__ == "__main__":
    test_suite_averages_reports_contributing_task_counts_and_fraction_scale()
    test_evaluate_checkpoint_resolves_the_device_instead_of_pinning_cpu()
    test_metric_key_prefers_acc_norm_over_acc()
    test_run_eval_suite_completes_on_tiny_model()
    test_evaluate_checkpoint_script_writes_eval_scores_json()
    print("All tests passed.")
