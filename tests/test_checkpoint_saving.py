"""Tests for the checkpoint-saving and end-of-training-eval helpers in
utils/experiment_worker.py.
Run directly: python tests/test_checkpoint_saving.py
"""
import contextlib
import json
import math
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for the shared tiny-HF fixture

# utils/run_status.py references signal.SIGHUP in a module-level default
# argument (it's meant for Linux HPC clusters), which doesn't exist on
# Windows -- importing utils.experiment_worker below would otherwise raise
# AttributeError before any test runs. This shim only adds the missing
# attribute for this process; it's a no-op on platforms that already have
# SIGHUP and changes no runtime behavior under test.
if not hasattr(signal, "SIGHUP"):
    signal.SIGHUP = signal.SIGTERM

import torch


def test_save_checkpoint_writes_loadable_state_dict():
    from config import ExperimentConfig
    from utils.experiment_worker import _save_checkpoint

    model = torch.nn.Linear(4, 4)
    router = torch.nn.Linear(8, 1)
    cfg = ExperimentConfig(experiment_name="ckpt_test")

    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        path = _save_checkpoint(cfg, model, router, save_dir)
        assert path == save_dir / "ckpt_test.pt"
        assert path.exists()

        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state.keys()) == {"model", "router"}

        reloaded = torch.nn.Linear(4, 4)
        reloaded.load_state_dict(state["model"])  # must not raise
        for p1, p2 in zip(model.parameters(), reloaded.parameters()):
            assert torch.equal(p1, p2)
    print("OK: checkpoint round-trips")


def test_save_checkpoint_without_router():
    from config import ExperimentConfig
    from utils.experiment_worker import _save_checkpoint

    model = torch.nn.Linear(4, 4)
    cfg = ExperimentConfig(experiment_name="ckpt_test_no_router")

    with tempfile.TemporaryDirectory() as tmp:
        path = _save_checkpoint(cfg, model, None, Path(tmp))
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert set(state.keys()) == {"model"}
    print("OK: checkpoint without router omits the 'router' key")


@contextlib.contextmanager
def _patched_run_eval_suite(fake):
    """Swap utils.eval_harness.run_eval_suite for `fake` for the duration of the block.

    _end_of_training_eval() does `from utils.eval_harness import run_eval_suite` *inside* the
    function, so the name is resolved off the module object at call time -- patching the
    attribute here is what the production code will actually pick up. suite_averages is
    deliberately left as the real implementation so the JSON these tests inspect comes out of
    the real aggregation path.
    """
    import utils.eval_harness as eval_harness

    original = eval_harness.run_eval_suite
    eval_harness.run_eval_suite = fake
    try:
        yield
    finally:
        eval_harness.run_eval_suite = original


@contextlib.contextmanager
def _single_process_gloo_group():
    """A real, single-process CPU process group, so DistributedDataParallel can legitimately
    wrap a model without any GPU or a second rank (same approach as
    tests/test_muon_optimizer.py's DDP test)."""
    import torch.distributed as dist

    # This torch build's gloo/TCPStore rendezvous on Windows fails ("use_libuv was requested
    # but PyTorch was built without libuv support") unless libuv is explicitly disabled.
    os.environ.setdefault("USE_LIBUV", "0")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_end_of_training_eval_unwraps_ddp_before_looking_for_hf():
    """Regression test for the end-of-training eval call: every training loop returns the DDP-WRAPPED
    model when cfg.world_size > 1 (the real --submit Slurm path), and DDP has no __getattr__
    passthrough, so the original `model.hf if hasattr(model, "hf") else model` handed the raw
    DistributedDataParallel object to lm-eval's HFLM -- which reads self._model.device and
    dies with AttributeError, outside run_eval_suite's per-task try/except.

    Uses a real gloo DDP wrap (not a stand-in) precisely because the bug WAS the DDP
    attribute-proxy behaviour; asserting on a hand-rolled fake wrapper would not prove it.
    """
    from config import ExperimentConfig
    from utils.experiment_worker import _end_of_training_eval
    from utils.metrics import MetricsTracker
    from torch.nn.parallel import DistributedDataParallel as DDP

    from test_muon_optimizer import _make_tiny_hf_gpt2

    model = _make_tiny_hf_gpt2()
    cfg = ExperimentConfig(experiment_name="ddp_eval_test", model_type="hf_pretrained",
                           hf_model_name="openai-community/gpt2", block=16, use_wandb=False)
    metrics = MetricsTracker(cfg.experiment_name, use_wandb=False)

    captured = {}

    def fake_run_eval_suite(m, tokenizer, cfg_, **kwargs):
        captured["model"] = m
        # Two real task names so the REAL suite_averages runs over them: one in-domain, one
        # OOD, plus one nan to exercise the nan-skipping / n-counting path.
        return {"winogrande": 0.5, "race": 0.25, "super_glue_axb": float("nan")}

    with _single_process_gloo_group():
        wrapped = DDP(model, device_ids=None)
        # The precondition that made the original code wrong. If a future torch adds an
        # attribute passthrough this assert fires and this test needs revisiting.
        assert not hasattr(wrapped, "hf"), "DDP now proxies .hf -- revisit the unwrap logic"

        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with _patched_run_eval_suite(fake_run_eval_suite):
                _end_of_training_eval(cfg, wrapped, tokenizer=None, experiment_metrics=metrics)

            assert "model" in captured, "run_eval_suite was never called for a DDP-wrapped model"
            assert captured["model"] is model.hf, (
                f"expected the underlying HF model, got {type(captured['model']).__name__}"
            )

            out = Path(tmp) / "results" / cfg.experiment_name / "eval_scores.json"
            assert out.exists(), f"{out} not written"
            data = json.loads(out.read_text())
            assert data["in_domain_avg"] == 0.5 and data["ood_avg"] == 0.25, data
            # Finding 7: the sample size behind each average is recorded, not implied.
            assert data["in_domain_n"] == 1 and data["ood_n"] == 1, data
            assert math.isnan(data["super_glue_axb"]), data
    print("OK: end-of-training eval unwraps DDP and hands HFLM the real HF model")


def test_end_of_training_eval_skips_model_without_hf():
    """model_type='tiny_gpt' has no .hf for lm-eval's HFLM to consume. That must be a clean
    one-line skip, not a crash and not a bare model handed to the harness."""
    import torch

    from config import ExperimentConfig
    from utils.experiment_worker import _end_of_training_eval
    from utils.metrics import MetricsTracker

    cfg = ExperimentConfig(experiment_name="tiny_gpt_eval_test", model_type="tiny_gpt",
                           use_wandb=False)
    metrics = MetricsTracker(cfg.experiment_name, use_wandb=False)
    calls = []

    def fake_run_eval_suite(*args, **kwargs):
        calls.append(args)
        raise AssertionError("run_eval_suite must not be called for a model without .hf")

    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        with _patched_run_eval_suite(fake_run_eval_suite):
            _end_of_training_eval(cfg, torch.nn.Linear(4, 4), tokenizer=None,
                                  experiment_metrics=metrics)
        assert not calls
        assert not (Path(tmp) / "results").exists(), "skipped eval must write nothing"
    print("OK: end-of-training eval skips (does not crash on) a model with no .hf")


def test_end_of_training_eval_never_propagates_a_failure():
    """A failing eval suite must not destroy an otherwise-successful multi-day training run:
    the checkpoint is already saved, and run_single_experiment still has to log total_time_s
    after this call. So _end_of_training_eval swallows anything the eval stage raises."""
    from config import ExperimentConfig
    from utils.experiment_worker import _end_of_training_eval
    from utils.metrics import MetricsTracker

    from test_muon_optimizer import _make_tiny_hf_gpt2

    model = _make_tiny_hf_gpt2()
    cfg = ExperimentConfig(experiment_name="failing_eval_test", model_type="hf_pretrained",
                           hf_model_name="openai-community/gpt2", block=16, use_wandb=False)
    metrics = MetricsTracker(cfg.experiment_name, use_wandb=False)

    def exploding_run_eval_suite(*args, **kwargs):
        raise RuntimeError("CUDA out of memory (simulated eval-stage failure)")

    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
        with _patched_run_eval_suite(exploding_run_eval_suite):
            # Must return normally. Before the fix this exception escaped the whole
            # `try:` block in run_single_experiment, skipping total_time_s and recording
            # the finished training run as FAILED.
            assert _end_of_training_eval(cfg, model, tokenizer=None,
                                         experiment_metrics=metrics) is None
    print("OK: an eval-stage exception is contained, not propagated")


def test_run_single_experiment_logs_total_time_after_a_failed_eval():
    """The point of the containment above, checked at the call site: with the eval stage
    blowing up, run_single_experiment must still reach experiment_metrics.log(total_time_s=...)
    and return normally. Everything expensive (dataset, training loop, comparison print) is
    stubbed -- this exercises control flow, not training."""
    import types

    from config import ExperimentConfig
    from utils import experiment_worker
    from utils.metrics import MetricsTracker

    from test_muon_optimizer import _make_tiny_hf_gpt2

    model = _make_tiny_hf_gpt2()
    cfg = ExperimentConfig(experiment_name="total_time_test", model_type="hf_pretrained",
                           hf_model_name="openai-community/gpt2", block=16, use_wandb=False,
                           save_model_at_end=True, run_random_batch_baseline=True)

    class _FakeDataset:
        """Only what run_single_experiment itself touches: len() and .domain_names for
        DiversityTracker. The training loop that would really read it is stubbed out."""
        domain_names = None

        def __len__(self):
            return 4

    fake_ds = _FakeDataset()

    saved = {}
    originals = {
        "build_model": experiment_worker.build_model,
        "train_baseline": experiment_worker.train_baseline,
        "compare_runs_experiments": experiment_worker.compare_runs_experiments,
        "_save_checkpoint": experiment_worker._save_checkpoint,
    }
    experiment_worker.build_model = lambda vocab_size, cfg_: model
    experiment_worker.train_baseline = lambda **kwargs: model
    experiment_worker.compare_runs_experiments = lambda *a, **k: None
    experiment_worker._save_checkpoint = (
        lambda cfg_, m, r, d: saved.setdefault("path", Path(d) / "fake.pt")
    )

    def exploding_run_eval_suite(*args, **kwargs):
        raise RuntimeError("simulated eval-stage failure")

    try:
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with _patched_run_eval_suite(exploding_run_eval_suite):
                metrics = experiment_worker.run_single_experiment(
                    cfg=cfg,
                    tokenizer=types.SimpleNamespace(vocab_size=100),
                    train_ds=fake_ds, val_ds=fake_ds,
                    base_metrics=MetricsTracker("base", use_wandb=False),
                    router_metrics=MetricsTracker("router", use_wandb=False),
                    save_dir=Path(tmp) / "ckpt",
                )
    finally:
        for name, fn in originals.items():
            setattr(experiment_worker, name, fn)

    assert saved.get("path") is not None, "checkpoint step did not run"
    assert metrics.get_total_time() is not None, (
        "total_time_s was not logged after a failing eval stage"
    )
    print(f"OK: total_time_s logged ({metrics.get_total_time():.3f}s) despite a failing eval")


if __name__ == "__main__":
    test_save_checkpoint_writes_loadable_state_dict()
    test_save_checkpoint_without_router()
    test_end_of_training_eval_unwraps_ddp_before_looking_for_hf()
    test_end_of_training_eval_skips_model_without_hf()
    test_end_of_training_eval_never_propagates_a_failure()
    test_run_single_experiment_logs_total_time_after_a_failed_eval()
    print("All tests passed.")
