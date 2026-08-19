"""Tests for utils/distributed_utils.py's process-group setup.
Run directly: python tests/test_distributed_utils.py
"""
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _with_torchrun_env(fn):
    """Run fn() with torchrun's env vars set, restoring whatever was there before."""
    keys = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
    backup = {k: os.environ.get(k) for k in keys}
    os.environ.update({"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"})
    try:
        return fn()
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_setup_distributed_uses_a_generous_collective_timeout():
    """Regression test: parallel_experiments.run_ddp_sweep() holds every rank at a
    dist.barrier() at each config boundary while rank 0 alone runs the end-of-training OPUS
    eval suite (~30-90 minutes, measured). Torch's default process-group timeout is 10 min
    (NCCL) / 30 min (gloo), so with the default the waiting ranks time out and NCCL's watchdog
    aborts the entire job -- after a possibly multi-day training run already succeeded.
    setup_distributed() must therefore pass an explicit, generous timeout.

    dist.init_process_group is stubbed: the point is which arguments are passed, and a real
    2-rank rendezvous can't be created from one process anyway.
    """
    import torch.distributed as dist
    from utils import distributed_utils

    captured = {}
    original = dist.init_process_group
    dist.init_process_group = lambda *a, **kw: captured.update(kw, _args=a)
    try:
        result = _with_torchrun_env(distributed_utils.setup_distributed)
    finally:
        dist.init_process_group = original

    assert result == (0, 0, 2), result
    assert "timeout" in captured, (
        "setup_distributed() passed no explicit timeout -- torch's 10-30 min default will "
        f"abort the job while rank 0 runs the eval suite. Got: {sorted(captured)}"
    )
    assert isinstance(captured["timeout"], datetime.timedelta), captured["timeout"]
    assert captured["timeout"] >= datetime.timedelta(hours=2), (
        f"timeout={captured['timeout']} leaves no margin above the ~90 min worst-case eval "
        f"suite runtime this branch measured"
    )
    print(f"OK: setup_distributed passes timeout={captured['timeout']}")


def test_setup_distributed_is_a_noop_without_torchrun():
    """Unchanged behaviour: no WORLD_SIZE means single-process, and torch.distributed must
    not be touched at all (every non-sweep entry point relies on this)."""
    import torch.distributed as dist
    from utils import distributed_utils

    backup = os.environ.pop("WORLD_SIZE", None)
    called = []
    original = dist.init_process_group
    dist.init_process_group = lambda *a, **kw: called.append((a, kw))
    try:
        result = distributed_utils.setup_distributed()
    finally:
        dist.init_process_group = original
        if backup is not None:
            os.environ["WORLD_SIZE"] = backup

    assert result == (0, 0, 1), result
    assert not called, "single-process path must not initialize a process group"
    print("OK: setup_distributed is a no-op without torchrun")


if __name__ == "__main__":
    test_setup_distributed_uses_a_generous_collective_timeout()
    test_setup_distributed_is_a_noop_without_torchrun()
    print("All tests passed.")
