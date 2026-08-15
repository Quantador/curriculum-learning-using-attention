# memory_snapshot.py
"""
CUDA allocator snapshots: which allocations, from which lines, at OOM time.

torch.cuda.max_memory_allocated() (what utils/gpu_memory_probe.py reports)
gives one number for a whole run. That's the right tool for "will this config
fit"; it is useless for "which tensor ate 40 GB". This records a stack trace
per allocation/free and dumps a snapshot you drop on

    https://docs.pytorch.org/memory_viz

to get a flame-graph timeline of the allocator.

Off unless CL_MEMORY_SNAPSHOT is set, because recording is not free: every
allocation walks the Python stack, and max_entries traces are held in memory.
Enable it for one debugging run, not for a sweep you care about the timing of.

    CL_MEMORY_SNAPSHOT=1 python parallel_experiments.py --config ...

Files land in <scratch_dir>/snapshots/ as <label>.rank<N>.<outcome>.pickle,
where outcome is 'oom', 'error' or 'final'. The OOM case is the point of the
whole thing: the dump happens inside the exception path, so the snapshot
captures the allocator exactly as it was when the allocation failed, rather
than after unwinding has freed everything.

A ready-made variant of this (periodic dumps, per-rank folders) is vendored
at GhostSuite/examples/torchtitan/torchtitan/tools/profiling.py --
maybe_enable_memory_snapshot(); it is coupled to torchtitan's own config
objects, so this is the same pattern reduced to what this repo needs.
"""
from __future__ import annotations

import os

from contextlib import contextmanager
from pathlib import Path

import torch

# How many alloc/free events to retain. torchtitan's default; ~tens of MB of
# pickle at this size, and enough to cover several training steps.
MAX_ENTRIES = 100_000

_ENV_VAR = "CL_MEMORY_SNAPSHOT"


def enabled() -> bool:
    """True when CL_MEMORY_SNAPSHOT is set to something other than 0/false/no."""
    return os.environ.get(_ENV_VAR, "").strip().lower() not in ("", "0", "false", "no")


def _dump(dump_dir: Path, label: str, rank: int, outcome: str) -> None:
    path = dump_dir / f"{label}.rank{rank}.{outcome}.pickle"
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))
        print(f"[memory-snapshot] wrote {path} — view at https://docs.pytorch.org/memory_viz")
    except Exception as exc:
        # Never let a diagnostic dump replace the error it was meant to
        # explain: an OOM has already left the CUDA context in a rough state,
        # and a failure to serialize here would otherwise mask the traceback.
        print(f"[memory-snapshot] failed to write {path}: {exc}")


@contextmanager
def record(dump_dir: Path | str, label: str, rank: int = 0):
    """Record allocator history for the block; dump on the way out.

    No-op when CL_MEMORY_SNAPSHOT is unset or CUDA is unavailable, so call
    sites need no conditionals of their own.
    """
    if not enabled() or not torch.cuda.is_available():
        yield
        return

    dump_dir = Path(dump_dir)
    torch.cuda.memory._record_memory_history(max_entries=MAX_ENTRIES)
    print(f"[memory-snapshot] recording (max_entries={MAX_ENTRIES:,}) for {label!r}")
    try:
        yield
    except torch.OutOfMemoryError:
        _dump(dump_dir, label, rank, "oom")
        raise
    except Exception:
        _dump(dump_dir, label, rank, "error")
        raise
    else:
        _dump(dump_dir, label, rank, "final")
    finally:
        # Stop recording before returning: the history is process-global, so
        # leaving it on would keep instrumenting whatever runs next (under
        # run_ddp_sweep that's the next config in the same process).
        torch.cuda.memory._record_memory_history(enabled=None)
