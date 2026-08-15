# run_status.py
"""
Durable, on-disk record of how each run *ended*.

Everything the scheduler says about an experiment's fate -- run_scheduler()'s
"[done]" / "[FAILED] ... (exit N)", run_ddp_sweep()'s "[FAILED]" -- is a
plain print() to the orchestrator's stdout, which under RunAI/Kubernetes is
the pod log. A deleted or preempted pod takes that log with it, so exactly
when a post-mortem matters most there is nothing left saying what happened:
a preempted run and a silently OOM-killed one both just stop mid-log.

The files written here live in the sweep's scratch dir (shared storage), so
they outlive the pod. Reading them back -- `python -m utils.run_status
results/_parallel_run/<stamp>`:

  end_reason "completed"     ran to the end.
  end_reason "exception"     crashed in Python; the traceback is in this file
                             and in logs/<name>.log.
  end_reason "signal:SIGTERM"  an external kill that gave us notice. On
                             Kubernetes this is preemption or a deleted pod:
                             SIGTERM first, SIGKILL after the grace period.
  end_reason "killed:SIGKILL"  the worker died with no warning at all, but the
                             orchestrator outlived it to say so -- almost
                             always the host OOM-killer taking one worker.
  state still "running"      nothing recorded an ending, so nothing got the
                             chance to: SIGKILL of the whole pod, node
                             failure, or hard reset. Cross-check against
                             TERMINATED.orchestrator.json -- if that exists,
                             the pod was preempted and this worker just died
                             with it.

That last distinction is the one that is otherwise unrecoverable, and it is
why the state file is written *before* the run starts rather than only at the
end: absence of an ending is itself the evidence.
"""
from __future__ import annotations

import functools
import json
import os
import signal
import sys
import traceback as _traceback

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Experiments currently in flight in THIS process, as {status file: base
# fields}. Only the signal handler reads it: a signal handler cannot be
# passed arguments, and it must be able to name the experiment that was
# running when the signal landed.
_ACTIVE: Dict[Path, Dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write `payload` to `path` atomically, and fsync before returning.

    A status file's whole job is to survive the process being killed
    milliseconds later -- during preemption we are racing a SIGKILL that
    lands ~30s after the SIGTERM. A half-written or still-buffered JSON file
    is worth no more than no file at all, so: write to a temp file, fsync it,
    then rename (atomic on POSIX) over the real path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def status_dir_for(scratch_dir: Path | str) -> Path:
    """The status/ subdir of a sweep's scratch dir -- one JSON per experiment,
    alongside its existing configs/ and logs/ siblings."""
    return Path(scratch_dir) / "status"


def status_path(status_dir: Path | str, name: str) -> Path:
    """`name` must already be filesystem-safe (both call sites pass
    general_utils.safe_name()'s output, matching logs/<name>.log)."""
    return Path(status_dir) / f"{name}.json"


@contextmanager
def track(status_dir: Path | str, name: str, **extra: Any):
    """Record one experiment's lifecycle to status_dir/<name>.json.

    Writes "running" on entry (so an ending that never arrives is
    detectable), then "finished" or "failed" on the way out. Signals are NOT
    handled here -- install_signal_handlers() covers those, because a
    SIGTERM handled the way we handle it never unwinds the stack and so never
    reaches this contextmanager's except clause.
    """
    path = status_path(status_dir, name)
    base = {
        "name": name,
        "pid": os.getpid(),
        "started_at": _now(),
        **extra,
    }
    _write_json(path, {**base, "state": "running", "end_reason": None})
    _ACTIVE[path] = base
    try:
        yield
    except BaseException as exc:
        # BaseException, not Exception: KeyboardInterrupt and SystemExit both
        # skip `except Exception`, and a run ending on either is still a run
        # whose ending we want on disk.
        _write_json(path, {
            **base,
            "state": "failed",
            "ended_at": _now(),
            "end_reason": "exception",
            "exception_type": type(exc).__name__,
            "exception": str(exc)[:2000],
            # Tail, not head: the innermost frames and the actual error
            # message are at the end of a traceback, and a deep torch stack
            # can otherwise push them past any sane cap.
            "traceback": _traceback.format_exc()[-8000:],
        })
        raise
    else:
        _write_json(path, {
            **base,
            "state": "finished",
            "ended_at": _now(),
            "end_reason": "completed",
        })
    finally:
        _ACTIVE.pop(path, None)


def record_child_exit(status_dir: Path | str, name: str, returncode: int) -> None:
    """Reconcile a worker subprocess's status file from the parent's side.

    Covers the case the child itself cannot: a SIGKILL (returncode -9, i.e.
    the host OOM-killer picking off one worker) leaves the child no chance to
    write anything, but the orchestrator survives and knows exactly what
    happened. Negative returncodes are -signum by Popen convention.

    A child that already recorded its own ending is left alone -- its record
    is strictly better than ours, since it has the traceback.
    """
    path = status_path(status_dir, name)
    try:
        current = json.loads(path.read_text())
    except (OSError, ValueError):
        current = {"name": name}
    if current.get("state") in {"finished", "failed", "terminated"}:
        return

    if returncode < 0:
        signame = signal.Signals(-returncode).name
        end_reason, state = f"killed:{signame}", "terminated"
    elif returncode == 0:
        end_reason, state = "completed", "finished"
    else:
        end_reason, state = "nonzero_exit", "failed"

    _write_json(path, {
        **current,
        "state": state,
        "ended_at": _now(),
        "end_reason": end_reason,
        "exit_code": returncode,
    })


def _handle_termination(signum: int, _frame: Any, *, marker_path: Path, role: str) -> None:
    signame = signal.Signals(signum).name
    stamp = _now()

    # Whatever was mid-flight ended *because* of this signal; say so in its
    # own status file too, so the per-experiment view and the sweep-level
    # marker agree and neither has to be read in light of the other.
    for path, base in list(_ACTIVE.items()):
        _write_json(path, {
            **base,
            "state": "terminated",
            "ended_at": stamp,
            "end_reason": f"signal:{signame}",
        })

    _write_json(marker_path, {
        "role": role,
        "signal": signame,
        "signum": signum,
        "pid": os.getpid(),
        "terminated_at": stamp,
        "active_experiments": sorted(p.stem for p in _ACTIVE),
    })

    # Restore the default disposition and re-raise at ourselves, rather than
    # sys.exit(): that gives the conventional 128+signum wait status (so the
    # parent's record_child_exit() sees a true -signum), and it skips every
    # `finally` on the stack. Skipping them is the point -- wandb.finish() in
    # run_single_experiment()'s finally can block for a long time, and
    # Kubernetes SIGKILLs the pod ~30s after the SIGTERM regardless. The
    # marker is already fsync'd; a clean-ish shutdown is not worth risking it.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def install_signal_handlers(
    scratch_dir: Path | str,
    role: str,
    signals: Iterable[signal.Signals] = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP),
) -> Path:
    """Make an external kill leave a trace. Returns the marker path.

    SIGTERM is the signal that matters: Kubernetes (and so RunAI) sends it
    before evicting or deleting a pod, waits out terminationGracePeriodSeconds,
    then SIGKILLs. Catching it is the only way to tell "preempted" apart from
    "died for some other silent reason" after the fact -- the pod log that
    would otherwise have said so is deleted along with the pod.

    `role` distinguishes the orchestrator's marker from a worker's, since
    both write into the same scratch dir.
    """
    marker_path = Path(scratch_dir) / f"TERMINATED.{role}.json"
    for sig in signals:
        try:
            signal.signal(sig, functools.partial(_handle_termination, marker_path=marker_path, role=role))
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or the platform has no such signal. This
            # is diagnostics: never a reason to fail a sweep that would
            # otherwise run.
            pass
    return marker_path


def summarize(scratch_dir: Path | str) -> List[Dict[str, Any]]:
    """Every experiment's recorded fate, ordered by name. Entries whose state
    is still "running" are the interesting ones -- see the module docstring."""
    out: List[Dict[str, Any]] = []
    for path in sorted(status_dir_for(scratch_dir).glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            out.append({"name": path.stem, "state": "unreadable", "end_reason": None})
    return out


def _main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m utils.run_status <scratch_dir>", file=sys.stderr)
        return 2
    scratch_dir = Path(argv[0])
    if not scratch_dir.is_dir():
        print(f"no such directory: {scratch_dir}", file=sys.stderr)
        return 2

    markers = sorted(scratch_dir.glob("TERMINATED.*.json"))
    for marker in markers:
        m = json.loads(marker.read_text())
        print(f"!! {m['role']} received {m['signal']} at {m['terminated_at']} "
              f"-- external kill (preemption / deleted pod), not a crash")

    rows = summarize(scratch_dir)
    if not rows:
        print(f"no status files under {status_dir_for(scratch_dir)} "
              f"(sweep predates run_status, or died before writing any)")
        return 1

    width = max(len(r.get("name", "?")) for r in rows)
    for r in rows:
        reason = r.get("end_reason") or "(none recorded)"
        note = ""
        if r.get("state") == "running":
            note = ("  <- no ending recorded: SIGKILL, host OOM-killer, or node failure"
                    if not markers else "  <- killed with the pod")
        print(f"  {r.get('name', '?'):<{width}}  {r.get('state', '?'):<11} {reason}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
