# max_batch_probe.py
"""
Find the largest per-GPU batch size a config's LM can forward/backward.

utils/gpu_memory_probe.py answers the forward question -- "how much memory
does THIS config need" -- by running the real training path once and printing
its peak. This is the inverse: hold the config fixed, vary global_batch_size,
and report the largest value that survives a forward/backward/step.

Each trial builds a fresh model and AdamW, then runs cfg-faithful steps on
random token ids: opt.zero_grad(); autocast forward; cross-entropy; backward;
step -- the same sequence as training.py's train_baseline inner loop. Two
steps by default, because AdamW allocates its two moment buffers lazily on
the first step(): a one-step trial understates peak memory by 2x the
parameter bytes, which for a 1.5B-param model is ~12 GB of pure error.

Search is exponential ramp (1, 2, 4, ... until OOM or --max-batch) followed
by binary search between the last success and first failure, so a batch
ceiling of N costs ~2*log2(N) trials rather than N.

SCOPE -- read this before trusting the number:
this measures the LM's own training step ONLY. It does not model the router's
feature pass over cfg.pool (= cfg.pool_mult * global_batch_size) candidates,
which extract_hierarchical_hidden's docstring calls the dominant memory cost
when enable_text_hierarchical=True. A real run of this repo at the batch size
printed here will need substantially more. The summary prints the pool
multiplier as a reminder; treat the result as an upper bound.

    python utils/max_batch_probe.py --config configs/gpt2-ddp-multinode.yaml

Prints a parseable final line:
    MAX_BATCH=<int> PEAK_BYTES=<int>
"""
from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config_from_yaml
from data import get_tokenizer
from models.model import build_model
from utils.general_utils import autocast_ctx

GB = 1024 ** 3


def _is_oom(exc: BaseException) -> bool:
    """True for CUDA OOM under either spelling.

    torch.OutOfMemoryError exists from 2.5; older versions raise a bare
    RuntimeError whose message is the only way to tell OOM from a real bug.
    Getting this wrong in either direction is bad -- miss an OOM and the
    probe dies instead of recording a ceiling; over-match and a genuine
    crash is silently reported as "batch too big".
    """
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _release() -> None:
    """Drop cached blocks between trials.

    Without this, the allocator holds freed-but-cached segments from the
    previous trial and the next one fails at a size that would actually fit
    -- the search then converges on a ceiling well below the real one.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_batch(cfg, vocab_size: int, batch: int, steps: int) -> tuple[bool, int]:
    """Run `steps` training steps at `batch`. Returns (fitted, peak_bytes)."""
    model = opt = X = Y = logits = loss = None
    try:
        _release()
        model = build_model(vocab_size, cfg).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(steps):
            # Fresh ids per step so nothing is accidentally cached; contents
            # are irrelevant to memory, only shape and dtype are.
            X = torch.randint(0, vocab_size, (batch, cfg.block), device=cfg.device)
            Y = torch.randint(0, vocab_size, (batch, cfg.block), device=cfg.device)
            opt.zero_grad()
            with autocast_ctx(cfg.device):
                logits = model(X)
                loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss.backward()
            opt.step()

        torch.cuda.synchronize()
        return True, torch.cuda.max_memory_allocated()
    except Exception as exc:
        if not _is_oom(exc):
            raise
        return False, 0
    finally:
        # Names must die before empty_cache() or their storages stay alive and
        # the release is a no-op. Locals are dropped explicitly rather than
        # left to scope exit because the except path above returns first.
        del model, opt, X, Y, logits, loss
        _release()


def find_max_batch(cfg, vocab_size: int, lo: int, hi: int, steps: int) -> tuple[int, int]:
    """Exponential ramp to bracket the ceiling, then binary search it."""
    best, best_peak = 0, 0
    batch = lo

    while batch <= hi:
        ok, peak = try_batch(cfg, vocab_size, batch, steps)
        print(
            f"  batch={batch:<6} {'fits' if ok else 'OOM ':4}"
            + (f"  peak={peak / GB:6.2f} GiB" if ok else "")
        )
        if not ok:
            break
        best, best_peak = batch, peak
        batch *= 2
    else:
        # Ramp exhausted --max-batch without a failure: hi is the answer as
        # far as this probe can tell, not a measured ceiling.
        return best, best_peak

    # Bracket is (best, batch): best fits, batch does not. Nothing to search
    # when even the starting size failed.
    if best == 0:
        return 0, 0

    low, high = best, batch
    while high - low > 1:
        mid = (low + high) // 2
        ok, peak = try_batch(cfg, vocab_size, mid, steps)
        print(
            f"  batch={mid:<6} {'fits' if ok else 'OOM ':4}"
            + (f"  peak={peak / GB:6.2f} GiB" if ok else "")
        )
        if ok:
            low, best_peak = mid, peak
        else:
            high = mid
    return low, best_peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config to size")
    parser.add_argument("--start", type=int, default=1, help="First batch size tried")
    parser.add_argument("--max-batch", type=int, default=4096, help="Ramp ceiling")
    parser.add_argument("--steps", type=int, default=2,
                        help="Steps per trial; >=2 to include AdamW moments")
    parser.add_argument("--block", type=int, default=None,
                        help="Override cfg.block (sequence length)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("max_batch_probe requires a CUDA device")
    if args.steps < 2:
        print("[warn] --steps < 2 skips AdamW's moment allocation; peak will be understated")

    cfg = load_config_from_yaml(args.config)
    overrides = {"use_wandb": False, "world_size": 1, "rank": 0}
    if args.block is not None:
        overrides["block"] = args.block
    cfg = replace(cfg, **overrides)

    tokenizer = get_tokenizer(cfg.tokenizer_name)
    total = torch.cuda.get_device_properties(0).total_memory

    print(f"device      : {torch.cuda.get_device_name(0)}  ({total / GB:.1f} GiB)")
    print(f"model       : {cfg.model_type}"
          + (f" ({cfg.hf_model_name})" if cfg.model_type == "hf_pretrained" else ""))
    print(f"d_model={cfg.d_model} n_layers={cfg.n_layers} block={cfg.block} "
          f"vocab={tokenizer.vocab_size}")
    print(f"searching {args.start}..{args.max_batch}, {args.steps} steps/trial\n")

    best, peak = find_max_batch(cfg, tokenizer.vocab_size, args.start, args.max_batch, args.steps)

    print()
    if best == 0:
        print(f"nothing fits: batch={args.start} already OOMs on this GPU.")
        print(f"MAX_BATCH=0 PEAK_BYTES=0")
        sys.exit(1)

    print(f"max batch that fits : {best}")
    print(f"peak at that batch  : {peak / GB:.2f} GiB of {total / GB:.1f} GiB "
          f"({100 * peak / total:.1f}%)")
    print(f"tokens per step     : {best * cfg.block:,}")
    print()
    print(f"NOTE: LM training step only. A real run also forwards the router's "
          f"pool of\n      pool_mult={cfg.pool_mult} x batch candidates "
          f"({best * cfg.pool_mult} samples at this batch), which is not "
          f"measured\n      here. Treat {best} as an upper bound, not a "
          f"setting to adopt directly.")
    print(f"MAX_BATCH={best} PEAK_BYTES={peak}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
