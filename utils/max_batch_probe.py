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

Two scopes:

  default        the LM's training step alone -- "what can this architecture
                 hold". Ignores the router entirely, so the answer is an upper
                 bound, not a setting to adopt.

  --with-router  additionally pays, each step, what train_router_experiments pays
                 before it trains: extract_router_features() over
                 pool_mult * batch candidate rows, then a router
                 forward/backward/step over the scores. Since the pool is
                 pool_mult times the training batch and (with
                 enable_text_hierarchical) runs the LM over every row of it,
                 this term usually dominates -- expect a much smaller ceiling
                 than the default scope reports.

Neither scope models the reward signal's extra loss_after forward pass, the
periodic evaluation, or the dataloader, so leave headroom either way.

    python utils/max_batch_probe.py --config configs/gpt2-ddp-multinode.yaml
    python utils/max_batch_probe.py --config configs/gpt2-ddp-multinode.yaml --with-router

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
from models.router import build_router_for_cfg, extract_router_features
from utils.general_utils import autocast_ctx, resolve_device

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


def _fake_external_embedding(cfg, rows: int):
    """Stand-in for TokenizedCorpus.embeddings when a config expects one.

    extract_router_features() raises unless external_embedding is supplied
    whenever use_external_embeddings or sentence_embedder_model is set. The
    real vectors come from a precomputed cache we have no reason to load here
    -- only their width reaches the router -- so a correctly-shaped random
    tensor measures the same memory.
    """
    if not (getattr(cfg, "use_external_embeddings", False)
            or getattr(cfg, "sentence_embedder_model", "")):
        return None
    dim = 0
    if getattr(cfg, "use_external_embeddings", False):
        dim += getattr(cfg, "external_embedding_dim", 768)
    if getattr(cfg, "sentence_embedder_model", ""):
        dim += getattr(cfg, "sentence_embedder_dim", 768)
    return torch.randn(rows, dim, device=cfg.device)


def try_batch(cfg, vocab_size: int, pad_token_id: int, batch: int, steps: int,
              with_router: bool) -> tuple[bool, int]:
    """Run `steps` training steps at `batch`. Returns (fitted, peak_bytes).

    With with_router, each step first pays what train_router_experiments pays
    before it ever trains: a feature pass over the whole candidate pool
    (cfg.pool_mult * batch rows, cfg.pool in a real run) and a router
    forward/backward over the resulting scores. That pool pass is the dominant
    memory term whenever enable_text_hierarchical is on -- it runs the LM over
    pool_mult times more rows than the training batch -- so the LM-only ceiling
    is a large overestimate for router configs.
    """
    model = opt = router = opt_router = None
    X = Y = logits = loss = X_pool = feats = scores = None
    try:
        _release()
        model = build_model(vocab_size, cfg).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
        loss_fn = nn.CrossEntropyLoss()

        if with_router:
            router = build_router_for_cfg(
                cfg, sequence_size=model.block, vocab_size=vocab_size
            )
            if router is not None:
                router = router.to(cfg.device)
                # Router state is not free either: AdamW's moments over an
                # EmbeddingRouter's tables are millions of params in their own
                # right, and they allocate on the first step() like the LM's.
                opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)

        pool_rows = batch * cfg.pool_mult
        for _ in range(steps):
            if router is not None:
                # Scoring the pool, exactly as the training loop does: the
                # hierarchical branch runs under no_grad internally, but the
                # router's own forward is grad-enabled so its update is real.
                X_pool = torch.randint(0, vocab_size, (pool_rows, cfg.block),
                                       device=cfg.device)
                feats = extract_router_features(
                    model=model, X=X_pool, cfg=cfg,
                    pad_token_id=pad_token_id, vocab_size=vocab_size,
                    external_embedding=_fake_external_embedding(cfg, pool_rows),
                )
                scores = router(feats)
                opt_router.zero_grad()
                # Stand-in for the policy-gradient objective. Its *value* is
                # meaningless; what matters is that a backward pass over the
                # router's graph really happens, since that is what holds the
                # score-path activations alive.
                scores.mean().backward()
                opt_router.step()

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
        del model, opt, router, opt_router, X, Y, logits, loss, X_pool, feats, scores
        _release()


def find_max_batch(cfg, vocab_size: int, pad_token_id: int, lo: int, hi: int,
                   steps: int, with_router: bool) -> tuple[int, int]:
    """Exponential ramp to bracket the ceiling, then binary search it."""
    best, best_peak = 0, 0
    batch = lo

    while batch <= hi:
        ok, peak = try_batch(cfg, vocab_size, pad_token_id, batch, steps, with_router)
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
        ok, peak = try_batch(cfg, vocab_size, pad_token_id, mid, steps, with_router)
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
    parser.add_argument("--with-router", action="store_true",
                        help="Also pay the router's pool feature pass (pool_mult x batch "
                             "rows through the LM) and a router update each step -- what "
                             "train_router_experiments actually costs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("max_batch_probe requires a CUDA device")
    if args.steps < 2:
        print("[warn] --steps < 2 skips AdamW's moment allocation; peak will be understated")

    cfg = load_config_from_yaml(args.config)
    overrides = {"use_wandb": False, "world_size": 1, "rank": 0}
    if args.block is not None:
        overrides["block"] = args.block
    cfg = resolve_device(replace(cfg, **overrides))

    tokenizer = get_tokenizer(cfg.tokenizer_name)
    total = torch.cuda.get_device_properties(0).total_memory

    print(f"device      : {torch.cuda.get_device_name(0)}  ({total / GB:.1f} GiB)")
    print(f"model       : {cfg.model_type}"
          + (f" ({cfg.hf_model_name})" if cfg.model_type == "hf_pretrained" else ""))
    print(f"d_model={cfg.d_model} n_layers={cfg.n_layers} block={cfg.block} "
          f"vocab={tokenizer.vocab_size}")
    if args.with_router:
        print(f"scope       : LM step + router pool pass "
              f"(pool_mult={cfg.pool_mult}, arch={cfg.router_architecture}, "
              f"features={cfg.router_feature_source}, "
              f"hierarchical={cfg.enable_text_hierarchical})")
    else:
        print("scope       : LM training step only (--with-router to include the pool pass)")
    print(f"searching {args.start}..{args.max_batch}, {args.steps} steps/trial\n")

    best, peak = find_max_batch(
        cfg, tokenizer.vocab_size, tokenizer.pad_token_id or 0,
        args.start, args.max_batch, args.steps, args.with_router,
    )

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
    if args.with_router:
        print(f"Scope: LM step + router pool pass over {best * cfg.pool_mult} candidates.")
        print(f"Still unmeasured: the reward signal's extra loss_after forward "
              f"(every\n      router_update_every={cfg.router_update_every} steps), "
              f"eval passes, and the dataloader. Leave headroom.")
    else:
        print(f"NOTE: LM training step only. A real run also forwards the router's "
              f"pool of\n      pool_mult={cfg.pool_mult} x batch candidates "
              f"({best * cfg.pool_mult} samples at this batch), which is not "
              f"measured\n      here. Re-run with --with-router for the number "
              f"you can actually use.")
    print(f"MAX_BATCH={best} PEAK_BYTES={peak}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
