"""
Smoke tests for all new experiment features.
Runs 1 epoch on tiny dataset subsets to verify every code path works.

Usage (from project root):
    python clean/test_run.py
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import replace

import torch

from config import ExperimentConfig
from data import get_tokenizer, make_mixed_chunks, make_single_chunks, MixedLMDataset
from model import TinyGPT
from modelExperiments import build_router, get_router_feature_dim
from training import train_baseline, train_router, compare_runs
from trainingExperiments import train_aux_baseline
from metrics import MetricsTracker, DiversityTracker


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 0) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def tiny_cfg(**overrides) -> ExperimentConfig:
    """Tiny config for fast smoke tests: small model, 1 epoch, no WandB."""
    base = ExperimentConfig()
    return replace(
        base,
        # Tiny model (d_model divisible by n_heads=4; block divisible by n_chunks=4)
        d_model=64, n_layers=2, n_heads=4, d_ff=128,
        n_chunks=4, block=64,
        # Tiny data
        easy_samples=300, hard_samples=150, max_chunks=5_000,
        # Tiny training
        batch=4, pool_mult=3, epochs=1, log_every=10,
        use_wandb=False,
        save_dir="results/smoke_test",
        **overrides,
    )


# Shared mixed dataset loaded once across tests that don't need a custom one.
_shared: dict = {}

def shared_data():
    if not _shared:
        cfg = tiny_cfg()
        tok = get_tokenizer()
        train_chunks = make_mixed_chunks("train",      cfg, tok)
        val_chunks   = make_mixed_chunks("validation", cfg, tok)
        _shared["tok"]      = tok
        _shared["train_ds"] = MixedLMDataset(train_chunks)
        _shared["val_ds"]   = MixedLMDataset(val_chunks)
        _shared["d_input"]  = get_router_feature_dim(cfg)
        print(f"  [shared] {len(_shared['train_ds'])} train  |  "
              f"{len(_shared['val_ds'])} val chunks")
    return _shared["tok"], _shared["train_ds"], _shared["val_ds"], _shared["d_input"]


results: dict[str, str] = {}

def run_test(label: str, fn) -> None:
    sep = "-" * 55
    print(f"\n{sep}\n  {label}\n{sep}")
    try:
        fn()
        results[label] = "PASS"
        print("  -> PASS")
    except Exception as exc:
        results[label] = f"FAIL: {exc}"
        print(f"  -> FAIL: {exc}")
        traceback.print_exc()


# ── Test 1: Default mixed dataset (TinyStories + OpenWebText2) ────────────────

def test_1_default_datasets():
    cfg = tiny_cfg()
    tok = get_tokenizer()
    set_seed()
    train_chunks = make_mixed_chunks("train",      cfg, tok)
    val_chunks   = make_mixed_chunks("validation", cfg, tok)
    train_ds = MixedLMDataset(train_chunks)
    val_ds   = MixedLMDataset(val_chunks)
    assert len(train_ds) > 0, "empty train set"
    assert len(val_ds)   > 0, "empty val set"
    assert any(d == 0 for d in train_ds.difficulty), "expected easy labels (0)"
    assert any(d == 1 for d in train_ds.difficulty), "expected hard labels (1)"
    print(f"  {len(train_ds)} train | {len(val_ds)} val chunks")


# ── Test 2: Custom datasets (WikiText easy + ML-ArXiv hard) ───────────────────

def test_2_custom_datasets():
    cfg = tiny_cfg(
        easy_dataset="Salesforce/wikitext",
        hard_dataset="CShorten/ML-ArXiv-Papers",
    )
    tok = get_tokenizer()
    set_seed()
    train_chunks = make_mixed_chunks("train",      cfg, tok)
    val_chunks   = make_mixed_chunks("validation", cfg, tok)
    train_ds = MixedLMDataset(train_chunks)
    val_ds   = MixedLMDataset(val_chunks)
    assert len(train_ds) > 0, "empty train set"
    print(f"  {len(train_ds)} train | {len(val_ds)} val chunks")


# ── Test 3: Multi-head router (n_heads = 2 and 4) ────────────────────────────

def test_3_multihead_router():
    tok, train_ds, val_ds, d_input = shared_data()

    for n_heads in [2, 4]:
        cfg    = tiny_cfg(router_n_heads=n_heads)
        model  = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
        router = build_router(d_input=d_input, arch="attention", n_heads=n_heads)

        n_params_multi = sum(p.numel() for p in router.parameters())
        n_params_single = sum(
            p.numel() for p in
            build_router(d_input=d_input, arch="attention", n_heads=1).parameters()
        )
        assert n_params_multi > n_params_single, (
            f"n_heads={n_heads} should have more params than n_heads=1"
        )

        set_seed()
        m   = MetricsTracker(f"mh{n_heads}", use_wandb=False)
        div = DiversityTracker(len(train_ds))
        train_router(cfg, model, router, train_ds, val_ds, tok, m, div)
        ppl = m.get_final_ppl()
        assert ppl is not None and ppl > 1.0, f"bad ppl for n_heads={n_heads}"
        print(f"  n_heads={n_heads}: {n_params_multi} router params  |  val_ppl={ppl:.1f}")


# ── Test 4: Single-dataset mode (no easy/hard split) ─────────────────────────

def test_4_single_dataset():
    # Use OpenWebText2 as a stand-in for FineWeb (same code path, known to work)
    cfg = tiny_cfg(
        use_single_dataset=True,
        single_dataset="Geralt-Targaryen/openwebtext2",
        single_dataset_samples=300,
    )
    tok = get_tokenizer()
    set_seed()
    train_chunks, val_chunks, train_embs, val_embs = make_single_chunks(cfg, tok)

    assert train_embs is None, "no external embeddings expected"
    assert val_embs   is None, "no external embeddings expected"

    train_ds = MixedLMDataset(train_chunks)
    val_ds   = MixedLMDataset(val_chunks)
    assert len(train_ds) > 0, "empty train set"
    assert all(d == 0  for d in train_ds.difficulty), "all train labels should be 0"
    assert all(d == -1 for d in val_ds.difficulty),   "all val labels should be -1"
    print(f"  {len(train_ds)} train | {len(val_ds)} val chunks  (all difficulty=0)")

    d_input = get_router_feature_dim(cfg)
    model  = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
    router = build_router(d_input=d_input, arch="attention")
    m      = MetricsTracker("single_ds", use_wandb=False)
    div    = DiversityTracker(len(train_ds))
    train_router(cfg, model, router, train_ds, val_ds, tok, m, div)
    ppl = m.get_final_ppl()
    assert ppl is not None and ppl > 1.0
    print(f"  val_ppl={ppl:.1f}")


# ── Test 5: Auxiliary network baseline ───────────────────────────────────────

def test_5_aux_baseline():
    tok, train_ds, val_ds, d_input = shared_data()
    cfg = tiny_cfg(aux_net_hidden=64)
    set_seed()

    model   = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
    aux_net = build_router(d_input=d_input, arch="auxnet", d_hidden=64)
    m   = MetricsTracker("aux", use_wandb=False)
    div = DiversityTracker(len(train_ds))
    train_aux_baseline(cfg, model, aux_net, train_ds, val_ds, tok, m, div)

    ppl = m.get_final_ppl()
    assert ppl is not None and ppl > 1.0, f"bad ppl: {ppl}"
    assert "loss_aux" in m.history,       "loss_aux should be logged"
    assert "avg_improvement" in m.history, "avg_improvement should be logged"
    print(f"  val_ppl={ppl:.1f}  |  "
          f"mean loss_aux={sum(m.history['loss_aux'])/len(m.history['loss_aux']):.6f}")


# ── Test 6: Full three-way comparison (baseline + router + aux) ───────────────

def test_6_full_comparison():
    tok, train_ds, val_ds, d_input = shared_data()
    cfg = tiny_cfg(aux_net_hidden=64)

    # Baseline
    set_seed()
    model_b = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
    base_m  = MetricsTracker("b", use_wandb=False)
    base_d  = DiversityTracker(len(train_ds))
    train_baseline(cfg, model_b, train_ds, val_ds, base_m, base_d)

    # Router
    set_seed()
    model_r = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
    router  = build_router(d_input=d_input, arch="attention")
    rtr_m   = MetricsTracker("r", use_wandb=False)
    rtr_d   = DiversityTracker(len(train_ds))
    train_router(cfg, model_r, router, train_ds, val_ds, tok, rtr_m, rtr_d)

    # Aux-net baseline
    set_seed()
    model_a = TinyGPT(vocab_size=tok.vocab_size, cfg=cfg)
    aux_net = build_router(d_input=d_input, arch="auxnet", d_hidden=64)
    aux_m   = MetricsTracker("a", use_wandb=False)
    aux_d   = DiversityTracker(len(train_ds))
    train_aux_baseline(cfg, model_a, aux_net, train_ds, val_ds, tok, aux_m, aux_d)

    compare_runs(base_m, rtr_m, aux_m)

    assert base_m.get_final_ppl() is not None
    assert rtr_m.get_final_ppl()  is not None
    assert aux_m.get_final_ppl()  is not None


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_test("1. Default mixed datasets (TinyStories + OpenWebText2)", test_1_default_datasets)
    run_test("2. Custom datasets       (WikiText easy + ML-ArXiv hard)", test_2_custom_datasets)
    run_test("3. Multi-head router     (n_heads = 2 and 4)",             test_3_multihead_router)
    run_test("4. Single-dataset mode   (OpenWebText2, no easy/hard)",    test_4_single_dataset)
    run_test("5. Auxiliary net baseline",                                test_5_aux_baseline)
    run_test("6. Full comparison       (baseline + router + aux-net)",   test_6_full_comparison)

    print(f"\n{'='*55}\n  RESULTS\n{'='*55}")
    all_pass = True
    for label, result in results.items():
        icon = "PASS" if result == "PASS" else "FAIL"
        print(f"  [{icon}]  {label}")
        if result != "PASS":
            all_pass = False
            print(f"       {result}")

    print()
    if all_pass:
        print(f"All {len(results)} tests passed.")
    else:
        n_fail = sum(1 for r in results.values() if r != "PASS")
        print(f"{n_fail}/{len(results)} tests FAILED.")
        sys.exit(1)
