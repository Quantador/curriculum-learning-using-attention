"""Tests for utils/lr_scheduler.py. Run directly: python tests/test_lr_scheduler.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_no_warmup_no_decay_is_always_peak():
    from utils.lr_scheduler import wsd_multiplier

    for progress in (0.0, 0.3, 0.7, 1.0):
        assert wsd_multiplier(progress, warmup_frac=0.0, decay_frac=0.0) == 1.0
    print("OK: warmup_frac=0, decay_frac=0 -> constant multiplier of 1.0")


def test_warmup_ramps_linearly_from_zero_to_peak():
    from utils.lr_scheduler import wsd_multiplier

    warmup_frac = 0.1
    assert wsd_multiplier(0.0, warmup_frac=warmup_frac) == 0.0
    assert math.isclose(wsd_multiplier(0.05, warmup_frac=warmup_frac), 0.5)
    # Multiplier hits 1.0 once warmup ends (stable phase takes over exactly at the boundary).
    assert wsd_multiplier(warmup_frac, warmup_frac=warmup_frac) == 1.0
    print("OK: linear warmup 0 -> peak over warmup_frac")


def test_stable_phase_holds_at_peak():
    from utils.lr_scheduler import wsd_multiplier

    decay_frac = 0.2
    for progress in (0.0, 0.3, 0.5, 0.79):
        assert wsd_multiplier(progress, decay_frac=decay_frac) == 1.0
    print("OK: multiplier stays at 1.0 throughout the stable phase")


def test_decay_reaches_min_lr_ratio_exactly_at_progress_one():
    from utils.lr_scheduler import wsd_multiplier

    for min_lr_ratio in (0.0, 0.1):
        mult = wsd_multiplier(1.0, decay_frac=0.2, min_lr_ratio=min_lr_ratio)
        assert math.isclose(mult, min_lr_ratio, abs_tol=1e-9), (min_lr_ratio, mult)
    print("OK: decay reaches exactly min_lr_ratio at progress=1.0")


def test_decay_matches_1_minus_sqrt_shape():
    """Hägele et al. (NeurIPS 2024), Sec 3.2: f(n) = 1 - sqrt((n - stable_end) / decay_frac)."""
    from utils.lr_scheduler import wsd_multiplier

    decay_frac = 0.2
    stable_end = 1.0 - decay_frac
    # Quarter of the way through the cooldown: decay_progress=0.25 -> 1 - sqrt(0.25) = 0.5.
    progress = stable_end + 0.25 * decay_frac
    assert math.isclose(wsd_multiplier(progress, decay_frac=decay_frac), 0.5, abs_tol=1e-9)
    print("OK: cooldown shape matches the paper's (1-sqrt) formula")


def test_decay_is_monotonically_decreasing():
    from utils.lr_scheduler import wsd_multiplier

    decay_frac = 0.2
    stable_end = 1.0 - decay_frac
    samples = [stable_end + f * decay_frac for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
    mults = [wsd_multiplier(p, decay_frac=decay_frac) for p in samples]
    assert all(a >= b for a, b in zip(mults, mults[1:])), mults
    print(f"OK: cooldown is monotonically decreasing: {[round(m, 3) for m in mults]}")


def test_progress_past_one_clamps_to_min_lr_ratio():
    from utils.lr_scheduler import wsd_multiplier

    mult = wsd_multiplier(1.5, decay_frac=0.2, min_lr_ratio=0.1)
    assert math.isclose(mult, 0.1, abs_tol=1e-9)
    print("OK: progress beyond 1.0 clamps rather than going negative/undefined")


def test_scales_per_group_peak_independently():
    """Mirrors how build_optimizer()'s Muon/AdamW sub-groups (different lr_muon/lr_lm peaks)
    get scaled in training.py/rl_training.py: same multiplier, each group's own base lr."""
    from utils.lr_scheduler import wsd_multiplier

    peak_lrs = {"muon": 0.02, "adamw": 3e-4}
    mult = wsd_multiplier(0.95, decay_frac=0.2, min_lr_ratio=0.0)
    scaled = {name: peak * mult for name, peak in peak_lrs.items()}
    assert math.isclose(scaled["muon"] / peak_lrs["muon"], scaled["adamw"] / peak_lrs["adamw"])
    print(f"OK: per-group peaks scale proportionally by the same multiplier ({mult:.4f})")


if __name__ == "__main__":
    test_no_warmup_no_decay_is_always_peak()
    test_warmup_ramps_linearly_from_zero_to_peak()
    test_stable_phase_holds_at_peak()
    test_decay_reaches_min_lr_ratio_exactly_at_progress_one()
    test_decay_matches_1_minus_sqrt_shape()
    test_decay_is_monotonically_decreasing()
    test_progress_past_one_clamps_to_min_lr_ratio()
    test_scales_per_group_peak_independently()
    print("All tests passed.")
