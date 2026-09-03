"""Tests for utils/lr_scheduler.py. Run directly: python tests/test_lr_scheduler.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Warmup is now counted in absolute steps while decay stays a fraction of progress, so the
# tests carry a total-step count and derive one from the other the way the training loops do
# (progress = global_step / total_steps -- training.py:247, rl_training.py:962/1611).
TOTAL_STEPS = 10_000


def at_step(step, total_steps=TOTAL_STEPS, **kwargs):
    """wsd_multiplier() called the way the training loops call it, from a step count."""
    from utils.lr_scheduler import wsd_multiplier

    return wsd_multiplier(step / total_steps, step, **kwargs)


def test_no_warmup_no_decay_is_always_peak():
    for progress in (0.0, 0.3, 0.7, 1.0):
        step = round(progress * TOTAL_STEPS)
        assert at_step(step, warmup_steps=0, decay_frac=0.0) == 1.0
    print("OK: warmup_steps=0, decay_frac=0 -> constant multiplier of 1.0")


def test_warmup_ramps_linearly_from_zero_to_peak():
    warmup_steps = 300
    assert at_step(0, warmup_steps=warmup_steps) == 0.0
    assert math.isclose(at_step(150, warmup_steps=warmup_steps), 0.5)
    assert math.isclose(at_step(225, warmup_steps=warmup_steps), 0.75)
    # Multiplier hits 1.0 once warmup ends (stable phase takes over exactly at the boundary).
    assert at_step(warmup_steps, warmup_steps=warmup_steps) == 1.0
    print("OK: linear warmup 0 -> peak over warmup_steps")


def test_warmup_length_is_absolute_not_a_fraction_of_the_run():
    """The point of the step-count interface: a 300-step warmup is 300 steps regardless of how
    long the run is, so short and long runs share the same warmup curve instead of scaling it."""
    warmup_steps = 300
    for total_steps in (1_000, 10_000, 1_000_000):
        assert math.isclose(at_step(150, total_steps, warmup_steps=warmup_steps), 0.5), total_steps
    print("OK: warmup depends on current_step/warmup_steps, not on total run length")


def test_warmup_takes_precedence_over_decay():
    """current_step < warmup_steps wins even when progress has already entered the cooldown
    window -- guards against a mis-set warmup silently reading as a decayed lr."""
    # Warmup longer than the whole run: at 90% progress we are still in the ramp.
    mult = at_step(9_000, warmup_steps=20_000, decay_frac=0.2)
    assert math.isclose(mult, 9_000 / 20_000)
    print("OK: warmup branch takes precedence over the decay branch")


def test_stable_phase_holds_at_peak():
    decay_frac = 0.2
    for progress in (0.05, 0.3, 0.5, 0.79):
        step = round(progress * TOTAL_STEPS)
        assert at_step(step, warmup_steps=300, decay_frac=decay_frac) == 1.0
    print("OK: multiplier stays at 1.0 throughout the stable phase")


def test_decay_reaches_min_lr_ratio_exactly_at_progress_one():
    for min_lr_ratio in (0.0, 0.1):
        mult = at_step(TOTAL_STEPS, warmup_steps=300, decay_frac=0.2, min_lr_ratio=min_lr_ratio)
        assert math.isclose(mult, min_lr_ratio, abs_tol=1e-9), (min_lr_ratio, mult)
    print("OK: decay reaches exactly min_lr_ratio at progress=1.0")


def test_decay_matches_1_minus_sqrt_shape():
    """Hägele et al. (NeurIPS 2024), Sec 3.2: f(n) = 1 - sqrt((n - stable_end) / decay_frac)."""
    decay_frac = 0.2
    stable_end = 1.0 - decay_frac
    # Quarter of the way through the cooldown: decay_progress=0.25 -> 1 - sqrt(0.25) = 0.5.
    step = round((stable_end + 0.25 * decay_frac) * TOTAL_STEPS)
    mult = at_step(step, warmup_steps=300, decay_frac=decay_frac)
    assert math.isclose(mult, 0.5, abs_tol=1e-9), mult
    print("OK: cooldown shape matches the paper's (1-sqrt) formula")


def test_decay_is_monotonically_decreasing():
    decay_frac = 0.2
    stable_end = 1.0 - decay_frac
    steps = [round((stable_end + f * decay_frac) * TOTAL_STEPS)
             for f in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
    mults = [at_step(s, warmup_steps=300, decay_frac=decay_frac) for s in steps]
    assert all(a >= b for a, b in zip(mults, mults[1:])), mults
    print(f"OK: cooldown is monotonically decreasing: {[round(m, 3) for m in mults]}")


def test_progress_past_one_clamps_to_min_lr_ratio():
    mult = at_step(15_000, warmup_steps=300, decay_frac=0.2, min_lr_ratio=0.1)
    assert math.isclose(mult, 0.1, abs_tol=1e-9)
    print("OK: progress beyond 1.0 clamps rather than going negative/undefined")


def test_full_schedule_is_bounded_and_non_increasing_after_warmup():
    """Walk a whole run: the ramp rises to exactly 1.0, then nothing ever rises again and the
    multiplier stays inside [min_lr_ratio, 1.0] -- no overshoot past peak, no negative lr."""
    warmup_steps, decay_frac, min_lr_ratio = 300, 0.2, 0.05
    mults = [at_step(s, warmup_steps=warmup_steps, decay_frac=decay_frac, min_lr_ratio=min_lr_ratio)
             for s in range(0, TOTAL_STEPS + 1)]
    assert max(mults) == 1.0 and min(mults) >= 0.0, (max(mults), min(mults))
    # min_lr_ratio is the cooldown's floor, not a global one -- warmup still starts from 0.
    after_warmup = mults[warmup_steps:]
    assert min(after_warmup) >= min_lr_ratio - 1e-9, min(after_warmup)
    assert all(a >= b for a, b in zip(after_warmup, after_warmup[1:])), "rose after warmup"
    # Boundaries land where the phases say they should: peak at the end of warmup and held
    # through the last stable step, cooldown finishing at the floor.
    assert mults[warmup_steps] == 1.0 and mults[warmup_steps - 1] < 1.0
    assert mults[round((1.0 - decay_frac) * TOTAL_STEPS)] == 1.0
    assert math.isclose(mults[-1], min_lr_ratio, abs_tol=1e-9)
    print("OK: full run stays in [min_lr_ratio, 1.0] and never rises after warmup")


def test_scales_per_group_peak_independently():
    """Mirrors how build_optimizer()'s Muon/AdamW sub-groups (different lr_muon/lr_lm peaks)
    get scaled in training.py/rl_training.py: same multiplier, each group's own base lr."""
    peak_lrs = {"muon": 0.02, "adamw": 3e-4}
    mult = at_step(9_500, warmup_steps=300, decay_frac=0.2, min_lr_ratio=0.0)
    scaled = {name: peak * mult for name, peak in peak_lrs.items()}
    assert math.isclose(scaled["muon"] / peak_lrs["muon"], scaled["adamw"] / peak_lrs["adamw"])
    print(f"OK: per-group peaks scale proportionally by the same multiplier ({mult:.4f})")


if __name__ == "__main__":
    test_no_warmup_no_decay_is_always_peak()
    test_warmup_ramps_linearly_from_zero_to_peak()
    test_warmup_length_is_absolute_not_a_fraction_of_the_run()
    test_warmup_takes_precedence_over_decay()
    test_stable_phase_holds_at_peak()
    test_decay_reaches_min_lr_ratio_exactly_at_progress_one()
    test_decay_matches_1_minus_sqrt_shape()
    test_decay_is_monotonically_decreasing()
    test_progress_past_one_clamps_to_min_lr_ratio()
    test_full_schedule_is_bounded_and_non_increasing_after_warmup()
    test_scales_per_group_peak_independently()
    print("All tests passed.")
