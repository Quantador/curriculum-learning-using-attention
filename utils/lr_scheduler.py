"""Warmup-Stable-Decay (WSD) learning rate multiplier.

Implements the "constant learning rate + cooldown" schedule from Hägele et al.,
"Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations"
(NeurIPS 2024) -- first proposed as "trapezoidal" by Zhai et al. (2022) and
later named WSD by Hu et al. (2024, MiniCPM). Three phases over training
progress n in [0, 1]:

  1. warmup: linear ramp 0 -> peak lr, over the first `warmup_frac` of steps.
  2. stable: held at peak lr for the bulk of training.
  3. decay:  peak lr -> min_lr_ratio * peak lr over the final `decay_frac` of
             steps, via the paper's (1-sqrt) cooldown shape (Sec 3.2/3.3):

                f(n) = 1 - sqrt((n - stable_end) / decay_frac)

             which they find beats both linear and untuned-cosine cooldowns of
             the same length. Their sweep (Fig. 5) found the benefit plateaus
             around decay_frac=0.2; even decay_frac=0.05 with this shape
             nearly matches a fully length-matched cosine schedule (Fig. 6),
             which matters when the cooldown itself is a real compute cost.

The paper's formula (Eq. 1) is expressed there in absolute step counts n, N,
N_decay; this is the same shape reparameterized as fractions of total steps,
since callers here already track progress = global_step / total_steps rather
than raw step counts.
"""
from __future__ import annotations

import math


def wsd_multiplier(
    progress: float,
    warmup_frac: float = 0.0,
    decay_frac: float = 0.2,
    min_lr_ratio: float = 0.0,
) -> float:
    """Fraction of peak lr at the given training `progress` (0.0 to 1.0).

    Multiply this by each optimizer param group's own peak lr (recorded once,
    before any scheduling is applied) rather than by one global peak -- that
    keeps sub-groups with different peaks (e.g. Muon's lr_muon vs AdamW's
    lr_lm in utils/muon_optimizer.build_optimizer()) scaled proportionally to
    their own configured peak instead of sharing one.
    """
    if warmup_frac > 0.0 and progress < warmup_frac:
        return progress / warmup_frac

    stable_end = 1.0 - decay_frac
    if decay_frac <= 0.0 or progress <= stable_end:
        return 1.0

    decay_progress = min(1.0, (progress - stable_end) / decay_frac)
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - math.sqrt(decay_progress))
