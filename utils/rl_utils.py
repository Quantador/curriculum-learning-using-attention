from __future__ import annotations 
from typing import Tuple

import torch 

from config import ExperimentConfig
from models.model import TinyGPT, AttentionRouter
from utils.entropy_utils import compute_entropy

def reinforce_update(
    router: AttentionRouter,
    opt_router: torch.optim.Optimizer,
    reward: torch.Tensor,
    baseline: torch.Tensor,
    sel_probs: torch.Tensor,
    all_probs: torch.Tensor,
    lambda_ent: float,
    entropy_type: str = "shannon",
    entropy_alpha: float = 2.0,
    entropy_q: float = 2.0,
    coverage_loss: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard REINFORCE (vanilla policy gradient) router update.

    Advantage = reward - baseline (reduces gradient variance).
    Policy loss = -mean(advantage * log_prob_of_selected_samples).
    Total loss = policy_loss + lambda_ent * entropy_term.

    See module docstring for the entropy sign convention.

    Returns (loss_router, reinforce_loss, entropy) where entropy = -H.
    """
    advantage = reward - baseline
    reinforce_loss = -(advantage * sel_probs.log()).mean()

    # Use configurable entropy formulation
    entropy = compute_entropy(all_probs, entropy_type, entropy_alpha, entropy_q)

    loss_router = reinforce_loss + lambda_ent * entropy

    # Add coverage regularization if provided
    if coverage_loss is not None:
        loss_router = loss_router + coverage_loss

    opt_router.zero_grad()
    loss_router.backward()
    opt_router.step()

    return loss_router, reinforce_loss, entropy


def grpo_update(
    router: AttentionRouter,
    opt_router: torch.optim.Optimizer,
    reward: torch.Tensor,
    sel_probs: torch.Tensor,
    all_probs: torch.Tensor,
    lambda_ent: float,
    group_size: int,
    entropy_type: str = "shannon",
    entropy_alpha: float = 2.0,
    entropy_q: float = 2.0,
    coverage_loss: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Group Relative Policy Optimization (GRPO) router update.

    Instead of a single global baseline, advantages are normalised within
    small groups of group_size samples:
        advantage_i = (r_i - group_mean) / group_std
    This provides lower-variance gradient estimates when rewards vary
    substantially across samples, without needing a learned value function.

    Returns (loss_router, grpo_loss, entropy) where entropy = -H.
    """
    B = len(reward)
    n_groups = max(1, B // group_size)

    # Split rewards into groups and compute group-relative advantages
    advantages = torch.zeros_like(reward)
    for i in range(n_groups):
        start = i * group_size
        end = min((i + 1) * group_size, B)
        group_reward = reward[start:end]
        group_baseline = group_reward.mean()
        group_std = group_reward.std().clamp(min=1e-8)
        advantages[start:end] = (group_reward - group_baseline) / group_std

    # Handle remainder
    if B % group_size != 0:
        remainder_start = n_groups * group_size
        group_reward = reward[remainder_start:]
        group_baseline = group_reward.mean()
        group_std = group_reward.std().clamp(min=1e-8)
        advantages[remainder_start:] = (group_reward - group_baseline) / group_std

    grpo_loss = -(advantages.detach() * sel_probs.log()).mean()

    # Use configurable entropy formulation
    entropy = compute_entropy(all_probs, entropy_type, entropy_alpha, entropy_q)

    loss_router = grpo_loss + lambda_ent * entropy

    # Add coverage regularization if provided
    if coverage_loss is not None:
        loss_router = loss_router + coverage_loss

    opt_router.zero_grad()
    loss_router.backward()
    opt_router.step()

    return loss_router, grpo_loss, entropy


def ppo_update(
    model: TinyGPT,
    router: AttentionRouter,
    opt_router: torch.optim.Optimizer,
    X_sel: torch.Tensor,
    Y_sel: torch.Tensor,
    old_log_probs: torch.Tensor,
    reward: torch.Tensor,
    baseline: torch.Tensor,
    feats: torch.Tensor,
    sel_idx: torch.Tensor,
    cfg: ExperimentConfig,
    lambda_ent: float,
    temperature: float,
    entropy_type: str = "shannon",
    entropy_alpha: float = 2.0,
    entropy_q: float = 2.0,
    coverage_loss: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Proximal Policy Optimization (PPO) router update.

    Runs cfg.ppo_epochs inner update steps with the clipped surrogate:
        L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    where ratio = new_log_prob / old_log_prob and ε = cfg.ppo_clip.
    Clipping prevents destructively large policy updates in a single step.

    `temperature` must be the same value the caller used to produce
    old_log_probs (i.e. the schedule-annealed current_temp, not the static
    cfg.temp) -- otherwise ratio = exp(new_log_probs - old_log_probs) mixes
    two different temperatures' softmax outputs, corrupting the ratio with a
    spurious offset unrelated to any actual policy drift, even on the very
    first inner epoch before router weights have moved at all.

    Advantages are normalised across the selected batch before clipping.
    Coverage loss is applied only on the first inner epoch to avoid
    double-counting the coverage penalty.

    Returns averaged (loss_router, policy_loss, entropy) over inner steps,
    where entropy = -H.
    """
    advantage = (reward - baseline).detach()
    # Normalize advantages
    adv_std = advantage.std().clamp(min=1e-8)
    advantage = (advantage - advantage.mean()) / adv_std

    total_loss = torch.tensor(0.0, device=cfg.device)
    total_policy_loss = torch.tensor(0.0, device=cfg.device)
    total_entropy = torch.tensor(0.0, device=cfg.device)

    for _ in range(cfg.ppo_epochs):
        # Recompute probabilities with current router, at the same
        # temperature old_log_probs was computed at (see docstring) --
        # NOT cfg.temp, which is the static/initial value and ignores
        # cfg.temp_schedule's annealing.
        scores = router(feats)
        probs = torch.softmax(scores / temperature, dim=0)
        new_log_probs = probs[sel_idx].clamp_min(1e-12).log()

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)

        policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        # Use configurable entropy formulation
        entropy = compute_entropy(probs, entropy_type, entropy_alpha, entropy_q)

        loss_router = policy_loss + lambda_ent * entropy

        # Add coverage regularization if provided (only on first PPO epoch)
        if coverage_loss is not None and _ == 0:
            loss_router = loss_router + coverage_loss

        opt_router.zero_grad()
        loss_router.backward()
        opt_router.step()

        total_loss += loss_router.detach()
        total_policy_loss += policy_loss.detach()
        total_entropy += entropy.detach()

    n = cfg.ppo_epochs
    return total_loss / n, total_policy_loss / n, total_entropy / n

