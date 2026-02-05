from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from config import ExperimentConfig
from data import make_index_loader, MixedLMDataset
from model import TinyGPT, AttentionRouter
from metrics import MetricsTracker, DiversityTracker
from training import evaluate  # keep using your existing evaluate()
from modelExperiments import extract_router_features


@torch.no_grad()
def compute_loss_per_sample_vectorized(
    logits: torch.Tensor,  # [B, L, V]
    targets: torch.Tensor,  # [B, L]
) -> torch.Tensor:
    """
    Vectorized per-sample CE loss from precomputed logits.

    Returns:
        loss_per_sample: [B] where each element is the mean CE over sequence positions.
    """
    B, L, V = logits.shape
    # token-level CE: [B*L] -> reshape to [B, L]
    token_ce = F.cross_entropy(
        logits.view(B * L, V),
        targets.view(B * L),
        reduction="none",
    ).view(B, L)
    return token_ce.mean(dim=1)  # [B]


# =============================================================================
# Schedule utilities
# =============================================================================

def get_scheduled_value(
    schedule: str,
    initial: float,
    minimum: float,
    progress: float,  # 0.0 to 1.0
    step: int = 0,
    cycle_length: int = 1000,
) -> float:
    """Get scheduled value based on training progress."""
    if schedule == "fixed":
        return initial

    elif schedule == "linear_decay":
        return initial + (minimum - initial) * progress

    elif schedule == "cosine_decay":
        return minimum + (initial - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

    elif schedule == "exponential_decay":
        # Exponential decay: initial * exp(-k * progress), where k is chosen
        # such that at progress=1, we get minimum
        if initial <= 0 or minimum <= 0:
            return minimum
        k = -math.log(minimum / initial)
        return initial * math.exp(-k * progress)

    elif schedule == "cyclic":
        # Warm restarts: cosine annealing with periodic resets
        cycle_progress = (step % cycle_length) / cycle_length
        return minimum + (initial - minimum) * 0.5 * (1 + math.cos(math.pi * cycle_progress))

    elif schedule == "adaptive":
        # Placeholder for adaptive scheduling (implemented separately)
        return initial

    else:
        return initial


# =============================================================================
# Entropy formulations
# =============================================================================

def compute_shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Standard Shannon entropy: H = -sum(p * log(p))

    Args:
        probs: Probability distribution [M]

    Returns:
        Scalar entropy value
    """
    return (probs * probs.clamp_min(1e-12).log()).sum()


def compute_renyi_entropy(probs: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Rényi entropy: H_α = (1/(1-α)) * log(sum(p^α))

    Special cases:
    - α → 1: Shannon entropy
    - α = 0: Max entropy (log of support size)
    - α = 2: Collision entropy (related to collision probability)
    - α → ∞: Min-entropy

    Args:
        probs: Probability distribution [M]
        alpha: Rényi parameter (> 0, != 1)

    Returns:
        Scalar Rényi entropy value (negated for use as loss)
    """
    if abs(alpha - 1.0) < 1e-6:
        # Limit case: Shannon entropy
        return compute_shannon_entropy(probs)

    # Rényi entropy: (1/(1-α)) * log(sum(p^α))
    p_alpha = probs.clamp_min(1e-12).pow(alpha)
    renyi = (1.0 / (1.0 - alpha)) * p_alpha.sum().clamp_min(1e-12).log()

    # Return negative for consistency (minimizing = maximizing entropy)
    return -renyi


def compute_tsallis_entropy(probs: torch.Tensor, q: float = 2.0) -> torch.Tensor:
    """
    Tsallis entropy: S_q = (1/(q-1)) * (1 - sum(p^q))

    Non-extensive entropy that generalizes Boltzmann-Gibbs.
    - q → 1: Shannon entropy
    - q < 1: Favors rare events
    - q > 1: Favors common events

    Args:
        probs: Probability distribution [M]
        q: Tsallis parameter (> 0)

    Returns:
        Scalar Tsallis entropy value (negated for use as loss)
    """
    if abs(q - 1.0) < 1e-6:
        # Limit case: Shannon entropy
        return compute_shannon_entropy(probs)

    p_q = probs.clamp_min(1e-12).pow(q)
    tsallis = (1.0 / (q - 1.0)) * (1.0 - p_q.sum())

    # Return negative for consistency
    return -tsallis


def compute_kl_from_uniform(probs: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from uniform distribution: KL(p || u)

    Measures how far the distribution is from uniform (maximum entropy).
    KL(p || u) = sum(p * log(p)) - log(1/n) = -H(p) + log(n)

    Args:
        probs: Probability distribution [M]

    Returns:
        KL divergence (0 = uniform, higher = more peaked)
    """
    n = probs.shape[0]
    log_n = math.log(n)
    shannon = -compute_shannon_entropy(probs)  # H(p)
    return log_n - shannon  # KL(p || u)


def compute_entropy(
    probs: torch.Tensor,
    entropy_type: str = "shannon",
    alpha: float = 2.0,
    q: float = 2.0,
) -> torch.Tensor:
    """
    Compute entropy using specified formulation.

    Args:
        probs: Probability distribution [M]
        entropy_type: 'shannon', 'renyi', 'tsallis', or 'kl_uniform'
        alpha: Parameter for Rényi entropy
        q: Parameter for Tsallis entropy

    Returns:
        Entropy value (to be used in loss with positive lambda_ent)
    """
    if entropy_type == "shannon":
        return compute_shannon_entropy(probs)

    elif entropy_type == "renyi":
        return compute_renyi_entropy(probs, alpha)

    elif entropy_type == "tsallis":
        return compute_tsallis_entropy(probs, q)

    elif entropy_type == "kl_uniform":
        # Return negative KL so that minimizing increases uniformity
        return -compute_kl_from_uniform(probs)

    else:
        return compute_shannon_entropy(probs)


# =============================================================================
# Entropy targeting (SAC-style automatic temperature adjustment)
# =============================================================================

class EntropyTargeting:
    """
    Automatic entropy coefficient adjustment (SAC-style).

    Maintains a target entropy level by adjusting lambda_ent.
    If current entropy < target, increase lambda_ent (encourage exploration).
    If current entropy > target, decrease lambda_ent (allow exploitation).
    """

    def __init__(
        self,
        target_entropy: float,
        initial_lambda: float = 0.005,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.target_entropy = target_entropy
        # Use log for numerical stability
        self.log_lambda = torch.tensor(
            math.log(initial_lambda), requires_grad=True, device=device
        )
        self.optimizer = torch.optim.Adam([self.log_lambda], lr=lr)

    @property
    def lambda_ent(self) -> float:
        return self.log_lambda.exp().item()

    def update(self, current_entropy: torch.Tensor) -> float:
        """
        Update entropy coefficient based on current entropy.

        Args:
            current_entropy: Current entropy of the router distribution

        Returns:
            Updated lambda_ent value
        """
        # Loss: lambda * (entropy - target)
        # If entropy < target, loss is negative, so gradient increases lambda
        # If entropy > target, loss is positive, so gradient decreases lambda
        entropy_loss = self.log_lambda.exp() * (
            current_entropy.detach() - self.target_entropy
        )

        self.optimizer.zero_grad()
        entropy_loss.backward()
        self.optimizer.step()

        return self.lambda_ent


def compute_max_entropy(n: int) -> float:
    """Maximum entropy for n outcomes (uniform distribution)."""
    return math.log(n)


# =============================================================================
# Coverage-based regularization
# =============================================================================

class CoverageTracker:
    """
    Tracks sample selection history for coverage-based regularization.

    Coverage regularization encourages the router to explore diverse samples
    by penalizing repeated selection of the same samples.
    """

    def __init__(
        self,
        dataset_size: int,
        coverage_type: str = "count",
        decay: float = 0.99,
        device: str = "cpu",
    ):
        """
        Args:
            dataset_size: Total number of samples in the dataset
            coverage_type: 'count', 'recency', or 'uncertainty'
            decay: Decay factor for recency-based tracking
            device: Device to store tensors on
        """
        self.dataset_size = dataset_size
        self.coverage_type = coverage_type
        self.decay = decay
        self.device = device

        # Selection counts per sample
        self.counts = torch.zeros(dataset_size, device=device)

        # Recency scores (higher = more recently selected)
        self.recency = torch.zeros(dataset_size, device=device)

        # Cumulative loss per sample (for uncertainty-based coverage)
        self.cumulative_loss = torch.zeros(dataset_size, device=device)
        self.loss_counts = torch.zeros(dataset_size, device=device)

        self.total_selections = 0

    def update(
        self,
        selected_indices: list[int],
        losses: torch.Tensor | None = None,
    ):
        """
        Update coverage statistics after selection.

        Args:
            selected_indices: List of selected sample indices
            losses: Per-sample losses for uncertainty tracking
        """
        # Decay recency for all samples
        self.recency *= self.decay

        # Update for selected samples
        for i, idx in enumerate(selected_indices):
            self.counts[idx] += 1
            self.recency[idx] = 1.0  # Mark as recently selected

            if losses is not None and i < len(losses):
                self.cumulative_loss[idx] += losses[i].item()
                self.loss_counts[idx] += 1

        self.total_selections += len(selected_indices)

    def get_coverage_bonus(
        self,
        pool_indices: list[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute coverage bonus for samples in the pool.

        Higher bonus for less-covered samples encourages exploration.

        Args:
            pool_indices: Indices of samples in the current pool
            temperature: Temperature for softening the bonus

        Returns:
            Coverage bonus tensor [pool_size]
        """
        pool_indices_t = torch.tensor(pool_indices, device=self.device)

        if self.coverage_type == "count":
            # Inverse of selection count (less selected = higher bonus)
            counts = self.counts[pool_indices_t]
            # Add 1 to avoid division by zero, normalize
            bonus = 1.0 / (counts + 1.0)
            # Normalize to [0, 1]
            bonus = bonus / bonus.max().clamp_min(1e-8)

        elif self.coverage_type == "recency":
            # Inverse of recency (less recently selected = higher bonus)
            recency = self.recency[pool_indices_t]
            bonus = 1.0 - recency  # 1 if never selected recently, 0 if just selected

        elif self.coverage_type == "uncertainty":
            # Higher average loss = more uncertain = higher bonus
            cum_loss = self.cumulative_loss[pool_indices_t]
            loss_counts = self.loss_counts[pool_indices_t].clamp_min(1)
            avg_loss = cum_loss / loss_counts

            # Normalize
            if avg_loss.max() > 0:
                bonus = avg_loss / avg_loss.max()
            else:
                bonus = torch.ones_like(avg_loss)

        else:
            bonus = torch.ones(len(pool_indices), device=self.device)

        # Apply temperature
        if temperature != 1.0:
            bonus = bonus.pow(1.0 / temperature)

        return bonus

    def get_coverage_stats(self) -> dict:
        """Get coverage statistics for logging."""
        selected_mask = self.counts > 0
        return {
            "coverage_ratio": selected_mask.float().mean().item(),
            "avg_selection_count": self.counts[selected_mask].mean().item() if selected_mask.any() else 0,
            "max_selection_count": self.counts.max().item(),
            "min_selection_count": self.counts[selected_mask].min().item() if selected_mask.any() else 0,
        }


def compute_coverage_regularization(
    probs: torch.Tensor,
    coverage_bonus: torch.Tensor,
    lambda_coverage: float,
) -> torch.Tensor:
    """
    Compute coverage regularization loss.

    Encourages the router to assign higher probability to less-covered samples.

    Args:
        probs: Router probability distribution [pool_size]
        coverage_bonus: Coverage bonus for each sample [pool_size]
        lambda_coverage: Weight for coverage regularization

    Returns:
        Coverage regularization loss (to be added to router loss)
    """
    # Weighted negative log probability: encourage high prob for high-bonus samples
    # Loss = -sum(bonus * prob), so minimizing encourages prob where bonus is high
    coverage_loss = -lambda_coverage * (coverage_bonus * probs).sum()
    return coverage_loss


# =============================================================================
# Selection strategies
# =============================================================================

def select_samples(
    probs: torch.Tensor,  # [M]
    k: int,
    strategy: str,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Select k samples from pool based on router probabilities.

    Args:
        probs: Probability distribution over pool [M]
        k: Number of samples to select
        strategy: 'topk', 'sample', or 'epsilon_greedy'
        epsilon: Exploration rate for epsilon_greedy

    Returns:
        Selected indices [k]
    """
    if strategy == "topk":
        return torch.topk(probs, k=k).indices

    elif strategy == "sample":
        # Sample without replacement according to probabilities
        return torch.multinomial(probs, num_samples=k, replacement=False)

    elif strategy == "epsilon_greedy":
        # With probability epsilon, sample randomly; otherwise use topk
        if torch.rand(1).item() < epsilon:
            # Random selection
            perm = torch.randperm(len(probs), device=probs.device)
            return perm[:k]
        else:
            return torch.topk(probs, k=k).indices

    else:
        return torch.topk(probs, k=k).indices


# =============================================================================
# Reward computation
# =============================================================================

def compute_entropy_per_sample(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample entropy from logits.

    Args:
        logits: [B, L, V] tensor of logits

    Returns:
        entropy: [B] tensor of mean entropy per sample
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # [B, L, V]
    # Compute entropy: -sum(p * log(p))
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
    return entropy.mean(dim=1)  # [B]


def compute_gradient_reward(
    params: list[torch.nn.Parameter],
    grad_ema: list[torch.Tensor] | None,
    reward_signal: str,
    ema_momentum: float,
    param_count: int,
    clip: float | None = None,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    """
    Compute scalar gradient-based reward from current LM gradients.

    Uses either L2 norm (gradient_norm) or alignment with EMA (gradient_alignment).
    """
    if reward_signal not in ("gradient_norm", "gradient_alignment"):
        return None, grad_ema
    if not params:
        return None, grad_ema

    device = params[0].device
    grad_norm_sq = torch.zeros((), device=device)
    dot = torch.zeros((), device=device) if reward_signal == "gradient_alignment" else None

    if reward_signal == "gradient_alignment" and grad_ema is None:
        grad_ema = [torch.zeros_like(p, device=p.device) for p in params]

    for i, p in enumerate(params):
        if p.grad is None:
            continue
        g = p.grad.detach()
        grad_norm_sq = grad_norm_sq + (g * g).sum()
        if reward_signal == "gradient_alignment":
            dot = dot + (g * grad_ema[i]).sum()
            grad_ema[i] = ema_momentum * grad_ema[i] + (1 - ema_momentum) * g

    grad_norm = grad_norm_sq.sqrt()
    if param_count > 0:
        scale = math.sqrt(param_count)
        grad_norm = grad_norm / scale
        if reward_signal == "gradient_alignment":
            dot = dot / float(param_count)

    reward = grad_norm if reward_signal == "gradient_norm" else dot
    if clip is not None and clip > 0:
        reward = reward.clamp(min=-clip, max=clip)
    return reward, grad_ema


def compute_reward(
    loss_before: torch.Tensor,
    loss_after: torch.Tensor,
    reward_signal: str,
    difficulty: torch.Tensor | None = None,
    entropy_before: torch.Tensor | None = None,
    entropy_after: torch.Tensor | None = None,
    gradient_reward: torch.Tensor | None = None,
    cfg: ExperimentConfig | None = None,
) -> torch.Tensor:
    """
    Compute reward signal for router update.

    Args:
        loss_before: Per-sample loss before LM update [B]
        loss_after: Per-sample loss after LM update [B]
        reward_signal: Type of reward signal to compute
        difficulty: Per-sample difficulty scores [B] (optional)
        entropy_before: Per-sample entropy before update [B] (optional)
        entropy_after: Per-sample entropy after update [B] (optional)
        gradient_reward: Scalar or per-sample gradient reward (optional)
        cfg: Config for reward weights (optional, needed for 'combined')

    Returns:
        reward: Per-sample reward [B]
    """
    improvement = (loss_before - loss_after).clamp(min=0.0)

    if reward_signal == "loss_improvement":
        return improvement

    elif reward_signal == "neg_loss":
        return -loss_after

    elif reward_signal == "relative_improvement":
        # Normalize improvement by initial loss
        # Avoids division by zero with small epsilon
        return improvement / (loss_before + 1e-8)

    elif reward_signal == "difficulty_weighted":
        # Reward more for learning difficult samples
        if difficulty is not None:
            # Normalize difficulty to [0, 1] range within batch
            diff_min = difficulty.min()
            diff_max = difficulty.max()
            diff_norm = (difficulty - diff_min) / (diff_max - diff_min + 1e-8)
            return improvement * (1.0 + diff_norm)
        return improvement

    elif reward_signal == "uncertainty_reduction":
        # Reward reducing model uncertainty (entropy)
        if entropy_before is not None and entropy_after is not None:
            entropy_reduction = (entropy_before - entropy_after).clamp(min=0.0)
            return entropy_reduction
        return improvement

    elif reward_signal in ("gradient_norm", "gradient_alignment"):
        if gradient_reward is None:
            return torch.zeros_like(loss_before)
        if gradient_reward.dim() == 0:
            return gradient_reward.expand_as(loss_before)
        return gradient_reward

    elif reward_signal == "combined":
        # Weighted combination of multiple signals
        if cfg is None:
            return improvement

        reward = cfg.reward_weight_improvement * improvement

        if difficulty is not None:
            diff_min = difficulty.min()
            diff_max = difficulty.max()
            diff_norm = (difficulty - diff_min) / (diff_max - diff_min + 1e-8)
            reward += cfg.reward_weight_difficulty * improvement * diff_norm

        if entropy_before is not None and entropy_after is not None:
            entropy_reduction = (entropy_before - entropy_after).clamp(min=0.0)
            # Normalize entropy reduction
            ent_max = entropy_reduction.max() + 1e-8
            reward += cfg.reward_weight_uncertainty * (entropy_reduction / ent_max)

        return reward

    else:
        return improvement


# =============================================================================
# Baseline computation
# =============================================================================

class MovingAverageBaseline:
    """Exponential moving average baseline for variance reduction."""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.value = None

    def update(self, reward: torch.Tensor) -> torch.Tensor:
        """Update baseline and return current value."""
        batch_mean = reward.mean().detach()
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * batch_mean
        return self.value


def compute_baseline(
    reward: torch.Tensor,
    baseline_type: str,
    moving_avg_baseline: MovingAverageBaseline | None = None,
) -> torch.Tensor:
    """Compute baseline for variance reduction."""
    if baseline_type == "batch_mean":
        return reward.mean().detach()
    elif baseline_type == "moving_avg":
        if moving_avg_baseline is not None:
            return moving_avg_baseline.update(reward)
        return reward.mean().detach()
    elif baseline_type == "none":
        return torch.zeros(1, device=reward.device)
    else:
        return reward.mean().detach()


# =============================================================================
# Training algorithms
# =============================================================================

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
    """Standard REINFORCE update with entropy regularization."""
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
    Group Relative Policy Optimization (GRPO) update.

    Instead of a global baseline, GRPO computes advantages relative to
    groups of samples, providing more stable gradients.
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
    entropy_type: str = "shannon",
    entropy_alpha: float = 2.0,
    entropy_q: float = 2.0,
    coverage_loss: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Proximal Policy Optimization (PPO) update.

    Performs multiple epochs of updates with clipped objective.
    """
    advantage = (reward - baseline).detach()
    # Normalize advantages
    adv_std = advantage.std().clamp(min=1e-8)
    advantage = (advantage - advantage.mean()) / adv_std

    total_loss = torch.tensor(0.0, device=cfg.device)
    total_policy_loss = torch.tensor(0.0, device=cfg.device)
    total_entropy = torch.tensor(0.0, device=cfg.device)

    for _ in range(cfg.ppo_epochs):
        # Recompute probabilities with current router
        scores = router(feats)
        probs = torch.softmax(scores / cfg.temp, dim=0)
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


def train_router_experiments(
    cfg: ExperimentConfig,
    model: TinyGPT,
    router: AttentionRouter,
    train_ds: MixedLMDataset,
    val_ds: MixedLMDataset,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> Tuple[TinyGPT, AttentionRouter]:
    """
    Train LM with a router using configurable curriculum learning.

    Supports multiple training algorithms, selection strategies, and schedules
    configured via ExperimentConfig.
    """

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=cfg.experiment_name,
        )

    model.to(cfg.device).train()
    router.to(cfg.device).train()

    opt_lm = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.Adam(router.parameters(), lr=cfg.lr_router)

    grad_params = [p for p in model.parameters() if p.requires_grad]
    grad_param_count = sum(p.numel() for p in grad_params)
    grad_ema = None

    # Initialize moving average baseline if needed
    moving_avg_baseline = None
    if cfg.baseline_type == "moving_avg":
        moving_avg_baseline = MovingAverageBaseline(cfg.baseline_momentum)

    # Initialize entropy targeting if enabled
    entropy_targeting = None
    if cfg.use_entropy_targeting:
        max_ent = compute_max_entropy(cfg.pool)
        target_entropy = cfg.target_entropy_ratio * max_ent
        entropy_targeting = EntropyTargeting(
            target_entropy=target_entropy,
            initial_lambda=cfg.lambda_ent,
            lr=cfg.entropy_lr,
            device=cfg.device,
        )

    # Initialize coverage tracker if enabled
    coverage_tracker = None
    if cfg.use_coverage_regularization:
        coverage_tracker = CoverageTracker(
            dataset_size=len(train_ds),
            coverage_type=cfg.coverage_type,
            decay=cfg.coverage_decay,
            device=cfg.device,
        )

    total_steps = max(1, (len(train_ds) // cfg.pool) * cfg.epochs)
    global_step = 0

    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in tqdm(idx_loader):
            if len(pool_indices) < cfg.batch:
                continue

            # Compute training progress for schedules
            progress = global_step / total_steps

            # Get scheduled values
            current_temp = get_scheduled_value(
                cfg.temp_schedule, cfg.temp, cfg.temp_min, progress,
                step=global_step, cycle_length=cfg.entropy_cycle_length,
            )

            # Get entropy coefficient (from targeting or schedule)
            if entropy_targeting is not None:
                current_lambda_ent = entropy_targeting.lambda_ent
            else:
                current_lambda_ent = get_scheduled_value(
                    cfg.entropy_schedule, cfg.lambda_ent, cfg.lambda_ent_min, progress,
                    step=global_step, cycle_length=cfg.entropy_cycle_length,
                )

            batch = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = zip(*batch)

            X = torch.stack(xs).to(cfg.device)  # [M, L]
            Y = torch.stack(ys).to(cfg.device)  # [M, L]

            # --- Router features over the full pool ---
            feats = extract_router_features(
                model=model,
                X=X,
                cfg=cfg,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
            )  # [M, F]

            scores = router(feats)  # [M]
            probs = torch.softmax(scores / current_temp, dim=0)  # [M]

            # --- Sample selection based on strategy ---
            sel_idx = select_samples(
                probs=probs,
                k=cfg.batch,
                strategy=cfg.selection_strategy,
                epsilon=cfg.epsilon_greedy,
            )
            sel_probs = probs[sel_idx].clamp_min(1e-12)

            # Store old log probs for PPO
            old_log_probs = sel_probs.log().detach()

            X_sel = X[sel_idx]  # [B, L]
            Y_sel = Y[sel_idx]  # [B, L]
            selected_diffs = [diffs[i] for i in sel_idx.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx.tolist()]

            # --- LM forward and update ---
            opt_lm.zero_grad()

            logits = model(X_sel)  # [B, L, V]

            # per-sample loss and entropy BEFORE update
            with torch.no_grad():
                loss_before = compute_loss_per_sample_vectorized(logits, Y_sel)
                entropy_before = compute_entropy_per_sample(logits) if cfg.reward_signal in ("uncertainty_reduction", "combined") else None

            # scalar loss for LM update
            B, L, V = logits.shape
            loss_lm = F.cross_entropy(
                logits.view(B * L, V),
                Y_sel.view(B * L),
                reduction="mean",
            )

            loss_lm.backward()

            gradient_reward = None
            if cfg.reward_signal in ("gradient_norm", "gradient_alignment"):
                gradient_reward, grad_ema = compute_gradient_reward(
                    params=grad_params,
                    grad_ema=grad_ema,
                    reward_signal=cfg.reward_signal,
                    ema_momentum=cfg.gradient_ema_momentum,
                    param_count=grad_param_count,
                    clip=cfg.gradient_reward_clip,
                )
            opt_lm.step()

            # loss and entropy AFTER update
            with torch.no_grad():
                logits_after = model(X_sel)
                loss_after = compute_loss_per_sample_vectorized(logits_after, Y_sel)
                entropy_after = compute_entropy_per_sample(logits_after) if cfg.reward_signal in ("uncertainty_reduction", "combined") else None

            # Get difficulty scores for selected samples
            difficulty_tensor = torch.tensor(selected_diffs, device=cfg.device, dtype=torch.float32) if cfg.reward_signal in ("difficulty_weighted", "combined") else None

            # --- Compute reward ---
            reward = compute_reward(
                loss_before=loss_before,
                loss_after=loss_after,
                reward_signal=cfg.reward_signal,
                difficulty=difficulty_tensor,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                gradient_reward=gradient_reward,
                cfg=cfg,
            )

            # --- Compute coverage regularization if enabled ---
            coverage_loss = None
            if coverage_tracker is not None:
                coverage_bonus = coverage_tracker.get_coverage_bonus(
                    pool_indices, cfg.coverage_temperature
                )
                coverage_loss = compute_coverage_regularization(
                    probs, coverage_bonus, cfg.lambda_coverage
                )

            # Common entropy parameters
            ent_kwargs = {
                "entropy_type": cfg.entropy_type,
                "entropy_alpha": cfg.entropy_alpha,
                "entropy_q": cfg.entropy_q,
                "coverage_loss": coverage_loss,
            }

            # --- Router update based on training algorithm ---
            if cfg.training_algorithm == "reinforce":
                baseline = compute_baseline(reward, cfg.baseline_type, moving_avg_baseline)
                loss_router, policy_loss, entropy = reinforce_update(
                    router=router,
                    opt_router=opt_router,
                    reward=reward,
                    baseline=baseline,
                    sel_probs=sel_probs,
                    all_probs=probs,
                    lambda_ent=current_lambda_ent,
                    **ent_kwargs,
                )

            elif cfg.training_algorithm == "grpo":
                loss_router, policy_loss, entropy = grpo_update(
                    router=router,
                    opt_router=opt_router,
                    reward=reward,
                    sel_probs=sel_probs,
                    all_probs=probs,
                    lambda_ent=current_lambda_ent,
                    group_size=cfg.grpo_group_size,
                    **ent_kwargs,
                )

            elif cfg.training_algorithm == "ppo":
                baseline = compute_baseline(reward, cfg.baseline_type, moving_avg_baseline)
                loss_router, policy_loss, entropy = ppo_update(
                    model=model,
                    router=router,
                    opt_router=opt_router,
                    X_sel=X_sel,
                    Y_sel=Y_sel,
                    old_log_probs=old_log_probs,
                    reward=reward,
                    baseline=baseline,
                    feats=feats,
                    sel_idx=sel_idx,
                    cfg=cfg,
                    lambda_ent=current_lambda_ent,
                    **ent_kwargs,
                )

            else:
                # Default to REINFORCE
                baseline = compute_baseline(reward, cfg.baseline_type, moving_avg_baseline)
                loss_router, policy_loss, entropy = reinforce_update(
                    router=router,
                    opt_router=opt_router,
                    reward=reward,
                    baseline=baseline,
                    sel_probs=sel_probs,
                    all_probs=probs,
                    lambda_ent=current_lambda_ent,
                    **ent_kwargs,
                )

            # Update entropy targeting if enabled
            if entropy_targeting is not None:
                entropy_targeting.update(-entropy)  # Note: entropy is negative

            # Update coverage tracker if enabled
            if coverage_tracker is not None:
                coverage_tracker.update(selected_indices, loss_after)

            diversity.update(selected_indices, selected_diffs)

            # --- Logging ---
            global_step += 1
            if global_step % cfg.log_every == 0:
                curriculum_strength = 1.0 - progress

                log_data = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss_lm": loss_lm.item(),
                    "loss_router": loss_router.item(),
                    "policy_loss": policy_loss.item(),
                    "entropy": -entropy.item(),
                    "avg_reward": reward.mean().item(),
                    "curriculum_strength": curriculum_strength,
                    "temperature": current_temp,
                    "lambda_ent": current_lambda_ent,
                    **diversity.get_metrics(),
                }

                # Add coverage stats if enabled
                if coverage_tracker is not None:
                    log_data.update(coverage_tracker.get_coverage_stats())

                metrics.log(**log_data)

                print(
                    f"[{cfg.training_algorithm.upper()}] Step {global_step} | "
                    f"loss_lm={loss_lm.item():.4f} | "
                    f"loss_router={loss_router.item():.4f} | "
                    f"temp={current_temp:.3f}"
                )

        # --- Validation ---
        loss_fn = nn.CrossEntropyLoss()
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn, cfg)

        metrics.log(
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
        )

        print(
            f"[{cfg.training_algorithm.upper()}] Epoch {epoch + 1}/{cfg.epochs} | "
            f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f}"
        )

    if cfg.use_wandb:
        import wandb
        wandb.finish()

    return model, router


def compare_runs_experiments(
    baseline_metrics: MetricsTracker,
    router_metrics: MetricsTracker,
    experiment_metrics: MetricsTracker,
) -> None:
    """Compare final validation perplexity across runs."""
    base_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()
    experiment_ppl = experiment_metrics.get_final_ppl()

    if base_ppl is None or router_ppl is None or experiment_ppl is None:
        print("Missing val_ppl in metrics; cannot compare.")
        return

    router_gain = (base_ppl - router_ppl) / base_ppl * 100.0
    experiment_gain = (base_ppl - experiment_ppl) / base_ppl * 100.0

    print("\n=== Comparison ===")
    print(f"Baseline   val_ppl: {base_ppl:.1f}")
    print(f"Router     val_ppl: {router_ppl:.1f} ({router_gain:.2f}%)")
    print(f"Experiment val_ppl: {experiment_ppl:.1f} ({experiment_gain:.2f}%)")
