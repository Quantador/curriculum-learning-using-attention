"""
Advanced RL-based training loops for curriculum learning experiments.

This is the experiment-grade router training loop used by parallel_experiments.py
/ utils/experiment_worker.py (ablation studies, one subprocess per config).

Key features over the reference loop in training.py:
  - Three policy gradient algorithms: REINFORCE, GRPO, PPO
  - Nine reward signals: loss_improvement, neg_loss, relative_improvement,
    difficulty_weighted, uncertainty_reduction, gradient_norm,
    gradient_alignment, combined, greats_score
  - Four entropy formulations: Shannon, Rényi, Tsallis, KL-uniform
  - SAC-style entropy targeting (auto-adjusts lambda_ent to hit a target entropy)
  - Coverage regularisation (penalises repeated sample selection)
  - Feature caching (amortises expensive transformer forward passes)
  - Supervised aux-net baseline (MSE alternative to policy gradient)

Entropy sign convention — READ THIS:
  All compute_*_entropy() functions return -H, the *negative* entropy.
  Adding `lambda_ent * entropy_term` to the router loss therefore
  penalises low-entropy distributions: minimising the total loss
  *maximises* entropy and encourages diverse sample selection.
  The logged 'entropy' value is always negated before display so the
  dashboard shows a positive, human-readable entropy number.

Entry points:
  train_router_experiments() — main RL training loop (REINFORCE/GRPO/PPO)
  train_aux_baseline()       — supervised MSE alternative
  compare_runs_experiments() — prints a performance comparison table
"""
from __future__ import annotations

import contextlib
import math
import os
import random
import time
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from config import ExperimentConfig
from data import make_pool_loader, TokenizedCorpus
from models.model import TinyGPT, AttentionRouter, extract_hierarchical_hidden, compute_text_statistics
from utils.metrics import MetricsTracker, DiversityTracker
from training import evaluate_per_domain
from models.router import extract_router_features
from GhostSuite.ghostEngines.engine_manager import GhostEngineManager
from utils.rl_utils import grpo_update, ppo_update, reinforce_update
from utils.general_utils import autocast_ctx
from utils.distributed_utils import eval_handles, wrap_model, wrap_replica

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
    progress: float,  # 0.0 to 1.0 , usually calculated from how many steps have passed.
    step: int = 0,
    cycle_length: int = 1000,
) -> float:
    """
    Return a scheduled hyperparameter value at the given training progress.

    Used for temperature annealing (cfg.temp_schedule) and entropy coefficient
    annealing (cfg.entropy_schedule). All schedules interpolate from `initial`
    at progress=0.0 to `minimum` at progress=1.0.

    Schedules:
      'fixed'             — constant initial throughout training
      'linear_decay'      — linear interpolation from initial to minimum
      'cosine_decay'      — cosine annealing (smooth S-curve decay)
      'exponential_decay' — fast initial drop, slower tail
      'cyclic'            — cosine warm restarts every cycle_length steps
      'adaptive'          — placeholder, returns initial (not implemented)
    """
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

    def get_coverage_stats(self, world_size: int = 1) -> dict:
        """
        Get coverage statistics for logging.

        world_size > 1: self.counts only reflects this rank's own
        DistributedSampler-sharded slice of the dataset (see data.py's
        make_pool_loader and CoverageTracker's class docstring on why that's
        the *correct* rank-local state for get_coverage_bonus()'s training
        signal). For logging, though, pass cfg.world_size to merge every
        rank's counts via all_reduce first so the reported coverage
        describes the whole world, not just this rank's shard. Uses a local
        copy -- self.counts (and therefore get_coverage_bonus()) is never
        mutated by this call. Every rank must call this the same number of
        times with the same world_size (it's a collective) -- true here
        since it's only ever invoked from a cfg.log_every-gated block that
        every rank reaches in lockstep.
        """
        counts = self.counts
        if world_size > 1:
            counts = counts.clone()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        selected_mask = counts > 0
        return {
            "coverage_ratio": selected_mask.float().mean().item(),
            "avg_selection_count": counts[selected_mask].mean().item() if selected_mask.any() else 0,
            "max_selection_count": counts.max().item(),
            "min_selection_count": counts[selected_mask].min().item() if selected_mask.any() else 0,
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


def pool_difficulty_stats(values: torch.Tensor, diffs: torch.Tensor, name: str) -> dict:
    """min/max/mean/count of `values` (aligned with the full candidate pool,
    not just the selected batch), split by difficulty label (0=easy, 1=hard).

    Used to diagnose curriculum collapse: whether hard samples are getting
    low router scores/probs, or low measured loss-improvement, or both.
    """
    stats = {}
    for label, group in (("easy", 0), ("hard", 1)):
        mask = diffs == group
        if mask.any():
            group_vals = values[mask]
            stats[f"{name}_{label}_min"] = group_vals.min().item()
            stats[f"{name}_{label}_max"] = group_vals.max().item()
            stats[f"{name}_{label}_mean"] = group_vals.mean().item()
            stats[f"{name}_{label}_count"] = float(mask.sum().item())
    return stats


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
    greats_reward: torch.Tensor | None = None,
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
        greats_reward: Scalar GREATS ghost-gradient-dot-product score, summed over
            the selected batch (optional; see reward_signal='greats_score' in Config).
            When cfg.greats_diversity_term is set, the caller has already folded the
            second-order redundancy penalty into this value before passing it in.
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

    elif reward_signal == "greats_score":
        # A single scalar (sum of the ghost gradient-dot-product scores of the
        # selected batch, against the fixed val batch) shared by every selected
        # sample — the router's "action" is the joint selection, not per-sample.
        if greats_reward is None:
            return torch.zeros_like(loss_before)
        if greats_reward.dim() == 0:
            return greats_reward.expand_as(loss_before)
        return greats_reward

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

def build_feature_cache(
    model: TinyGPT,
    train_ds: TokenizedCorpus,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    """
    Pre-compute and cache hierarchical hidden features for the full training set.

    Running a full transformer forward pass over M pool samples at every step
    is the dominant cost when enable_text_hierarchical=True. This function
    amortises that cost by running the model once over the entire dataset,
    storing the result as fp16 on CPU, and reusing it for feature_cache_epochs
    epochs before rebuilding.

    Cache validity: if the stored shape or dtype does not match expectations
    (e.g. after changing d_model or n_chunks), the cache is discarded and rebuilt.

    Disk persistence: if cfg.feature_cache_path is non-empty, the cache is saved
    as a .pt file and loaded on the next call instead of recomputing. This is an
    opt-in, cross-run cache -- if it's set and already holds a shape/dtype-matching
    file, that file is reused as-is even on a mid-run rebuild call, so don't point
    two runs with different model weights (or two feature_cache_epochs rebuilds
    you want to actually diverge) at the same path.

    Distributed (cfg.world_size > 1): only rank 0 runs the expensive full-dataset
    forward pass; the other ranks block on a barrier and then load the identical
    cache rank 0 just wrote to a same-run scratch file under
    results/<experiment_name>/. Without this, every one of world_size ranks would
    redundantly recompute (and separately hold in CPU RAM) its own byte-identical
    copy -- at world_size=16 that's 16x the GPU compute for zero benefit. Each
    rank still ends up with its own full in-memory copy afterwards (there's no
    cross-process shared memory here), so CPU RAM use is unchanged at
    world_size * cache_size_in_bytes -- only the redundant *compute* is removed.
    This scratch file is intentionally separate from cfg.feature_cache_path
    (that one is the user-facing, opt-in, persists-across-runs cache described
    above; reusing it here would make every within-run rebuild after the first
    silently reload the first rebuild's now-stale features instead of the fresh
    ones this call just computed).

    Returns: [N, n_chunks * d_model] fp16 CPU tensor.
    """
    expected_shape = (len(train_ds), cfg.n_chunks * cfg.d_model)

    # Try loading from disk if a valid file already exists
    if cfg.feature_cache_path and os.path.exists(cfg.feature_cache_path):
        try:
            cache = torch.load(cfg.feature_cache_path, map_location="cpu", weights_only=True)
            if tuple(cache.shape) == expected_shape and cache.dtype == torch.float16:
                if cfg.rank == 0:
                    print(f"[Cache] Loaded from {cfg.feature_cache_path}")
                return cache
            if cfg.rank == 0:
                print(f"[Cache] Shape mismatch ({cache.shape} vs {expected_shape}), rebuilding...")
        except Exception as e:
            if cfg.rank == 0:
                print(f"[Cache] Could not load ({e}), rebuilding...")

    ddp_sync_path = (
        os.path.join("results", cfg.experiment_name, "_feature_cache_ddp_sync.pt")
        if cfg.world_size > 1 else None
    )
    if ddp_sync_path and cfg.rank != 0:
        # Rank 0 is about to (re)build and save the cache below -- wait for
        # it instead of redundantly repeating the same full-dataset forward
        # pass on this rank's own model replica.
        dist.barrier()
        cache = torch.load(ddp_sync_path, map_location="cpu", weights_only=True)
        # Second barrier: only release rank 0 (waiting at the matching
        # barrier below) once every rank has finished reading, so rank 0
        # can't race ahead into a *later* rebuild and overwrite ddp_sync_path
        # while a slow rank is still mid-load here.
        dist.barrier()
        return cache

    N, F = expected_shape
    cache = torch.zeros(N, F, dtype=torch.float16)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, N, cfg.feature_cache_batch_size), desc="Building feature cache"):
            end = min(start + cfg.feature_cache_batch_size, N)
            xs = [train_ds[i][0] for i in range(start, end)]
            X = torch.stack(xs).to(cfg.device)
            hidden = extract_hierarchical_hidden(model, X, cfg)  # [B, n_chunks * d_model]
            cache[start:end] = hidden.cpu().half()
    model.train()

    if cfg.feature_cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.feature_cache_path)), exist_ok=True)
        torch.save(cache, cfg.feature_cache_path)
        print(f"[Cache] Saved to {cfg.feature_cache_path}")

    if ddp_sync_path:
        os.makedirs(os.path.dirname(ddp_sync_path), exist_ok=True)
        torch.save(cache, ddp_sync_path)
        dist.barrier()  # release the ranks waiting above
        dist.barrier()  # wait until every rank has finished reading it

    return cache


def train_router_experiments(
    cfg: ExperimentConfig,
    model: TinyGPT,
    router: AttentionRouter,
    train_ds: TokenizedCorpus,
    val_ds: TokenizedCorpus,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> Tuple[TinyGPT, AttentionRouter]:
    """
    Train LM with a router using configurable curriculum learning.

    Supports multiple training algorithms, selection strategies, and schedules
    configured via ExperimentConfig.
    """

    if cfg.use_wandb and cfg.rank == 0:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=cfg.experiment_name,
        )
        if cfg.config_path:
            wandb.save(cfg.config_path, policy="now")

    model.to(cfg.device).train()
    router.to(cfg.device).train()
    # DDP-replicated or FSDP2-sharded per cfg.distributed. The router is
    # always replicated, never sharded -- it's a few thousand parameters, so
    # sharding it would cost collectives to save nothing.
    model = wrap_model(model, cfg)
    router = wrap_replica(router, cfg)

    # Which module to evaluate, and whether this rank must join in: DDP
    # evaluates the unwrapped replica on rank 0 alone, FSDP evaluates the
    # sharded module on every rank (no rank holds a whole copy). See
    # utils/distributed_utils.eval_handles().
    eval_model, this_rank_evaluates = eval_handles(model, cfg)

    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.0)
    opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router, weight_decay=0.0)

    grad_params = [p for p in model.parameters() if p.requires_grad]
    grad_param_count = sum(p.numel() for p in grad_params)
    grad_ema = None

    # GREATS ghost gradient-dot-product scorer, built when reward_signal='greats_score'
    # (GPT-2-family HF checkpoints only, validated in Config.__post_init__);
    # GhostSuite's per-sample-gradient hooks can't see TinyGPT's nn.MultiheadAttention
    # or Qwen3's RMSNorm/custom Linear stack.
    ghost_engine = None
    if cfg.reward_signal == "greats_score":
        val_idx = random.sample(range(len(val_ds)), cfg.greats_val_batch_size)
        Xv, Yv, _ = zip(*(val_ds[i] for i in val_idx))
        X_val = torch.stack(Xv).to(cfg.device)
        Y_val = torch.stack(Yv).to(cfg.device)
        ghost_engine = GhostEngineManager(
            config=SimpleNamespace(
                method="GradDotProd",
                result_dir=os.path.join("results", cfg.experiment_name, "ghost"),
                val_batch_size=cfg.greats_val_batch_size,
                log_grad_norms=cfg.greats_log_grad_norms,
                score_exclude_params=cfg.greats_score_exclude_params,
                # Eager engine only: the decoupled/compiled fast path hardcodes
                # GPT-2/nanoGPT-shaped model.transformer.h + forward(idx, idx)->.loss,
                # which doesn't match build_model()'s HFCausalLM/TinyGPT forward signature.
                decoupled_fn=False,
                separate_val=False,
            ),
            # Unwrapped: GhostSuite's per-sample-gradient hooks match
            # nn.Linear/nn.Embedding/nn.LayerNorm/HF Conv1D by exact type (see
            # the model_type check above) and walk named_modules() directly --
            # a DDP wrapper would shadow every submodule path under "module."
            # and could interfere with per-sample gradient capture, so hand it
            # the real model regardless of whether DDP is wrapping it for the
            # actual LM forward/backward elsewhere in this function.
            model=model.module if isinstance(model, DDP) else model,
            optimizer=opt_lm,
            ddp_info={"master_process": cfg.rank == 0},
            val_data=(X_val, Y_val),
        )

    # Initialize moving average baseline if needed
    moving_avg_baseline = None
    if cfg.baseline_type == "moving_avg":
        moving_avg_baseline = MovingAverageBaseline(cfg.baseline_momentum)

    # Initialize entropy targeting if enabled
    entropy_targeting = None
    if cfg.use_entropy_targeting:
        max_ent = compute_max_entropy(cfg.per_rank_pool_size)
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

    # // world_size before // per_rank_pool_size: under DDP each rank only
    # sees its shard (make_pool_loader's DistributedSampler truncates to
    # len(ds)//world_size candidates per rank, drop_last=True -- see data.py),
    # then chunked into cfg.per_rank_pool_size-sized steps (cfg.pool split
    # across ranks -- see Config.per_rank_pool_size), so this must match
    # steps actually taken per rank per epoch, not the single-process count,
    # or progress (used below to drive every schedule) would never reach 1.0.
    total_steps = max(1, (len(train_ds) // cfg.world_size // cfg.per_rank_pool_size) * cfg.epochs)
    global_step = 0
    # Tokens fed to the LM so far. TokenizedCorpus yields fixed-length windows
    # (block tokens, no padding — see data.py), so this is just
    # X_sel.numel() accumulated each step; no tokenizer call needed.
    # Multiplied by world_size since only rank 0 logs but every rank processes
    # its own equally-sized shard each step under DDP.
    total_tokens_seen = 0

    # Feature cache: None until first rebuild (never built during epoch 0)
    feature_cache: torch.Tensor | None = None

    pool_loader = make_pool_loader(
        train_ds, cfg.per_rank_pool_size,
        num_workers=cfg.dataloader_num_workers, pin_memory=(cfg.device != "cpu"),
        rank=cfg.rank, world_size=cfg.world_size, seed=cfg.seed,
    )

    budget_reached = False
    for epoch in range(cfg.epochs):
        # DistributedSampler reshuffles from `seed + epoch`; without this call
        # every rank would see the identical pool order every epoch. No-op
        # (AttributeError-free via hasattr) in the single-process case, where
        # the default RandomSampler already reshuffles on every fresh
        # `for ... in loader`.
        if hasattr(pool_loader.sampler, "set_epoch"):
            pool_loader.sampler.set_epoch(epoch)

        # The feature cache is never built at epoch 0: the model's weights are
        # randomly initialised, so the hidden states are noise. Caching garbage
        # features would waste memory and mislead the router. Rebuilding every
        # feature_cache_epochs epochs (starting at epoch 1) keeps the cache
        # fresh as the model's representations improve.
        # Rebuild cache at epoch 1, then every feature_cache_epochs epochs after that
        if (
            cfg.feature_cache_epochs > 0
            and cfg.enable_text_hierarchical
            and epoch > 0
            and (epoch - 1) % cfg.feature_cache_epochs == 0
        ):
            feature_cache = build_feature_cache(model, train_ds, cfg)

        epoch_start = time.perf_counter()
        total_feat_time = 0.0

        # ── Per-step curriculum loop ──────────────────────────────────────────
        # Each iteration implements the core curriculum learning cycle:
        #   1. Sample M = cfg.per_rank_pool_size candidate indices (pre-shuffled each epoch,
        #      prefetched by pool_loader's workers while the previous step's
        #      GPU work is still running -- see cfg.dataloader_num_workers).
        #   2. Extract router features for all M samples.
        #   3. Router scores pool → softmax(/ temp) → select k samples (k =
        #      cfg.per_rank_batch_size, or an annealed fraction of the pool when
        #      cfg.use_curriculum_ratio_schedule is on).
        #   4. LM forward + backward on selected batch.
        #   5. Compute reward signal (loss improvement, gradient norm, etc.).
        #   6. Router RL update (REINFORCE / GRPO / PPO + entropy regularisation).
        # ─────────────────────────────────────────────────────────────────────
        for pool_idx, X, Y, domains in tqdm(pool_loader, disable=(cfg.rank != 0)):
            pool_indices = pool_idx.tolist()
            domains = domains.tolist()
            X = X.to(cfg.device, non_blocking=True)  # [M, L]
            Y = Y.to(cfg.device, non_blocking=True)  # [M, L]

            # Compute training progress for schedules
            progress = global_step / total_steps

            # Router freeze: past this point the router keeps scoring/
            # selecting samples with its current weights (selection logic
            # below is completely unchanged), it just stops learning -- see
            # config.py's router_freeze_progress docstring.
            router_frozen = (
                cfg.router_freeze_progress is not None
                and progress >= cfg.router_freeze_progress
            )

            # Router update cadence: only the every-router_update_every-th
            # step actually pays for the extra loss_after forward pass and
            # performs a policy-gradient update below -- see config.py's
            # router_update_every docstring. Deterministic on every rank
            # (fixed per-step counter, no data dependence), so this is
            # DDP-safe without a broadcast, same as router_frozen above.
            router_update_due = global_step % cfg.router_update_every == 0

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

            # Curriculum-ratio schedule: shrink the selected batch from a
            # weakly-selective fraction of the pool down to a strongly-selective
            # one over training, instead of a fixed cfg.per_rank_batch_size. See
            # config.py's use_curriculum_ratio_schedule docstring.
            if cfg.use_curriculum_ratio_schedule:
                current_ratio = get_scheduled_value(
                    cfg.curriculum_ratio_schedule, cfg.curriculum_ratio_initial, cfg.curriculum_ratio_min,
                    progress, step=global_step, cycle_length=cfg.entropy_cycle_length,
                )
                select_k = max(1, min(len(pool_indices), round(current_ratio * len(pool_indices))))
            else:
                select_k = cfg.per_rank_batch_size

            # --- Router features over the full pool ---
            feat_start = time.perf_counter()
            external_embedding = None
            if train_ds.embeddings is not None:
                # .float() on read: TokenizedCorpus.embeddings is the fp16
                # sentence-embedder cache (a storage format, chosen to halve
                # its footprint), and everything downstream of it -- the
                # router, and the feature-cache concat below -- is fp32. Same
                # upcast-on-read the feature cache itself already does.
                external_embedding = torch.stack(
                    [train_ds.embeddings[i] for i in pool_indices]
                ).to(cfg.device).float()

            if feature_cache is not None:
                hidden_feats = feature_cache[pool_indices].to(cfg.device).float()
                parts = [hidden_feats]
                if cfg.enable_text_stat:
                    stats = compute_text_statistics(
                        X,
                        pad_token_id=tokenizer.pad_token_id,
                        vocab_size=tokenizer.vocab_size,
                        block=cfg.block,
                    )
                    parts.append(stats)
                if external_embedding is not None:
                    parts.append(external_embedding)
                feats = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            else:
                feats = extract_router_features(
                    model=model,
                    X=X,
                    cfg=cfg,
                    pad_token_id=tokenizer.pad_token_id,
                    vocab_size=tokenizer.vocab_size,
                    external_embedding=external_embedding,
                )  # [M, F]
            total_feat_time += time.perf_counter() - feat_start

            scores = router(feats)  # [M]
            probs = torch.softmax(scores / current_temp, dim=0)  # [M]

            # --- Sample selection based on strategy ---
            sel_idx = select_samples(
                probs=probs,
                k=select_k,
                strategy=cfg.selection_strategy,
                epsilon=cfg.epsilon_greedy,
            )
            sel_probs = probs[sel_idx].clamp_min(1e-12)

            # Store old log probs for PPO
            old_log_probs = sel_probs.log().detach()

            X_sel = X[sel_idx]  # [B, L]
            Y_sel = Y[sel_idx]  # [B, L]
            selected_domains = [domains[i] for i in sel_idx.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx.tolist()]
            total_tokens_seen += X_sel.numel() * cfg.world_size
            # Deterministic on every rank (fixed per-step increment, no data
            # dependence), so checking/breaking here is DDP-safe without a
            # broadcast -- every rank reaches the same verdict at the same point.
            budget_reached = cfg.max_tokens is not None and total_tokens_seen >= cfg.max_tokens

            # --- GREATS ghost-gradient scoring (before the real LM update: a separate
            # scoring backward on X_sel/Y_sel against the fixed val batch, discarded
            # afterwards) ---
            greats_reward = None
            greats_train_norms = None
            if ghost_engine is not None and cfg.reward_signal == "greats_score":
                ghost_engine.begin_step()
                ghost_engine.attach_train_batch(X_sel, Y_sel, global_step)
                with ghost_engine.saved_tensors_context():
                    Xf, Yf = ghost_engine.prepare_forward_input(X_sel, Y_sel)
                    logits_score = model(Xf)
                    B_s, L_s, V_s = logits_score.shape
                    loss_score = F.cross_entropy(
                        logits_score.view(B_s * L_s, V_s), Yf.view(B_s * L_s),
                    )
                    loss_score.backward()
                ghost_engine.collect_microbatch()
                greats_reward = ghost_engine.read_scores(
                    metric=cfg.greats_score_metric
                ).to(cfg.device).sum()
                # Per-sample ||g_i|| for the diversity term below (config.__post_init__
                # guarantees greats_log_grad_norms=True whenever greats_diversity_term is set).
                greats_train_norms = (
                    ghost_engine.read_train_grad_norms() if cfg.greats_diversity_term else None
                )
                ghost_engine.discard_scores()

            # --- LM forward and update ---
            # Clears the GREATS scoring pass's leftover grads (if any) as well as
            # zeroing for this real step.
            opt_lm.zero_grad()

            with autocast_ctx(cfg.device):
                logits = model(X_sel)  # [B, L, V]
            # Cast back to fp32 so every downstream consumer (entropy/reward
            # functions, GhostSuite's per-sample gradient hooks, DDP grad
            # sync) sees the same dtype as before -- autocast's memory win
            # comes from the transformer's internal activations having run
            # in bf16, not from the dtype of this final tensor.
            logits = logits.float()

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

            # DDP's Reducer all-reduces (averages) gradients across ranks as
            # part of backward() itself -- by the time backward() returns,
            # .grad is already the world-averaged gradient, not this rank's
            # own. That's exactly right for opt_lm.step() (one shared model),
            # but it means compute_gradient_reward() below would read a value
            # that's identical on every rank and barely moved by this rank's
            # own selection -- useless as a per-rank RL reward. no_sync()
            # skips that automatic reduction so backward() leaves .grad
            # purely local to this rank's own X_sel/Y_sel.
            #
            # The isinstance(model, DDP) test is what keeps this DDP-only.
            # FSDP2 has no no_sync() (its equivalent, set_requires_gradient_sync,
            # doesn't preserve this meaning -- gradients are reduce-scattered
            # into shards, so no rank ever holds its own complete gradient),
            # which is why ExperimentConfig.__post_init__ rejects these two
            # reward signals under distributed='FSDP' outright rather than
            # letting them silently fall through to the unreduced branch here.
            local_grad_reward = cfg.world_size > 1 and isinstance(model, DDP) and cfg.reward_signal in ("gradient_norm", "gradient_alignment")
            backward_ctx = model.no_sync() if local_grad_reward else contextlib.nullcontext()
            with backward_ctx:
                loss_lm.backward()

            gradient_reward = None
            if cfg.reward_signal in ("gradient_norm", "gradient_alignment"):
                # Gradient reward is computed AFTER loss_lm.backward() populates
                # .grad on all parameters but BEFORE opt_lm.step() zeroes them.
                # This window is the only point where the raw batch gradients exist.
                gradient_reward, grad_ema = compute_gradient_reward(
                    params=grad_params,
                    grad_ema=grad_ema,
                    reward_signal=cfg.reward_signal,
                    ema_momentum=cfg.gradient_ema_momentum,
                    param_count=grad_param_count,
                    clip=cfg.gradient_reward_clip,
                )

            if greats_train_norms is not None:
                # Second-order (Hessian ~= identity) redundancy penalty for the ALREADY-selected
                # batch X_sel: for a fixed set S, sum_{i<j in S} <g_i,g_j> collapses to
                # (||sum_i g_i||^2 - sum_i ||g_i||^2) / 2 -- no candidate-candidate Gram matrix and
                # no greedy loop needed (those are only required to *choose* S; the router already
                # did that). Read in the same post-backward, pre-zero_grad window as gradient_reward
                # above, since ||sum_i g_i||^2 comes straight from the real update's own .grad.
                #
                # Both terms must land in the SAME per-sample scale as greats_reward (a sum of
                # <g_i, g_val> dot products from the separate ghost pass over [X_sel ++ val]) to be
                # combined with it. The ghost pass's loss is mean-reduced over the combined
                # train+val batch, so every factor it produces implicitly carries a 1/total_bs
                # scale: greats_train_norms holds ||g_i,ghost|| where g_i,ghost = g_i / total_bs.
                # sum_i g_i,ghost = (train_bs / total_bs) * mean_i(g_i), and that mean IS the
                # gradient loss_lm.backward() just populated in .grad (mean-reduced over X_sel
                # alone, no val) -- so no extra forward/backward pass is needed for this term.
                with torch.no_grad():
                    train_bs = X_sel.shape[0]
                    total_bs = train_bs + ghost_engine.val_batch_size
                    agg_grad_norm_sq = sum(
                        p.grad.float().pow(2).sum()
                        for p in grad_params if p.grad is not None
                    )
                    agg_grad_norm_sq_native = (train_bs / total_bs) ** 2 * agg_grad_norm_sq
                    sum_sq_norms_native = greats_train_norms.to(cfg.device).float().pow(2).sum()
                    redundancy = (agg_grad_norm_sq_native - sum_sq_norms_native) / 2.0
                    greats_reward = cfg.lr_lm * greats_reward - (cfg.lr_lm ** 2) * redundancy

            if local_grad_reward:
                # no_sync() above skipped DDP's automatic averaging, so
                # manually replicate it now (sum then divide by world_size)
                # before opt_lm.step() -- otherwise every rank's LM replica
                # would drift out of sync, applying only its own local
                # gradient instead of the shared, averaged update.
                for p in grad_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.div_(cfg.world_size)

            opt_lm.step()

            # Skip it on steps that won't perform a router update (router_update_due is
            # False -- see config.py's router_update_every docstring); note
            # router_frozen deliberately does NOT skip this, so reward stays
            # logged every step even once the router has stopped learning.
            # The one other consumer, coverage_tracker.update()'s
            # uncertainty-based bonus, still needs a real per-sample loss
            # regardless, so keep computing it then.
            needs_real_loss_after = (
                cfg.reward_signal != "greats_score" and router_update_due
            ) or (
                coverage_tracker is not None and cfg.coverage_type == "uncertainty"
            )
            with torch.no_grad(), autocast_ctx(cfg.device):
                if needs_real_loss_after:
                    logits_after = model(X_sel).float()
                    loss_after = compute_loss_per_sample_vectorized(logits_after, Y_sel)
                    entropy_after = compute_entropy_per_sample(logits_after) if cfg.reward_signal in ("uncertainty_reduction", "combined") else None
                else:
                    # Discarded stand-in: never read by compute_reward()'s
                    # greats_score branch, and coverage_tracker.update()'s
                    # `losses` arg is only read when coverage_type ==
                    # 'uncertainty', ruled out above.
                    loss_after = loss_before
                    entropy_after = None
    
            # Get difficulty scores for selected samples
            # Conditions for selected_domains to act as difficulty markers 
            # are checked in config.py in post_init
            difficulty_tensor = torch.tensor(selected_domains, device=cfg.device, dtype=torch.float32) if cfg.reward_signal in ("difficulty_weighted", "combined") else None

            # --- Compute reward ---
            reward = compute_reward(
                loss_before=loss_before,
                loss_after=loss_after,
                reward_signal=cfg.reward_signal,
                difficulty=difficulty_tensor,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                gradient_reward=gradient_reward,
                greats_reward=greats_reward,
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
            if router_frozen or not router_update_due:
                # Router already scored/selected this step's samples above
                # with its current weights -- just skip the backward/
                # optimizer step, either because it's permanently frozen
                # (router_frozen) or because this isn't a router_update_every
                # update step (router_update_due). Zero placeholders keep the
                # unconditional logging code below (which reads loss_router/
                # policy_loss/entropy every step) working unchanged.
                loss_router = torch.zeros((), device=cfg.device)
                policy_loss = torch.zeros((), device=cfg.device)
                entropy = torch.zeros((), device=cfg.device)
            elif cfg.training_algorithm == "reinforce":
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
                    temperature=current_temp,
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

            # Update entropy targeting if enabled -- skipped when frozen or
            # this isn't a router_update_every update step, since entropy is
            # a zero placeholder then, not a real signal from an actual
            # router update.
            if entropy_targeting is not None and not router_frozen and router_update_due:
                entropy_targeting.update(-entropy)  # Note: entropy is negative

            # Update coverage tracker if enabled
            if coverage_tracker is not None:
                coverage_tracker.update(selected_indices, loss_after)

            diversity.update(selected_indices, selected_domains, loss_before.tolist())

            # --- Logging ---
            global_step += 1
            if global_step % cfg.log_every == 0:
                # loss_lm/loss_router/policy_loss/entropy/avg_reward are all
                # this rank's own local-shard values; average across ranks
                # before logging so world_size>1 runs plot the whole step,
                # not just rank 0's 1/world_size sliver. Every rank hits this
                # collective in lockstep (equal per-rank step counts, see
                # make_pool_loader's DistributedSampler) -- only rank 0
                # then actually writes to wandb/prints below.
                log_scalars = torch.stack([
                    loss_lm.detach(), loss_router.detach(), policy_loss.detach(),
                    entropy.detach(), reward.mean().detach(),
                ])
                if cfg.world_size > 1:
                    log_scalars = log_scalars.clone()
                    dist.all_reduce(log_scalars, op=dist.ReduceOp.SUM)
                    log_scalars /= cfg.world_size
                agg_loss_lm, agg_loss_router, agg_policy_loss, agg_entropy, agg_avg_reward = log_scalars.tolist()

                # get_metrics()/get_coverage_stats() themselves issue
                # collectives (all_reduce/all_gather_object) when
                # world_size > 1, so every rank must call them here, not
                # just rank 0.
                div_metrics = diversity.get_metrics(world_size=cfg.world_size)
                coverage_stats = (
                    coverage_tracker.get_coverage_stats(world_size=cfg.world_size)
                    if coverage_tracker is not None else None
                )

                if cfg.rank == 0:
                    curriculum_strength = 1.0 - progress

                    log_data = {
                        "epoch": epoch,
                        "step": global_step,
                        "loss_lm": agg_loss_lm,
                        "loss_router": agg_loss_router,
                        "policy_loss": agg_policy_loss,
                        "entropy": -agg_entropy,  # entropy is -H; negate to log positive H
                        "avg_reward": agg_avg_reward,
                        "curriculum_strength": curriculum_strength,
                        "tokens_seen": total_tokens_seen,
                        "temperature": current_temp,
                        "lambda_ent": current_lambda_ent,
                        "select_k": select_k,
                        "feat_time_ms": total_feat_time / cfg.log_every * 1000,
                        **div_metrics,
                    }
                    total_feat_time = 0.0

                    # Add coverage stats if enabled
                    if coverage_stats is not None:
                        log_data.update(coverage_stats)

                    metrics.log(**log_data)

                    print(
                        f"[{cfg.training_algorithm.upper()}] Step {global_step} | "
                        f"loss_lm={agg_loss_lm:.4f} | "
                        f"loss_router={agg_loss_router:.4f} | "
                        f"temp={current_temp:.3f} | "
                    )

            if budget_reached:
                break

        # Collective (all_gather_object under world_size>1) -- every rank must
        # reach this the same number of times, so it's called here, before the
        # this_rank_evaluates gate below (which excludes non-zero DDP ranks).
        train_ppl_domain = diversity.get_train_ppl_domain(world_size=cfg.world_size)

        # --- Validation ---
        # val_ds is small and identical on every rank. Under DDP only rank 0
        # evaluates and the others wait at the barrier below, so nobody starts
        # the next epoch's DDP-synchronizing .backward() mid-eval; under FSDP
        # every rank must evaluate (the forward all-gathers), and they all
        # compute the same number, so logging still happens on rank 0 only.
        if this_rank_evaluates:
            loss_fn = nn.CrossEntropyLoss()
            (val_loss, val_ppl), per_domain_ppl = evaluate_per_domain(eval_model, val_ds, loss_fn, cfg)

            epoch_time = time.perf_counter() - epoch_start
            if cfg.rank == 0:
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    epoch_time_s=epoch_time,
                    **{f"val_ppl_domain/{name}": ppl for name, (_, ppl) in per_domain_ppl.items()},
                    **train_ppl_domain,
                )

                print(
                    f"[{cfg.training_algorithm.upper()}] Epoch {epoch + 1}/{cfg.epochs} | "
                    f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | "
                    f"epoch_time={epoch_time:.1f}s"
                )
        if cfg.world_size > 1:
            dist.barrier()

        if budget_reached:
            break

    # per_domain_ppl is left over from the last epoch's evaluate_per_domain()
    # call above -- already logged there, so this is just the human-readable
    # summary of the fully trained model, with no extra forward pass.
    if this_rank_evaluates and cfg.rank == 0:
        print(
            "[Final per-domain val perplexity] "
            + ", ".join(f"{name}={ppl:.1f}" for name, (_, ppl) in sorted(per_domain_ppl.items()))
        )

    # wandb.finish() is deferred to the caller (utils/experiment_worker.py),
    # which logs a couple more summary metrics (e.g. total_time_s) into this
    # same run before closing it.

    return model, router


def _avg_epoch_time(metrics: MetricsTracker) -> float | None:
    times = metrics.history.get("epoch_time_s")
    if not times:
        return None
    # Skip epoch 0 — cache is never active then and it skews the average
    relevant = times[1:] if len(times) > 1 else times
    return sum(relevant) / len(relevant)

def train_aux_baseline(
    cfg: ExperimentConfig,
    model: TinyGPT,
    aux_net: nn.Module,
    train_ds: TokenizedCorpus,
    val_ds: TokenizedCorpus,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> tuple[TinyGPT, nn.Module]:
    """
    Curriculum learning via a supervised auxiliary network.

    At each step:
      1. Extract features for the pool.
      2. Score pool with aux_net (predicted loss improvement) → top-k selection.
      3. Compute actual per-sample loss_before and loss_after.
      4. Train aux_net with MSE(predicted, actual_improvement).
      5. Update LM on selected samples.

    This is a direct supervised alternative to the policy-gradient router:
    same features, same top-k selection, same feature pipeline — only the
    training objective differs (MSE regression vs. REINFORCE).
    """
    if cfg.use_wandb and cfg.rank == 0:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=cfg.experiment_name,
        )
        if cfg.config_path:
            wandb.save(cfg.config_path, policy="now")

    model.to(cfg.device)
    aux_net.to(cfg.device)
    model.train()
    aux_net.train()
    # DDP-replicated or FSDP2-sharded per cfg.distributed; aux_net is small
    # enough to always replicate (see wrap_replica).
    model = wrap_model(model, cfg)
    aux_net = wrap_replica(aux_net, cfg)
    eval_model, this_rank_evaluates = eval_handles(model, cfg)

    print(f"{aux_net=}")
    loss_fn = nn.CrossEntropyLoss()
    mse_fn  = nn.MSELoss()
    opt_lm  = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.0)
    opt_aux = torch.optim.AdamW(aux_net.parameters(), lr=cfg.lr_router, weight_decay=0.0)

    # // world_size before // per_rank_pool_size: under DDP each rank only
    # sees its shard (make_pool_loader's DistributedSampler truncates to
    # len(ds)//world_size candidates per rank, drop_last=True -- see data.py),
    # then chunked into cfg.per_rank_pool_size-sized steps (cfg.pool split
    # across ranks -- see Config.per_rank_pool_size), so this must match
    # steps actually taken per rank per epoch, not the single-process count,
    # or training_progress (used below) would never reach 1.0.
    total_steps = max(1, (len(train_ds) // cfg.world_size // cfg.per_rank_pool_size) * cfg.epochs)
    global_step = 0
    total_tokens_seen = 0

    pool_loader = make_pool_loader(
        train_ds, cfg.per_rank_pool_size,
        num_workers=cfg.dataloader_num_workers, pin_memory=(cfg.device != "cpu"),
        rank=cfg.rank, world_size=cfg.world_size, seed=cfg.seed,
    )

    budget_reached = False
    for epoch in range(cfg.epochs):
        if hasattr(pool_loader.sampler, "set_epoch"):
            pool_loader.sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()

        for pool_idx, X, Y, diffs in tqdm(pool_loader, disable=(cfg.rank != 0)):
            pool_indices = pool_idx.tolist()
            diffs = diffs.tolist()
            X = X.to(cfg.device, non_blocking=True)  # [M, L]
            Y = Y.to(cfg.device, non_blocking=True)  # [M, L]

            # --- Feature extraction ---
            external_embedding = None
            if train_ds.embeddings is not None:
                # .float() on read: TokenizedCorpus.embeddings is the fp16
                # sentence-embedder cache (a storage format, chosen to halve
                # its footprint), and everything downstream of it -- the
                # router, and the feature-cache concat below -- is fp32. Same
                # upcast-on-read the feature cache itself already does.
                external_embedding = torch.stack(
                    [train_ds.embeddings[i] for i in pool_indices]
                ).to(cfg.device).float()

            feats = extract_router_features(
                model=model,
                X=X,
                cfg=cfg,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
                external_embedding=external_embedding,
            )  # [M, F]

            # --- Selection: top-k by predicted improvement ---
            with torch.no_grad():
                predicted_improvement = aux_net(feats.detach())  # [M]
            topk = torch.topk(predicted_improvement, k=cfg.per_rank_batch_size)
            sel_idx_local = topk.indices

            X_sel = X[sel_idx_local]
            Y_sel = Y[sel_idx_local]
            feats_sel = feats[sel_idx_local]
            selected_diffs   = [diffs[i] for i in sel_idx_local.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx_local.tolist()]
            total_tokens_seen += X_sel.numel() * cfg.world_size
            # Deterministic on every rank (fixed per-step increment, no data
            # dependence), so checking/breaking here is DDP-safe without a
            # broadcast -- every rank reaches the same verdict at the same point.
            budget_reached = cfg.max_tokens is not None and total_tokens_seen >= cfg.max_tokens

            # --- Compute actual improvement ---
            with torch.no_grad(), autocast_ctx(cfg.device):
                loss_before = compute_loss_per_sample_vectorized(model(X_sel).float(), Y_sel)

            opt_lm.zero_grad()
            with autocast_ctx(cfg.device):
                logits_sel = model(X_sel)
            logits_sel = logits_sel.float()
            loss_lm = loss_fn(
                logits_sel.view(-1, logits_sel.size(-1)),
                Y_sel.view(-1),
            )
            loss_lm.backward()
            opt_lm.step()

            with torch.no_grad(), autocast_ctx(cfg.device):
                loss_after = compute_loss_per_sample_vectorized(model(X_sel).float(), Y_sel)

            actual_improvement = (loss_before - loss_after).clamp(min=0.0).detach()

            # --- Train aux_net with supervised MSE ---
            pred = aux_net(feats_sel.detach())  # [B]
            loss_aux = mse_fn(pred, actual_improvement)
            opt_aux.zero_grad()
            loss_aux.backward()
            opt_aux.step()

            diversity.update(selected_indices, selected_diffs, loss_before.tolist())

            global_step += 1
            if global_step % cfg.log_every == 0:
                # loss_lm/loss_aux/avg_improvement are all this rank's own
                # local-shard values; average across ranks before logging so
                # world_size>1 runs plot the whole step, not just rank 0's
                # 1/world_size sliver. Every rank hits this collective in
                # lockstep (equal per-rank step counts, see make_pool_loader's
                # DistributedSampler) -- only rank 0 then actually writes to
                # wandb/prints below.
                log_scalars = torch.stack([
                    loss_lm.detach(), loss_aux.detach(), actual_improvement.mean().detach(),
                ])
                if cfg.world_size > 1:
                    log_scalars = log_scalars.clone()
                    dist.all_reduce(log_scalars, op=dist.ReduceOp.SUM)
                    log_scalars /= cfg.world_size
                agg_loss_lm, agg_loss_aux, agg_avg_improvement = log_scalars.tolist()

                # get_metrics() itself issues collectives (all_reduce/
                # all_gather_object) when world_size > 1, so every rank must
                # call it here, not just rank 0.
                div_metrics = diversity.get_metrics(world_size=cfg.world_size)

                if cfg.rank == 0:
                    training_progress = global_step / total_steps
                    metrics.log(
                        epoch=epoch,
                        step=global_step,
                        loss_lm=agg_loss_lm,
                        loss_aux=agg_loss_aux,
                        avg_improvement=agg_avg_improvement,
                        curriculum_strength=1.0 - training_progress,
                        tokens_seen=total_tokens_seen,
                        **div_metrics,
                    )
                    print(
                        f"[AuxNet] Step {global_step} | "
                        f"loss_lm={agg_loss_lm:.4f} | "
                        f"loss_aux={agg_loss_aux:.6f}"
                    )

            if budget_reached:
                break

        # DDP: rank 0 alone evaluates the unwrapped replica, others wait at
        # the barrier below. FSDP: every rank must join the sharded forward's
        # all-gathers. See utils/distributed_utils.eval_handles().
        # Collective (all_gather_object under world_size>1) -- every rank must
        # reach this the same number of times, so it's called here, before the
        # this_rank_evaluates gate below (which excludes non-zero DDP ranks).
        train_ppl_domain = diversity.get_train_ppl_domain(world_size=cfg.world_size)

        if this_rank_evaluates:
            (val_loss, val_ppl), per_domain_ppl = evaluate_per_domain(eval_model, val_ds, loss_fn, cfg)
            epoch_time = time.perf_counter() - epoch_start
            if cfg.rank == 0:
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    epoch_time_s=epoch_time,
                    **{f"val_ppl_domain/{name}": ppl for name, (_, ppl) in per_domain_ppl.items()},
                    **train_ppl_domain,
                )
                print(
                    f"[AuxNet] Epoch {epoch + 1}/{cfg.epochs} | "
                    f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f}"
                )
        if cfg.world_size > 1:
            dist.barrier()

        if budget_reached:
            break

    # wandb.finish() is deferred to the caller (utils/experiment_worker.py),
    # which logs a couple more summary metrics (e.g. total_time_s) into this
    # same run before closing it.

    return model, aux_net


def compare_runs_experiments(
    baseline_metrics: MetricsTracker,
    router_metrics: MetricsTracker,
    experiment_metrics: MetricsTracker,
) -> None:
    """Compare final validation perplexity and training speed across runs."""
    base_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()
    experiment_ppl = experiment_metrics.get_final_ppl()

    if base_ppl is None or router_ppl is None or experiment_ppl is None:
        print("Missing val_ppl in metrics; cannot compare.")
        return

    router_gain = (base_ppl - router_ppl) / base_ppl * 100.0
    experiment_gain = (base_ppl - experiment_ppl) / base_ppl * 100.0

    print("\n=== Performance ===")
    print(f"Baseline   val_ppl: {base_ppl:.1f}")
    print(f"Router     val_ppl: {router_ppl:.1f} ({router_gain:+.2f}%)")
    print(f"Experiment val_ppl: {experiment_ppl:.1f} ({experiment_gain:+.2f}%)")

    router_time = _avg_epoch_time(router_metrics)
    experiment_time = _avg_epoch_time(experiment_metrics)

    if router_time is not None and experiment_time is not None:
        speedup = router_time / experiment_time
        time_saved = (router_time - experiment_time) / router_time * 100.0
        print("\n=== Speed (avg epoch time, excl. epoch 0) ===")
        print(f"Router     : {router_time:.1f}s / epoch")
        print(f"Experiment : {experiment_time:.1f}s / epoch  ({speedup:.2f}x speedup, {time_saved:.1f}% faster)")

    base_total_time = baseline_metrics.get_total_time()
    router_total_time = router_metrics.get_total_time()
    experiment_total_time = experiment_metrics.get_total_time()

    if base_total_time is not None or router_total_time is not None or experiment_total_time is not None:
        print("\n=== Total run time ===")
        for label, total_time in (
            ("Baseline", base_total_time),
            ("Router", router_total_time),
            ("Experiment", experiment_total_time),
        ):
            if total_time is not None:
                print(f"{label:<10} : {total_time:.1f}s ({total_time / 3600:.2f}h)")
