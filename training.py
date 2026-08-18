# training.py
"""
Reference training loop used for the baseline comparison.

  train_baseline() — uniform random batch selection, standard cross-entropy SGD.
                     No router. The performance floor every other method must beat.

For the full experiment-grade loop with configurable algorithms (REINFORCE/GRPO/PPO),
reward signals, entropy formulations, and feature caching, see rl_training.py.

evaluate_per_domain() (which also gives the aggregate) is shared by
train_baseline() and rl_training.py; evaluate() is the plain aggregate-only
building block it's defined against.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from config import Config
from data import make_baseline_loader, TokenizedCorpus
from models.model import TinyGPT
from utils.general_utils import autocast_ctx
from utils.distributed_utils import eval_handles, wrap_model
from utils.metrics import MetricsTracker, DiversityTracker


def evaluate(
    model: TinyGPT,
    ds: TokenizedCorpus,
    loss_fn: nn.Module,
    cfg: Config,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """
    Compute mean cross-entropy loss and perplexity over the full dataset.

    Uses a plain sequential DataLoader (no shuffling needed for a full-pass
    eval) -- TokenizedCorpus windows are all exactly cfg.block tokens (no
    padding, see data.py), so batching never introduces padding artefacts.
    batch_size is independent of cfg.global_batch_size: no gradients are held here, so it
    can be much larger. pin_memory speeds up the host->GPU copy;
    cfg.dataloader_num_workers lets the next batch's CPU-side gather overlap
    with the current batch's forward pass (matters less here than in the
    per-step training loops, since eval only runs once per epoch).
    Sets model to eval() before the loop and restores train() after.

    Returns (avg_loss, perplexity) where perplexity = exp(avg_loss).
    """
    model.eval()
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=(cfg.device != "cpu"),
    )
    total_loss, total_tok = 0.0, 0
    with torch.no_grad(), autocast_ctx(cfg.device):
        for X, Y, _ in loader:
            X = X.to(cfg.device, non_blocking=True)
            Y = Y.to(cfg.device, non_blocking=True)
            logits = model(X)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
            )
            n_tok = Y.numel()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
    model.train()
    avg_loss = total_loss / max(total_tok, 1)
    return avg_loss, math.exp(avg_loss)


def evaluate_per_domain(
    model: TinyGPT,
    ds: TokenizedCorpus,
    loss_fn: nn.Module,
    cfg: Config,
    batch_size: int = 64,
) -> Tuple[Tuple[float, float], Dict[str, Tuple[float, float]]]:
    """
    Like evaluate(), but also broken out per domain: one (avg_loss, perplexity)
    pair per distinct domain in ds, computed in the SAME single pass over ds
    that produces the aggregate -- so this is safe to call every epoch in
    place of evaluate() instead of only as a one-off final report; the
    per-domain bookkeeping adds no extra pass, just a per-sample loss and a
    dict update per batch.

    Batched like evaluate() via the same kind of sequential DataLoader (same
    no-padding argument), but needs a per-sample loss to split by domain, so
    it always computes cross-entropy with reduction='none' internally
    regardless of loss_fn's own reduction (loss_fn is only used by
    evaluate() for the aggregate case).

    Keyed by domain name (ds.domain_names[domain_id]) when ds carries one,
    falling back to str(domain_id) otherwise -- matches DiversityTracker's
    domain_ratio/{name} convention so the two line up in W&B.

    Returns ((agg_avg_loss, agg_ppl), {domain_name: (avg_loss, ppl)}) -- the
    aggregate is the exact same token-weighted average evaluate() computes
    over the same ds, just derived from the same per-domain sums instead of
    a second pass.
    """
    model.eval()
    domain_loss: Dict[int, float] = defaultdict(float)
    domain_tok: Dict[int, int] = defaultdict(int)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=(cfg.device != "cpu"),
    )
    with torch.no_grad(), autocast_ctx(cfg.device):
        for X, Y, domains in loader:
            X = X.to(cfg.device, non_blocking=True)
            Y = Y.to(cfg.device, non_blocking=True)
            logits = model(X)
            B, L, V = logits.shape
            token_loss = F.cross_entropy(
                logits.view(B * L, V), Y.view(B * L), reduction="none"
            ).view(B, L)
            per_sample_loss = token_loss.sum(dim=1)  # [B], matches n_tok weighting below
            for b, domain in enumerate(domains.tolist()):
                domain_loss[domain] += per_sample_loss[b].item()
                domain_tok[domain] += L
    model.train()

    domain_names = getattr(ds, "domain_names", None)

    def label(domain_id: int) -> str:
        if domain_names is not None and 0 <= domain_id < len(domain_names):
            return domain_names[domain_id]
        return str(domain_id)

    results = {}
    total_loss, total_tok = 0.0, 0
    for domain_id, loss_sum in domain_loss.items():
        tok = domain_tok[domain_id]
        avg_loss = loss_sum / max(1, tok)
        results[label(domain_id)] = (avg_loss, math.exp(avg_loss))
        total_loss += loss_sum
        total_tok += tok
    agg_avg_loss = total_loss / max(1, total_tok)
    return (agg_avg_loss, math.exp(agg_avg_loss)), results


def train_baseline(
    cfg: Config,
    model: TinyGPT,
    train_ds: TokenizedCorpus,
    val_ds: TokenizedCorpus,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> TinyGPT:
    """
    Train TinyGPT with uniform random batch selection (no curriculum).

    At each step, draws cfg.per_rank_batch_size samples uniformly at random
    from a pool of cfg.per_rank_pool_size candidates (cfg.pool = pool_mult ×
    global_batch_size, split evenly across ranks same as global_batch_size
    itself, preserving the pool_mult ratio at any world_size). This is the
    control condition — it sets the performance floor that the router should
    beat.
    """

    if cfg.use_wandb and cfg.rank == 0:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = vars(cfg),
            name = cfg.experiment_name,
        )
        if cfg.config_path:
            wandb.save(cfg.config_path, policy="now")

        print("WandB initialized for baseline training.")

    model.to(cfg.device)
    model.train()
    # DDP-replicated or FSDP2-sharded per cfg.distributed. This path matters
    # for FSDP as much as the router one does: random_pool_baseline and the
    # other non-router controls train the same LM, so leaving them
    # DDP-only would OOM exactly where the router run now fits.
    model = wrap_model(model, cfg)
    eval_model, this_rank_evaluates = eval_handles(model, cfg)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.0)

    global_step = 0
    total_tokens_seen = 0
    baseline_loader = make_baseline_loader(
        train_ds, cfg.per_rank_pool_size, cfg.per_rank_batch_size,
        num_workers=cfg.dataloader_num_workers, pin_memory=(cfg.device != "cpu"),
        rank=cfg.rank, world_size=cfg.world_size, seed=cfg.seed,
    )
    budget_reached = False
    for epoch in range(cfg.epochs):
        # See PooledBatchSampler.set_epoch(): required under DDP so pools
        # reshuffle across epochs (its RNG is local, not the global `random`
        # module, so it has no other source of cross-epoch variation).
        baseline_loader.batch_sampler.set_epoch(epoch)

        for selected_idx, X, Y, diffs in tqdm(baseline_loader, disable=(cfg.rank != 0)):
            selected_indices = selected_idx.tolist()
            diffs = diffs.tolist()
            X = X.to(cfg.device, non_blocking=True)
            Y = Y.to(cfg.device, non_blocking=True)
            total_tokens_seen += X.numel() * cfg.world_size
            # Deterministic on every rank (fixed per-step increment, no data
            # dependence), so checking/breaking here is DDP-safe without a
            # broadcast -- every rank reaches the same verdict at the same point.
            budget_reached = cfg.max_tokens is not None and total_tokens_seen >= cfg.max_tokens

            opt.zero_grad()
            with autocast_ctx(cfg.device):
                logits = model(X)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                )
            loss.backward()
            opt.step()

            # Per-sample loss for diversity's train_ppl_domain/* tracking --
            # reuses this step's already-computed logits (detached, no extra
            # forward pass), just a differently-reduced view of the same CE.
            with torch.no_grad():
                B, L, V = logits.shape
                per_sample_loss = F.cross_entropy(
                    logits.detach().view(B * L, V), Y.view(B * L), reduction="none"
                ).view(B, L).mean(dim=1)
            diversity.update(selected_indices, diffs, per_sample_loss.tolist())

            global_step += 1
            if global_step % cfg.log_every == 0:
                # loss is this rank's own local-shard batch loss; average
                # across ranks before logging so world_size>1 runs plot the
                # whole step's loss, not just rank 0's 1/world_size slice.
                # Every rank hits this collective in lockstep since the
                # DDP-sharded loader gives every rank the same step count
                # per epoch (see PooledBatchSampler) -- only rank 0 then
                # actually writes to wandb/prints below.
                log_loss = loss.detach()
                if cfg.world_size > 1:
                    log_loss = log_loss.clone()
                    dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                    log_loss /= cfg.world_size

                # get_metrics() itself issues collectives (all_reduce/
                # all_gather_object) when world_size > 1, so every rank must
                # call it here, not just rank 0.
                div_metrics = diversity.get_metrics(world_size=cfg.world_size)

                if cfg.rank == 0:
                    metrics.log(
                        epoch=epoch,
                        step=global_step,
                        loss_lm=log_loss.item(),
                        entropy=math.log(cfg.per_rank_batch_size),
                        tokens_seen=total_tokens_seen,
                        **div_metrics,
                    )

                    print(f"[Baseline] Step {global_step} - loss_lm={log_loss.item():.4f}")

            if budget_reached:
                break

        # Collective (all_gather_object under world_size>1) -- every rank must
        # reach this the same number of times, so it's called here, before the
        # this_rank_evaluates gate below (which excludes non-zero DDP ranks).
        train_ppl_domain = diversity.get_train_ppl_domain(world_size=cfg.world_size)

        # DDP: rank 0 alone evaluates the unwrapped replica, others wait at
        # the barrier below so nobody starts the next epoch's synchronizing
        # .backward() mid-eval. FSDP: every rank must join the sharded
        # forward's all-gathers, and all compute the same number, so only
        # rank 0 logs. See utils/distributed_utils.eval_handles().
        if this_rank_evaluates:
            (val_loss, val_ppl), per_domain_ppl = evaluate_per_domain(eval_model, val_ds, loss_fn, cfg)
            if cfg.rank == 0:
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    val_ppl=val_ppl,
                    **{f"val_ppl_domain/{name}": ppl for name, (_, ppl) in per_domain_ppl.items()},
                    **train_ppl_domain,
                )
                print(
                    f"[Baseline] Epoch {epoch+1}/{cfg.epochs} "
                    f"- val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}"
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

    return model