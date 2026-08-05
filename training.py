# training.py
"""
Reference training loop used for the baseline comparison.

  train_baseline() — uniform random batch selection, standard cross-entropy SGD.
                     No router. The performance floor every other method must beat.

For the full experiment-grade loop with configurable algorithms (REINFORCE/GRPO/PPO),
reward signals, entropy formulations, and feature caching, see rl_training.py.

evaluate() is shared by train_baseline() and rl_training.py.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tqdm import tqdm
from config import Config
from data import make_baseline_loader, TokenizedCorpus
from models.model import TinyGPT
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
    batch_size is independent of cfg.batch: no gradients are held here, so it
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
    with torch.no_grad():
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
) -> Dict[str, Tuple[float, float]]:
    """
    Like evaluate(), but broken out per domain: one (avg_loss, perplexity)
    pair per distinct domain in ds, computed in a single pass over ds.

    Meant for a one-off "final perplexity per domain" report on the fully
    trained model (e.g. at the end of training), not per-epoch logging --
    call evaluate() for the cheap aggregate val_loss/val_ppl tracked every
    epoch instead.

    Batched like evaluate() via the same kind of sequential DataLoader (same
    no-padding argument), but needs a per-sample loss to split by domain, so
    it always computes cross-entropy with reduction='none' internally
    regardless of loss_fn's own reduction (loss_fn is only used by
    evaluate() for the aggregate case).

    Keyed by domain name (ds.domain_names[domain_id]) when ds carries one,
    falling back to str(domain_id) otherwise -- matches DiversityTracker's
    domain_ratio/{name} convention so the two line up in W&B.
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
    with torch.no_grad():
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
    for domain_id, loss_sum in domain_loss.items():
        avg_loss = loss_sum / max(1, domain_tok[domain_id])
        results[label(domain_id)] = (avg_loss, math.exp(avg_loss))
    return results


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

    At each step, draws cfg.batch samples uniformly at random from a pool
    of cfg.pool candidates (pool_mult × batch). This is the control condition —
    it sets the performance floor that the router should beat.
    """

    if cfg.use_wandb and cfg.rank == 0:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = vars(cfg),
            name = f"{cfg.experiment_name}_baseline",
        )
        if cfg.config_path:
            wandb.save(cfg.config_path, policy="now")

        print("WandB initialized for baseline training.")

    model.to(cfg.device)
    model.train()
    if cfg.world_size > 1:
        # device_ids must be None for CPU modules (only single/multi-GPU
        # modules accept it) -- only relevant for the gloo/CPU smoke-test
        # path, since real DDP training always runs on CUDA.
        ddp_device_ids = [cfg.local_rank] if torch.cuda.is_available() else None
        model = DDP(model, device_ids=ddp_device_ids)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)

    global_step = 0
    total_tokens_seen = 0
    baseline_loader = make_baseline_loader(
        train_ds, cfg.pool, cfg.batch,
        num_workers=cfg.dataloader_num_workers, pin_memory=(cfg.device != "cpu"),
        rank=cfg.rank, world_size=cfg.world_size, seed=cfg.seed,
    )
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

            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
            )
            loss.backward()
            opt.step()

            diversity.update(selected_indices, diffs)

            global_step += 1
            if global_step % cfg.log_every == 0 and cfg.rank == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss.item(),
                    entropy=math.log(cfg.batch),
                    tokens_seen=total_tokens_seen,
                    **div_metrics,
                )

                print(f"[Baseline] Step {global_step} - loss_lm={loss.item():.4f}")

        # Only rank 0 evaluates (val_ds is small and identical on every rank);
        # other ranks wait so nobody starts the next epoch's DDP-synchronizing
        # .backward() calls before rank 0 has finished its forward-only pass.
        if cfg.rank == 0:
            val_loss, val_ppl = evaluate(model, val_ds, loss_fn, cfg)
            metrics.log(
                epoch=epoch,
                step=global_step,
                val_loss=val_loss,
                val_ppl=val_ppl,
            )
            print(
                f"[Baseline] Epoch {epoch+1}/{cfg.epochs} "
                f"- val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}"
            )
        if cfg.world_size > 1:
            dist.barrier()

    # --- Final per-domain perplexity, fully trained model ---
    if cfg.rank == 0:
        per_domain_ppl = evaluate_per_domain(model, val_ds, loss_fn, cfg)
        metrics.log(
            step=global_step,
            **{f"val_ppl_domain/{name}": ppl for name, (_, ppl) in per_domain_ppl.items()},
        )
        print(
            "[Final per-domain val perplexity] "
            + ", ".join(f"{name}={ppl:.1f}" for name, (_, ppl) in sorted(per_domain_ppl.items()))
        )

    # wandb.finish() is deferred to the caller (utils/experiment_worker.py),
    # which logs a couple more summary metrics (e.g. total_time_s) into this
    # same run before closing it.

    return model