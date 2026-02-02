from __future__ import annotations

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
    Train LM with a router using REINFORCE-based curriculum learning.

    Upgrades implemented:
      1) Remove the redundant LM forward previously used for loss_before:
         - loss_before is computed from the same logits used for the LM update.
      2) Vectorize per-sample CE computation (no Python loop).
    """

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name="router",
        )

    model.to(cfg.device).train()
    router.to(cfg.device).train()

    opt_lm = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.Adam(router.parameters(), lr=cfg.lr_router)

    total_steps = max(1, (len(train_ds) // cfg.pool) * cfg.epochs)
    global_step = 0

    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in tqdm(idx_loader):
            if len(pool_indices) < cfg.batch:
                continue

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
            probs = torch.softmax(scores / cfg.temp, dim=0)  # [M]

            # --- Top-K selection ---
            topk = torch.topk(probs, k=cfg.batch)
            sel_idx = topk.indices
            sel_probs = probs[sel_idx].clamp_min(1e-12)

            X_sel = X[sel_idx]  # [B, L]
            Y_sel = Y[sel_idx]  # [B, L]
            selected_diffs = [diffs[i] for i in sel_idx.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx.tolist()]

            # ============================================================
            # UPGRADE #1 + #2:
            #   Single forward for both:
            #     - loss_before (per-sample, detached)
            #     - loss_lm (scalar, for backward)
            # ============================================================
            opt_lm.zero_grad()

            logits = model(X_sel)  # [B, L, V]  <-- single forward

            # per-sample loss BEFORE update, computed from same logits (detached)
            with torch.no_grad():
                loss_before = compute_loss_per_sample_vectorized(logits, Y_sel)  # [B]

            # scalar loss for LM update
            B, L, V = logits.shape
            loss_lm = F.cross_entropy(
                logits.view(B * L, V),
                Y_sel.view(B * L),
                reduction="mean",
            )

            loss_lm.backward()
            opt_lm.step()

            # loss AFTER update (still needs a forward with new weights)
            with torch.no_grad():
                logits_after = model(X_sel)
                loss_after = compute_loss_per_sample_vectorized(logits_after, Y_sel)  # [B]

            # --- Reward + router update ---
            improvement = (loss_before - loss_after).clamp(min=0.0)
            baseline = improvement.mean().detach()

            reinforce = -((improvement - baseline) * sel_probs.log()).mean()
            entropy = (probs * probs.clamp_min(1e-12).log()).sum()

            loss_router = reinforce + cfg.lambda_ent * entropy

            opt_router.zero_grad()
            loss_router.backward()
            opt_router.step()

            diversity.update(selected_indices, selected_diffs)

            # --- Logging ---
            global_step += 1
            if global_step % cfg.log_every == 0:
                progress = global_step / total_steps
                curriculum_strength = 1.0 - progress

                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router=loss_router.item(),
                    reinforce=reinforce.item(),
                    entropy=-entropy.item(),
                    avg_improvement=improvement.mean().item(),
                    curriculum_strength=curriculum_strength,
                    **diversity.get_metrics(),
                )

                print(
                    f"[Router] Step {global_step} | "
                    f"loss_lm={loss_lm.item():.4f} | "
                    f"loss_router={loss_router.item():.4f}"
                )

        # --- Validation ---
        loss_fn = nn.CrossEntropyLoss()  # evaluate() expects a loss_fn in your codebase
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn, cfg)

        metrics.log(
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
        )

        print(
            f"[Router] Epoch {epoch + 1}/{cfg.epochs} | "
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
