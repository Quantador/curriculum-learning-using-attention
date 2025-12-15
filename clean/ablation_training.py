# ablation_training.py
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from config import Config
from data import make_index_loader, MixedLMDataset
from model import TinyGPT, extract_router_features, build_router
from metrics import MetricsTracker, DiversityTracker
from training import compute_loss_per_sample, evaluate


def train_router_ablation(
    cfg: Config,
    model: TinyGPT,
    train_ds: MixedLMDataset,
    val_ds: MixedLMDataset,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
    feature_mode: str = "full",          # group 1
    arch: str = "attention",             # group 2
    reward_type: str = "improvement",    # group 3
    selection_type: str = "topk",        # group 4
) -> Tuple[TinyGPT, torch.nn.Module | None]:
    """
    feature_mode:  {"full", "no_hidden", "no_stats", "no_hier", "random"}
    arch:          {"attention", "linear", "mlp", "random"}
    reward_type:   {"improvement", "improvement_no_baseline",
                    "constant", "loss_before", "minus_loss_after"}
    selection_type:{"topk", "sample", "gumbel_topk"}
    """
    device = cfg.device
    model.to(device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    opt_lm = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)

    router = None
    opt_router = None
    d_input = None

    total_steps = max(1, (len(train_ds) // cfg.pool) * cfg.epochs)
    global_step = 0

    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue

            batch = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = zip(*batch)

            X = torch.stack(xs).to(device)  # [M, L]
            Y = torch.stack(ys).to(device)  # [M, L]
            M = X.size(0)

            feats = extract_router_features(
                model=model,
                X=X,
                cfg=cfg,
                pad_token_id=pad_id,
                vocab_size=vocab_size,
                mode=feature_mode,
            )  # [M, d_in]
            if d_input is None:
                d_input = feats.size(1)
                if arch != "random":
                    router = build_router(d_input=d_input, arch=arch)
                    router.to(device)
                    opt_router = torch.optim.Adam(
                        router.parameters(), lr=cfg.lr_router
                    )

            if arch == "random":
                scores = torch.randn(M, device=device)
            else:
                scores = router(feats)  # [M]

            probs = torch.softmax(scores / cfg.temp, dim=0)  # [M]

            if selection_type == "topk":
                sel_idx_local = torch.topk(probs, k=cfg.batch).indices
            elif selection_type == "sample":
                sel_idx_local = torch.multinomial(probs, cfg.batch, replacement=False)
            elif selection_type == "gumbel_topk":
                gumbel = -torch.log(-torch.log(torch.rand_like(probs)))
                noisy_scores = scores + gumbel
                sel_idx_local = torch.topk(noisy_scores, k=cfg.batch).indices
            else:
                raise ValueError(f"Unknown selection_type: {selection_type}")

            sel_probs = probs[sel_idx_local].clamp_min(1e-12)

            X_sel = X[sel_idx_local]
            Y_sel = Y[sel_idx_local]
            selected_diffs = [diffs[i] for i in sel_idx_local.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx_local.tolist()]

            loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)

            opt_lm.zero_grad()
            logits_sel = model(X_sel)
            loss_lm = loss_fn(
                logits_sel.view(-1, logits_sel.size(-1)),
                Y_sel.view(-1),
            )
            loss_lm.backward()
            opt_lm.step()

            loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            improvement = (loss_before - loss_after).clamp(min=0.0)

            if reward_type == "improvement":
                baseline = improvement.mean().detach()
                adv = improvement - baseline
            elif reward_type == "improvement_no_baseline":
                adv = improvement
            elif reward_type == "constant":
                adv = torch.ones_like(improvement)
            elif reward_type == "loss_before":
                adv = loss_before.detach()
            elif reward_type == "minus_loss_after":
                adv = -loss_after.detach()
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")

            reinforce = -(adv * sel_probs.log()).mean()
            ent = (probs * probs.clamp_min(1e-12).log()).sum()
            loss_router = reinforce + cfg.lambda_ent * ent

            if arch != "random":
                opt_router.zero_grad()
                loss_router.backward()
                opt_router.step()

            diversity.update(selected_indices, selected_diffs)

            global_step += 1
            if global_step % cfg.log_every == 0:
                training_progress = global_step / total_steps
                curriculum_strength = 1.0 - training_progress
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router=loss_router.item() if arch != "random" else 0.0,
                    reinforce=reinforce.item(),
                    entropy=-ent.item(),
                    avg_improvement=improvement.mean().item(),
                    curriculum_strength=curriculum_strength,
                    **div_metrics,
                )

        val_loss, val_ppl = evaluate(model, val_ds, loss_fn, cfg)
        metrics.log(
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
        )
        print(
            f"[Ablation-{feature_mode}/{arch}/{reward_type}/{selection_type}] "
            f"Epoch {epoch+1}/{cfg.epochs} "
            f"- val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}"
        )

    return model, router
