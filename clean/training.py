# training.py
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
from torch import mode, nn

from tqdm import tqdm

from config import Config
from data import make_index_loader, MixedLMDataset
from model import TinyGPT, extract_hierarchical_features
from metrics import MetricsTracker, DiversityTracker
import math



def normalize_advantage(x: torch.Tensor, mode: str, eps: float, clip: float | None):
    """
    x: [K] reward-like tensor for selected samples.
    Returns normalized advantage.
    """
    if mode == "none":
        adv = x
    elif mode == "center":
        adv = x - x.mean()
    elif mode == "zscore":
        adv = (x - x.mean()) / (x.std(unbiased=False) + eps)
    else:
        raise ValueError(f"Unknown advantage_norm: {mode}")

    if clip is not None:
        adv = adv.clamp(-clip, clip)
    return adv


def scheduled_value(step: int, start: float, final_mult: float, warmup: int, anneal: int, schedule: str) -> float:
    """
    Returns a value that starts at `start` and anneals to `start * final_mult`.
    - warmup: steps to hold constant at start
    - anneal: steps over which to anneal (after warmup)
    """
    if schedule == "constant" or anneal <= 0:
        return start

    if step < warmup:
        return start

    t = min(1.0, (step - warmup) / float(anneal))
    end = start * final_mult

    if schedule == "linear":
        return start + t * (end - start)
    elif schedule == "exp":
        # exponential interpolation in log-space (requires start > 0, end >= 0)
        # If end == 0, approximate with a small floor to avoid log(0)
        floor = 1e-12
        a = max(start, floor)
        b = max(end, floor)
        return a * ((b / a) ** t)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def novelty_from_counts(counts: torch.Tensor, mode: str) -> torch.Tensor:
    """
    counts: int tensor [K] (how many times each selected dataset index was selected)
    returns: float tensor [K] novelty bonus
    """
    if mode == "inv_sqrt":
        return 1.0 / torch.sqrt(counts.float() + 1.0)
    if mode == "exp":
        return torch.exp(-counts.float())
    if mode == "first_time":
        return (counts == 0).float()
    raise ValueError(f"Unknown novelty_mode: {mode}")



def compute_loss_per_sample(
    model: TinyGPT,
    X: torch.Tensor,
    Y: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    b = X.size(0)
    losses = []
    with torch.no_grad():
        logits = model(X)  # [b, L, V]
    for i in range(b):
        li = loss_fn(
            logits[i].view(-1, logits.size(-1)),
            Y[i].view(-1),
        )
        losses.append(li)
    return torch.stack(losses, dim=0)


def evaluate(
    model: TinyGPT,
    ds: MixedLMDataset,
    loss_fn: nn.Module,
    cfg: Config,
) -> Tuple[float, float]:
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, _ = ds[i]
            x = x.unsqueeze(0).to(cfg.device)
            y = y.unsqueeze(0).to(cfg.device)
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            n_tok = y.numel()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
    model.train()
    avg_loss = total_loss / max(total_tok, 1)
    return avg_loss, math.exp(avg_loss)


def train_baseline(
    cfg: Config,
    model: TinyGPT,
    train_ds: MixedLMDataset,
    val_ds: MixedLMDataset,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> TinyGPT:
    
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = vars(cfg),
            name = "baseline",
        )
        
        print("WandB initialized for baseline training.")
    
    model.to(cfg.device)
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)

    global_step = 0
    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in tqdm(idx_loader):
            if len(pool_indices) < cfg.batch:
                continue

            selected_indices = random.sample(pool_indices, cfg.batch)
            batch = [train_ds[i] for i in selected_indices]
            xs, ys, diffs = zip(*batch)

            X = torch.stack(xs).to(cfg.device)
            Y = torch.stack(ys).to(cfg.device)

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
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss.item(),
                    entropy=math.log(cfg.batch),
                    **div_metrics,
                )
                
                print(f"[Baseline] Step {global_step} - loss_lm={loss.item():.4f}")

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
        
    if cfg.use_wandb:
        wandb.finish()

    return model


def train_router(
    cfg: Config,
    model: TinyGPT,
    router: Optional[nn.Module],
    train_ds: MixedLMDataset,
    val_ds: MixedLMDataset,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> Tuple[TinyGPT, Optional[nn.Module]]:
    
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = vars(cfg),
            name = "router"
        )
    
    model.to(cfg.device)
    if router is not None:
        router.to(cfg.device)
    model.train()
    if router is not None:
        router.train()
    
    loss_fn = nn.CrossEntropyLoss()
    opt_lm = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)
    opt_router = None
    if router is not None:
        opt_router = torch.optim.Adam(router.parameters(), lr=cfg.lr_router)

    total_steps = max(1, (len(train_ds) // cfg.pool) * cfg.epochs)
    sel_counts = torch.zeros(len(train_ds), dtype=torch.int32)

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
            M = X.size(0)

            feats = extract_hierarchical_features(
                model=model,
                X=X,
                cfg=cfg,
                pad_token_id=tokenizer.pad_token_id,
                vocab_size=tokenizer.vocab_size,
                mode=cfg.feature_mode,
            )  # [M, d_in]

            if router is None:
                scores = torch.randn(feats.size(0), device=feats.device)
            else:   
                scores = router(feats)  # [M]
                
            probs = torch.softmax(scores / cfg.temp, dim=0)  # [M]

            topk = torch.topk(probs, k=cfg.batch)
            sel_idx_local = topk.indices
            sel_probs = probs[sel_idx_local].clamp_min(1e-12)

            X_sel = X[sel_idx_local]
            Y_sel = Y[sel_idx_local]
            selected_diffs = [diffs[i] for i in sel_idx_local.tolist()]
            selected_indices = [pool_indices[i] for i in sel_idx_local.tolist()]
            
            # Map pool positions -> global dataset indices
            # pool_indices is usually a Python list of dataset indices
            pool_idx_t = torch.as_tensor(pool_indices, dtype=torch.long)
            sel_dataset_idx = pool_idx_t[sel_idx_local.detach().cpu()]  # [K] on CPU

            # novelty bonus from counts BEFORE increment
            if cfg.novelty_bonus > 0.0:
                counts = sel_counts[sel_dataset_idx]  # [K]
                nov = novelty_from_counts(counts, cfg.novelty_mode)  # [K] float on CPU
            else:
                nov = None

            # update counts AFTER reading
            sel_counts[sel_dataset_idx] += 1


            loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)

            opt_lm.zero_grad()
            logits_sel = model(X_sel)
            loss_lm = loss_fn(
                logits_sel.view(-1, logits_sel.size(-1)),
                Y_sel.view(-1),
            )
            loss_lm.backward()
            opt_lm.step()

            # ---- reward for router update (ablation-ready) ----
            loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)

            if cfg.reward_mode == "progress":
                # Full method: learning progress
                reward = (loss_before - loss_after).clamp(min=0.0).detach()
            elif cfg.reward_mode == "neg_loss":
                # Ablation A5: no progress, just prefer high/low loss depending on sign
                # (reward = -loss_before encourages selecting LOWER-loss samples)
                reward = (-loss_before).detach()
            else:
                raise ValueError(f"Unknown reward_mode: {cfg.reward_mode}")

            if cfg.novelty_bonus > 0.0 and nov is not None:
                eta = scheduled_value(
                    step=global_step,
                    start=cfg.novelty_bonus,
                    final_mult=cfg.novelty_final_mult,
                    warmup=0,
                    anneal=cfg.entropy_anneal_steps if cfg.entropy_anneal_steps > 0 else 0,
                    schedule=cfg.novelty_anneal,
                )
                reward = reward + eta * nov.to(device=reward.device, dtype=reward.dtype)

            # Entropy (note: this is usually negative; keep same convention you had)
            ent = (probs * probs.clamp_min(1e-12).log()).sum()
            
            adv = normalize_advantage(
                reward,
                mode=cfg.advantage_norm,
                eps=cfg.advantage_eps,
                clip=cfg.advantage_clip,
            )
            
            lam_ent = scheduled_value(
                step=global_step,
                start=cfg.lambda_ent,
                final_mult=cfg.entropy_final_mult,
                warmup=cfg.entropy_warmup_steps,
                anneal=cfg.entropy_anneal_steps,
                schedule=cfg.entropy_schedule,
            )

            # REINFORCE loss only if router is learnable
            if router is not None:
                reinforce = -(adv  * sel_probs.log()).mean()
                loss_router = reinforce + lam_ent * ent

                opt_router.zero_grad()
                loss_router.backward()
                opt_router.step()
            else:
                loss_router = torch.tensor(0.0, device=probs.device)
                reinforce = torch.tensor(0.0, device=probs.device)
            improvement = (loss_before - loss_after)


            diversity.update(selected_indices, selected_diffs)

            global_step += 1
            if global_step % cfg.log_every == 0:
                training_progress = global_step / total_steps
                curriculum_strength = 1.0 - training_progress
                div_metrics = diversity.get_metrics()
                if cfg.novelty_bonus > 0.0 and nov is not None:
                    metrics.log(
                        epoch=epoch,
                        step=global_step,
                        loss_lm=loss_lm.item(),
                        loss_router=loss_router.item(),
                        reinforce=reinforce.item(),
                        entropy=-ent.item(),
                        avg_improvement=improvement.mean().item(),
                        curriculum_strength=curriculum_strength,
                        lam_ent=lam_ent,
                        novelty_bonus=eta,
                        novelty_avg=nov.mean().item(),
                        novelty_frac_first=(nov > 0).float().mean().item(),
                        **div_metrics,
                    )
                else:
                    metrics.log(
                        epoch=epoch,
                        step=global_step,
                        loss_lm=loss_lm.item(),
                        loss_router=loss_router.item(),
                        reinforce=reinforce.item(),
                        entropy=-ent.item(),
                        avg_improvement=improvement.mean().item(),
                        curriculum_strength=curriculum_strength,
                        lam_ent=lam_ent,
                        **div_metrics,
                    )
                
                print(f"[Router] Step {global_step} - loss_lm={loss_lm.item():.4f}, loss_router={loss_router.item():.4f}")

        val_loss, val_ppl = evaluate(model, val_ds, loss_fn, cfg)
        metrics.log(
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
        )
        print(
            f"[Router] Epoch {epoch+1}/{cfg.epochs} "
            f"- val_loss={val_loss:.4f}, val_ppl={val_ppl:.1f}"
        )
        
    if cfg.use_wandb:
        wandb.finish()

    return model, router


def compare_runs(
    baseline_metrics: MetricsTracker,
    router_metrics: MetricsTracker,
):
    base_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()

    if base_ppl is None or router_ppl is None:
        print("Missing val_ppl in metrics; cannot compare.")
        return

    improvement = (base_ppl - router_ppl) / base_ppl * 100.0
    print("\n=== Comparison ===")
    print(f"Baseline final val_ppl: {base_ppl:.1f}")
    print(f"Router   final val_ppl: {router_ppl:.1f}")
    print(f"Relative improvement:   {improvement:.2f}%")
