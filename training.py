# training.py
"""
Reference training loops used for the baseline comparison.

  train_baseline() — uniform random batch selection, standard cross-entropy SGD.
                     No router. The performance floor every other method must beat.
  train_router()   — basic RL curriculum learning: REINFORCE with loss_improvement
                     reward, fixed temperature, top-k selection, and Shannon entropy
                     regularisation. This is the simplified reference router loop.

For the full experiment-grade loop with configurable algorithms (REINFORCE/GRPO/PPO),
reward signals, entropy formulations, and feature caching, see rl_training.py.

evaluate() is shared by both loops above and by rl_training.py.

Entry points that call this module:
  compare.py    — runs baseline + router side-by-side
  smoke_test.py — verifies all code paths via tiny smoke tests
"""
from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from tqdm import tqdm
from time import perf_counter
from config import Config
from data import make_index_loader, TokenizedCorpus
from models.model import TinyGPT, AttentionRouter, extract_hierarchical_features
from utils.metrics import MetricsTracker, DiversityTracker
from GhostSuite.ghostEngines.engine_manager import GhostEngineManager

@contextmanager
def _timed(stage_times: dict, name: str, device: str, active: bool):
    """
    Accumulate wall-clock time for a named stage into stage_times[name].

    No-ops (near-zero overhead) unless `active`, so callers should only pass
    active=True on the same steps they're about to log — a full CUDA sync on
    every step would itself distort the measurements it's trying to take.
    """
    if not active:
        yield
        return
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t0 = perf_counter()
    yield
    if is_cuda:
        torch.cuda.synchronize()
    stage_times[name] = stage_times.get(name, 0.0) + (perf_counter() - t0)


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
    ds: TokenizedCorpus,
    loss_fn: nn.Module,
    cfg: Config,
) -> Tuple[float, float]:
    """
    Compute mean cross-entropy loss and perplexity over the full dataset.

    Iterates sample-by-sample (not batched) to avoid padding artefacts.
    Sets model to eval() before the loop and restores train() after.

    Returns (avg_loss, perplexity) where perplexity = exp(avg_loss).
    """
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


def evaluate_per_domain(
    model: TinyGPT,
    ds: TokenizedCorpus,
    loss_fn: nn.Module,
    cfg: Config,
) -> Dict[str, Tuple[float, float]]:
    """
    Like evaluate(), but broken out per domain: one (avg_loss, perplexity)
    pair per distinct domain in ds, computed in a single pass over ds.

    Meant for a one-off "final perplexity per domain" report on the fully
    trained model (e.g. at the end of training), not per-epoch logging --
    call evaluate() for the cheap aggregate val_loss/val_ppl tracked every
    epoch instead.

    Keyed by domain name (ds.domain_names[domain_id]) when ds carries one,
    falling back to str(domain_id) otherwise -- matches DiversityTracker's
    domain_ratio/{name} convention so the two line up in W&B.
    """
    model.eval()
    domain_loss: Dict[int, float] = defaultdict(float)
    domain_tok: Dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for i in range(len(ds)):
            x, y, domain = ds[i]
            x = x.unsqueeze(0).to(cfg.device)
            y = y.unsqueeze(0).to(cfg.device)
            logits = model(x)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            n_tok = y.numel()
            domain_loss[domain] += loss.item() * n_tok
            domain_tok[domain] += n_tok
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

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)

    global_step = 0
    total_tokens_seen = 0
    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in tqdm(idx_loader, disable=(cfg.rank != 0)):
            if len(pool_indices) < cfg.batch:
                continue

            selected_indices = random.sample(pool_indices, cfg.batch)
            batch = [train_ds[i] for i in selected_indices]
            xs, ys, diffs = zip(*batch)

            X = torch.stack(xs).to(cfg.device)
            Y = torch.stack(ys).to(cfg.device)
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

    # wandb.finish() is deferred to the caller (utils/experiment_worker.py),
    # which logs a couple more summary metrics (e.g. total_time_s) into this
    # same run before closing it.

    return model


def train_router(
    cfg: Config,
    model: TinyGPT,
    router: AttentionRouter,
    train_ds: TokenizedCorpus,
    val_ds: TokenizedCorpus,
    tokenizer,
    metrics: MetricsTracker,
    diversity: DiversityTracker,
) -> Tuple[TinyGPT, AttentionRouter]:
    """
    Train TinyGPT with a basic RL curriculum learning router.

    Per step:
      1. Extract hierarchical features for the full pool (one transformer
         forward pass over M samples — the main cost per step).
      2. Router scores → softmax → top-k selection of cfg.batch samples.
      3. LM forward+backward on selected batch.
      4. Reward = (loss_before - loss_after).clamp(0) per sample.
      5. REINFORCE update: minimise -(advantage * log_prob) + entropy_term.

    This is the simplified reference loop. For ablatable algorithms and
    reward signals, see rl_training.py::train_router_experiments().
    """

    if cfg.use_wandb:
        import wandb
        wandb.init(
            project = cfg.wandb_project,
            entity = cfg.wandb_entity,
            config = vars(cfg),
            name = f"{cfg.experiment_name}_router"
        )
        if cfg.config_path:
            wandb.save(cfg.config_path, policy="now")

    model.to(cfg.device)
    router.to(cfg.device)
    model.train()
    router.train()

    loss_fn = nn.CrossEntropyLoss()
    opt_lm = torch.optim.Adam(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.Adam(router.parameters(), lr=cfg.lr_router)

    total_steps = max(1, (len(train_ds) // cfg.pool) * cfg.epochs)
    global_step = 0
    stage_times: dict = {}
    
    ghost_engine = None
    if cfg.reward_signal == "greats_score":
        val_idx = random.sample(range(len(val_ds)), cfg.greats_val_batch_size)
        Xv, Yv, _ = zip(*(val_ds[i] for i in val_idx))
        X_val = torch.stack(Xv).to(cfg.device)
        Y_val = torch.stack(Yv).to(cfg.device)
        
        print(f"{X_val.shape=}, {Y_val.shape=}")

        ghost_engine = GhostEngineManager(
            config=SimpleNamespace(
                method="GradDotProd",
                result_dir=os.path.join(cfg.save_dir, "ghost"),
                val_batch_size=cfg.greats_val_batch_size,
                log_grad_norms=cfg.greats_log_grad_norms,
                score_exclude_params=cfg.greats_score_exclude_params,
                # Eager engine only: the decoupled/compiled fast path hardcodes
                # GPT-2/nanoGPT-shaped model.transformer.h + forward(idx, idx)->.loss,
                # which doesn't match build_model()'s HFCausalLM/TinyGPT forward signature.
                decoupled_fn=False,
                separate_val=False,
            ),
            model=model,
            optimizer=opt_lm,
            ddp_info={"master_process": cfg.rank == 0},
            val_data=(X_val, Y_val),
        )
    greats_baseline = None  # EMA baseline for the GREATS-sum reward (see router_update below)

    for epoch in range(cfg.epochs):
        idx_loader = make_index_loader(len(train_ds), cfg.pool)

        for pool_indices in tqdm(idx_loader):
            if len(pool_indices) < cfg.batch:
                continue

            # Time this step iff it's the one about to hit the log_every
            # print/log below — see _timed()'s docstring for why.
            profile_step = ((global_step + 1) % cfg.log_every == 0)

            batch = [train_ds[i] for i in pool_indices] 
            xs, ys, diffs = zip(*batch)

            easy = 0
            difficult = 0 
            for b in diffs:
                if b == 0:
                    easy+=1
                else:
                    difficult+=1

            X = torch.stack(xs).to(cfg.device)  # [M, L]
            Y = torch.stack(ys).to(cfg.device)  # [M, L]
            M = X.size(0)

            with _timed(stage_times, "feature_extraction", cfg.device, profile_step):
                feats = extract_hierarchical_features(
                    model=model,
                    X=X,
                    cfg=cfg,
                    pad_token_id=tokenizer.pad_token_id,
                    vocab_size=len(tokenizer),
                )  # [M, d_in]

                # Append pre-computed external embeddings when available.
                if train_ds.embeddings is not None:
                    pool_embs = torch.stack(
                        [train_ds.embeddings[i] for i in pool_indices]
                    ).to(cfg.device)
                    feats = torch.cat([feats, pool_embs], dim=1)

            with _timed(stage_times, "router_select", cfg.device, profile_step):
                scores = router(feats)  # [M]
                probs = torch.softmax(scores / cfg.temp, dim=0)  # [M]

                topk = torch.topk(probs, k=cfg.batch)
                sel_idx_local = topk.indices
                sel_probs = probs[sel_idx_local].clamp_min(1e-12)

                X_sel = X[sel_idx_local]
                Y_sel = Y[sel_idx_local]
                selected_diffs = [diffs[i] for i in sel_idx_local.tolist()]
                selected_indices = [pool_indices[i] for i in sel_idx_local.tolist()]


            # GREATS ghost-gradient scoring: score the router's SELECTED batch against the
            # fixed validation batch (a separate scoring backward, discarded afterwards — the
            # real LM update below is an ordinary forward/backward on the same X_sel/Y_sel).
            ghost_scores_sel = None
            if ghost_engine is not None:
                with _timed(stage_times, "greats_scoring", cfg.device, profile_step):
                    ghost_engine.begin_step()
                    ghost_engine.attach_train_batch(X_sel, Y_sel, global_step)
                    with ghost_engine.saved_tensors_context():
                        Xf, Yf = ghost_engine.prepare_forward_input(X_sel, Y_sel)
                        logits_score = model(Xf)
                        loss_score = loss_fn(
                            logits_score.view(-1, logits_score.size(-1)),
                            Yf.view(-1),
                        )
                        loss_score.backward()
                    ghost_engine.collect_microbatch()
                    ghost_scores_sel = ghost_engine.read_scores(
                        metric=cfg.greats_score_metric
                    ).to(cfg.device)
                    ghost_engine.discard_scores()
            else:
                with _timed(stage_times, "loss_before", cfg.device, profile_step):
                    loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)

            with _timed(stage_times, "lm_forward_backward", cfg.device, profile_step):
                # Clears the scoring pass's leftover grads (if any) as well as zeroing for
                # this real step.
                opt_lm.zero_grad()
                logits_sel = model(X_sel)
                loss_lm = loss_fn(
                    logits_sel.view(-1, logits_sel.size(-1)),
                    Y_sel.view(-1),
                )
                loss_lm.backward()
                opt_lm.step()

            if ghost_engine is None:
                with _timed(stage_times, "loss_after", cfg.device, profile_step):
                    loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)

            with _timed(stage_times, "router_update", cfg.device, profile_step):
                if ghost_scores_sel is not None:
                    # Reward = sum of the GREATS scores of the samples the router chose: a
                    # single scalar shared by every selected sample (the "action" is the joint
                    # selection). Baselined with an EMA across steps, since a per-step batch-mean
                    # baseline would always cancel a scalar reward to zero.
                    reward = ghost_scores_sel.sum()
                    if greats_baseline is None:
                        greats_baseline = reward.detach()
                    else:
                        m = cfg.baseline_momentum
                        greats_baseline = m * greats_baseline + (1 - m) * reward.detach()
                    advantage = reward - greats_baseline
                    improvement_metric = reward.detach()
                else:
                    improvement = (loss_before - loss_after).clamp(min=0.0)
                    advantage = improvement - improvement.mean().detach()
                    improvement_metric = improvement.mean().detach()

                reinforce = -(advantage * sel_probs.log()).mean()
                ent = (probs * probs.clamp_min(1e-12).log()).sum()
                # ent = sum(p * log p) = -H(p), the *negative* Shannon entropy.
                # Adding lambda_ent * ent to the loss penalises low-entropy distributions,
                # so minimising the total loss pushes the router toward diverse selection.

                loss_router = reinforce + cfg.lambda_ent * ent

                opt_router.zero_grad()
                loss_router.backward()
                opt_router.step()

            with _timed(stage_times, "diversity_update", cfg.device, profile_step):
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
                    loss_router=loss_router.item(),
                    reinforce=reinforce.item(),
                    entropy=-ent.item(),
                    avg_improvement=improvement_metric.item(),
                    curriculum_strength=curriculum_strength,
                    **{f"time/{k}_ms": v * 1000 for k, v in stage_times.items()},
                    **div_metrics,
                )

                total_t = sum(stage_times.values()) or 1e-12
                breakdown = "  ".join(
                    f"{k}={v / total_t * 100:.0f}%({v * 1000:.0f}ms)"
                    for k, v in sorted(stage_times.items(), key=lambda kv: -kv[1])
                )
                print(f"[Router] Step {global_step} - loss_lm={loss_lm.item():.4f}, loss_router={loss_router.item():.4f}")
                print(f"[Router] Step {global_step} timing (1 step, sums to {total_t*1000:.0f}ms): {breakdown}")
                stage_times = {}

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
    aux_metrics: Optional[MetricsTracker] = None,
):
    base_ppl   = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()

    if base_ppl is None or router_ppl is None:
        print("Missing val_ppl in metrics; cannot compare.")
        return

    print("\n=== Comparison ===")
    print(f"Baseline       val_ppl: {base_ppl:.1f}")
    print(f"Router         val_ppl: {router_ppl:.1f}  "
          f"({(base_ppl - router_ppl) / base_ppl * 100:.2f}% vs baseline)")

    if aux_metrics is not None:
        aux_ppl = aux_metrics.get_final_ppl()
        if aux_ppl is not None:
            print(f"Aux-net        val_ppl: {aux_ppl:.1f}  "
                  f"({(base_ppl - aux_ppl) / base_ppl * 100:.2f}% vs baseline)")
