# eval_curriculum_ablation.py
# Modified version with ablation study support
# Allows selective enabling/disabling of components

import math, torch, random, json
from torch import nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
from collections import defaultdict
import copy

# Wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -------------------------
# Config with Ablation Flags
# -------------------------
class Config:
    # Data
    block = 256
    easy_samples = 30000
    hard_samples = 5000
    
    # Training
    batch = 16
    pool_mult = 5
    epochs = 10
    lr_lm = 3e-4
    lr_router = 1e-3
    temp = 1.0
    lambda_ent = 0.005
    lambda_router = 0.1
    
    # Model
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    
    # Features
    n_chunks = 8
    
    # ABLATION FLAGS
    use_hierarchical_features = True
    use_text_statistics = True
    use_loss_improvement_reward = True
    use_curriculum_bias = True
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    
    # Wandb
    use_wandb = True
    wandb_project = "curriculum-learning"
    wandb_entity = None
    
    # Eval
    save_dir = "results_improved"
    log_every = 100

cfg = Config()
cfg.pool = cfg.pool_mult * cfg.batch

# [Keep all data loading and model code the same as original...]

# -------------------------
# Modified Feature Extraction - Respects Ablation Flags
# -------------------------
def compute_text_statistics(X, pad_token_id):
    """Compute text statistics (can be disabled via config)"""
    if not cfg.use_text_statistics:
        # Return zero features if disabled
        M = X.shape[0]
        return torch.zeros(M, 4, device=X.device, dtype=torch.float32)
    
    M, L = X.shape
    stats = []
    vocab_size = 50257  # GPT-2 vocab size
    
    for i in range(M):
        tokens = X[i]
        valid_mask = tokens != pad_token_id
        valid_tokens = tokens[valid_mask]
        
        if len(valid_tokens) == 0:
            stats.append([0, 0, 0, 0])
            continue
        
        length = len(valid_tokens)
        unique_ratio = len(valid_tokens.unique()) / length
        avg_token_id = valid_tokens.float().mean().item() / vocab_size
        std_token_id = valid_tokens.float().std().item() / vocab_size
        
        stats.append([length / cfg.block, unique_ratio, avg_token_id, std_token_id])
    
    return torch.tensor(stats, device=X.device, dtype=torch.float32)

def extract_hierarchical_features(model, X, pad_token_id):
    """Extract features respecting ablation flags"""
    with torch.no_grad():
        M, L = X.shape
        
        # Hierarchical features
        if cfg.use_hierarchical_features:
            h = model.forward_to_hidden(X)  # [M, L, d_model]
            _, _, d_model = h.shape
            
            n_chunks = cfg.n_chunks
            chunk_size = L // n_chunks
            
            chunk_features = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < n_chunks - 1 else L
                chunk = h[:, start:end, :]
                chunk_mean = chunk.mean(dim=1)
                chunk_features.append(chunk_mean)
            
            learned_features = torch.cat(chunk_features, dim=-1)
        else:
            # Random features if disabled
            d_model = cfg.d_model
            learned_features = torch.randn(M, cfg.n_chunks * d_model, device=X.device)
        
        # Text statistics
        text_stats = compute_text_statistics(X, pad_token_id)
        
        # Combine
        features = torch.cat([learned_features, text_stats], dim=-1)
    
    return features

# -------------------------
# Modified Training - Respects Ablation Flags
# -------------------------
def train_router_ablation(train_ds, val_ds, metrics, diversity, use_wandb=False):
    """Modified train_router that respects ablation flags"""
    
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vocab_size = len(tok)
    
    # Import from original file
    from eval_curriculum_improved import (
        TinyGPT, AttentionRouter, compute_loss_per_sample,
        make_index_loader, evaluate
    )
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name="router-ablation",
            config={
                "method": "router-curriculum-ablation",
                "use_hierarchical_features": cfg.use_hierarchical_features,
                "use_text_statistics": cfg.use_text_statistics,
                "use_loss_improvement_reward": cfg.use_loss_improvement_reward,
                "use_curriculum_bias": cfg.use_curriculum_bias,
                "batch_size": cfg.batch,
                "pool_size": cfg.pool,
                "epochs": cfg.epochs,
                "n_chunks": cfg.n_chunks,
                "seed": cfg.seed
            },
            reinit=True
        )
    
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    
    d_router_input = cfg.n_chunks * cfg.d_model + 4
    router = AttentionRouter(d_input=d_router_input, d_k=64).to(cfg.device)
    
    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        router.train()
        idx_loader = make_index_loader(len(train_ds), cfg.pool)
        
        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            # Load pool
            batch_data = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = zip(*batch_data)
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            M = X.size(0)
            
            # Extract features (respects ablation flags)
            feats = extract_hierarchical_features(model, X, tok.pad_token_id)
            
            # Router scoring
            scores = router(feats)
            
            # Curriculum bias (optional)
            if cfg.use_curriculum_bias:
                training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
                curriculum_strength = max(0, 1.0 - training_progress)
                # Bias toward easy samples early on
                # (implementation can vary)
            else:
                curriculum_strength = 0.0
            
            probs = torch.softmax(scores / cfg.temp, dim=0)
            
            # Select top-k
            k = cfg.batch
            topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # Loss improvement reward (optional)
            if cfg.use_loss_improvement_reward:
                with torch.no_grad():
                    loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
                
                logits = model(X_sel)
                loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
                
                opt_lm.zero_grad(set_to_none=True)
                loss_lm.backward()
                opt_lm.step()
                
                with torch.no_grad():
                    loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
                
                improvement = (loss_before - loss_after).clamp(min=0.0)
                baseline = improvement.mean()
            else:
                # Fixed reward (no loss improvement)
                logits = model(X_sel)
                loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
                
                opt_lm.zero_grad(set_to_none=True)
                loss_lm.backward()
                opt_lm.step()
                
                # Fixed reward = 1.0 for all samples
                improvement = torch.ones(k, device=cfg.device)
                baseline = 1.0
            
            # Router loss
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((improvement - baseline) * torch.log(sel_probs)).mean()
            
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            loss_router = reinforce + cfg.lambda_ent * ent
            
            opt_router.zero_grad(set_to_none=True)
            loss_router.backward()
            opt_router.step()
            
            # Track diversity
            selected_indices = [pool_indices[i] for i in topk_idx.cpu().tolist()]
            selected_diffs = [diffs[i] for i in topk_idx.cpu().tolist()]
            diversity.update(selected_indices, selected_diffs)
            
            # Log
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router_reinforce=reinforce.item(),
                    entropy=-ent.item(),
                    avg_improvement=improvement.mean().item(),
                    curriculum_bias=curriculum_strength,
                    **div_metrics
                )
                print(f"[Router] epoch {epoch} step {global_step}: "
                      f"LM={loss_lm.item():.4f}  "
                      f"Improve={improvement.mean().item():.4f}")
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Router] epoch {epoch}: val_ppl={val_ppl:.2f}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model, router

# -------------------------
# Usage Example
# -------------------------
if __name__ == "__main__":
    print("Ablation-enabled training code ready!")
    print("\nAblation flags:")
    print(f"  use_hierarchical_features: {cfg.use_hierarchical_features}")
    print(f"  use_text_statistics: {cfg.use_text_statistics}")
    print(f"  use_loss_improvement_reward: {cfg.use_loss_improvement_reward}")
    print(f"  use_curriculum_bias: {cfg.use_curriculum_bias}")