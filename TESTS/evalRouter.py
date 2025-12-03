# eval_curriculum_clean.py
# Evaluation pipeline comparing baseline (uniform) vs router-based curriculum
# Based on working notebook implementation

import math, torch, random, json
from torch import nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
from collections import defaultdict

# -------------------------
# Config
# -------------------------
class Config:
    # Data
    block = 256
    
    # Training
    batch = 16           # final selected batch size (k)
    pool_mult = 3        # candidate pool multiplier -> M = pool_mult * batch
    epochs = 3
    lr_lm = 3e-4
    lr_router = 1e-3
    temp = 2.0
    lambda_ent = 1e-2
    lambda_router = 0.1
    
    # Model
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    
    # Eval
    save_dir = "results"
    log_every = 100

cfg = Config()
cfg.pool = cfg.pool_mult * cfg.batch

# Set seed
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# Tokenizer
tok = GPT2TokenizerFast.from_pretrained("gpt2")
if tok.pad_token is None: 
    tok.pad_token = tok.eos_token
vocab_size = len(tok)

# -------------------------
# Data
# -------------------------
def make_chunks(split):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = tok.eos_token.join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    L = (len(ids) // (cfg.block + 1)) * (cfg.block + 1)
    ids = ids[:L]
    chunks = [ids[i:i+cfg.block+1] for i in range(0, L, cfg.block+1)]
    return chunks

class LMDataset(Dataset):
    def __init__(self, chunks):
        self.x = [torch.tensor(c[:-1], dtype=torch.long) for c in chunks]
        self.y = [torch.tensor(c[1:],  dtype=torch.long) for c in chunks]
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        return self.x[i], self.y[i]

def make_index_loader(ds_len, pool_size):
    order = list(range(ds_len))
    random.shuffle(order)
    for i in range(0, ds_len, pool_size):
        yield order[i:i+pool_size]

# -------------------------
# Model
# -------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model=256, n_layers=6, n_heads=8, d_ff=1024, block=256):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(block, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, batch_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        # tie weights
        self.lm_head.weight = self.tok_emb.weight
        
    def _causal_mask(self, L):
        m = torch.ones(L, L, dtype=torch.bool, device=self.lm_head.weight.device).triu(1)
        return m

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = self._causal_mask(L)
        h = self.tr(h, mask=mask)
        return self.lm_head(h)

# -------------------------
# Router
# -------------------------
class AttentionRouter(nn.Module):
    def __init__(self, d_model=256, d_k=64):
        super().__init__()
        self.K = nn.Linear(d_model, d_k, bias=False)
        self.q = nn.Parameter(torch.randn(d_k))
        nn.init.normal_(self.q, std=0.02)
        
    def forward(self, feats):  # [M, d_model]
        K = self.K(feats)      # [M, d_k]
        scores = K @ self.q    # [M]
        return scores

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, ds, loss_fn):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(len(ds)):
        x, y = ds[i]
        x = x.unsqueeze(0).to(model.lm_head.weight.device)
        y = y.unsqueeze(0).to(model.lm_head.weight.device)
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()
    model.train()
    avg_loss = total_loss / max(total_tok, 1)
    return avg_loss, math.exp(avg_loss)

# -------------------------
# Metrics tracker
# -------------------------
class MetricsTracker:
    def __init__(self):
        self.history = defaultdict(list)
        
    def log(self, epoch, step, **kwargs):
        self.history['epoch'].append(epoch)
        self.history['step'].append(step)
        for k, v in kwargs.items():
            self.history[k].append(v)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
    
    def get_final_ppl(self):
        if 'val_ppl' in self.history and len(self.history['val_ppl']) > 0:
            return self.history['val_ppl'][-1]
        return float('inf')

# -------------------------
# Diversity tracker
# -------------------------
class DiversityTracker:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.selection_counts = torch.zeros(dataset_size)
        self.step_selections = []
        
    def update(self, selected_indices):
        self.step_selections.append(set(selected_indices))
        for idx in selected_indices:
            if 0 <= idx < self.dataset_size:
                self.selection_counts[idx] += 1
    
    def get_metrics(self):
        nonzero = self.selection_counts[self.selection_counts > 0]
        coverage = (self.selection_counts > 0).float().mean().item()
        balance_std = nonzero.std().item() if len(nonzero) > 0 else 0
        
        # Unique samples in last 100 steps
        recent = self.step_selections[-100:] if len(self.step_selections) >= 100 else self.step_selections
        if recent:
            unique_recent = len(set.union(*recent))
            total_selections = sum(len(s) for s in recent)
            unique_ratio = unique_recent / max(total_selections, 1)
        else:
            unique_ratio = 0
            
        return {
            'coverage': coverage,
            'balance_std': balance_std,
            'unique_ratio': unique_ratio
        }

# -------------------------
# Training: REINFORCE (router-based)
# -------------------------
def train_router(train_ds, val_ds, metrics, diversity):
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    router = AttentionRouter(d_model=model.tok_emb.embedding_dim, d_k=64).to(cfg.device)
    
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
            xs, ys = zip(*[train_ds[i] for i in pool_indices])
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            M = X.size(0)
            
            # Features
            with torch.no_grad():
                emb = model.tok_emb(X) 
                feats = emb.mean(dim=1)  # [M, d_model]
                            
            # Router scoring
            scores = router(feats)
            probs = torch.softmax(scores / cfg.temp, dim=0)
            
            # Hard select top-k
            k = cfg.batch
            topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            # Selected batch
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # LM forward
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            # Router loss (REINFORCE)
            with torch.no_grad():
                per_sample_loss = nn.functional.cross_entropy(
                    logits.detach().reshape(-1, logits.size(-1)),
                    Y_sel.reshape(-1),
                    reduction='none'
                ).reshape(k, -1).mean(dim=1)
                baseline = per_sample_loss.mean()
            
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((per_sample_loss - baseline) * torch.log(sel_probs)).mean()
            
            # Entropy
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            loss_router = reinforce + cfg.lambda_ent * ent
            
            # Combined step
            opt_lm.zero_grad(set_to_none=True)
            opt_router.zero_grad(set_to_none=True)
            (loss_lm + cfg.lambda_router * loss_router).backward()
            opt_lm.step()
            opt_router.step()
            
            # Track diversity
            selected_indices = [pool_indices[i] for i in topk_idx.cpu().tolist()]
            diversity.update(selected_indices)
            
            # Log
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router_reinforce=reinforce.item(),
                    entropy=-ent.item(),
                    **div_metrics
                )
                print(f"[Router] epoch {epoch} step {global_step}: "
                      f"LM={loss_lm.item():.4f}  "
                      f"Reinf={reinforce.item():.4f}  "
                      f"H={-ent.item():.3f}  "
                      f"Cov={div_metrics['coverage']:.3f}")
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Router] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    return model, router

# -------------------------
# Training: Baseline (uniform)
# -------------------------
def train_baseline(train_ds, val_ds, metrics, diversity):
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        idx_loader = make_index_loader(len(train_ds), cfg.pool)
        
        for pool_indices in idx_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            # Uniform random selection from pool
            selected_indices = random.sample(pool_indices, cfg.batch)
            
            # Load batch
            xs, ys = zip(*[train_ds[i] for i in selected_indices])
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            
            # Forward
            logits = model(X)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            
            # Backward
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            # Track diversity
            diversity.update(selected_indices)
            
            # Log
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss.item(),
                    entropy=math.log(cfg.batch),  # max entropy
                    **div_metrics
                )
                print(f"[Baseline] epoch {epoch} step {global_step}: "
                      f"LM={loss.item():.4f}  "
                      f"Cov={div_metrics['coverage']:.3f}")
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Baseline] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    return model

# -------------------------
# Comparison
# -------------------------
def compare_runs(baseline_metrics, router_metrics):
    print("\n" + "="*70)
    print("COMPARISON REPORT")
    print("="*70)
    
    # Final PPL
    baseline_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()
    improvement = ((baseline_ppl - router_ppl) / baseline_ppl) * 100
    
    print(f"\n1. Final Validation Perplexity:")
    print(f"   Baseline: {baseline_ppl:.2f}")
    print(f"   Router:   {router_ppl:.2f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    # PPL trajectory (find when each reaches thresholds)
    def steps_to_ppl(metrics, target):
        for i, ppl in enumerate(metrics.history.get('val_ppl', [])):
            if ppl <= target:
                return metrics.history['step'][i]
        return None
    
    target = 10000
    baseline_steps = steps_to_ppl(baseline_metrics, target)
    router_steps = steps_to_ppl(router_metrics, target)
    
    print(f"\n2. Steps to reach PPL ≤ {target}:")
    print(f"   Baseline: {baseline_steps if baseline_steps else 'Not reached'}")
    print(f"   Router:   {router_steps if router_steps else 'Not reached'}")
    if baseline_steps and router_steps:
        speedup = ((baseline_steps - router_steps) / baseline_steps) * 100
        print(f"   Speedup: {speedup:+.2f}%")
    
    # Sample diversity
    def avg_last(metrics, key, n=5):
        vals = metrics.history.get(key, [])
        if len(vals) >= n:
            return sum(vals[-n:]) / n
        return 0
    
    print(f"\n3. Sample Diversity (avg last 5 checkpoints):")
    print(f"   Baseline Coverage: {avg_last(baseline_metrics, 'coverage'):.3f}")
    print(f"   Router Coverage:   {avg_last(router_metrics, 'coverage'):.3f}")
    print(f"   Baseline Entropy:  {avg_last(baseline_metrics, 'entropy'):.3f}")
    print(f"   Router Entropy:    {avg_last(router_metrics, 'entropy'):.3f}")
    
    # Verdict
    print(f"\n4. Verdict:")
    if improvement > 5 and router_ppl < baseline_ppl:
        print("   ✓✓ Router provides significant improvement (>5%)")
    elif improvement > 2 and router_ppl < baseline_ppl:
        print("   ✓ Router shows meaningful improvement (2-5%)")
    elif improvement > 0:
        print("   ~ Router shows modest improvement (<2%)")
    else:
        print("   ✗ Router does not improve over baseline")
    
    print("\n" + "="*70 + "\n")

# -------------------------
# Main
# -------------------------
def main():
    print("Loading WikiText-2...")
    train_chunks = make_chunks("train")
    val_chunks = make_chunks("validation")
    train_ds = LMDataset(train_chunks)
    val_ds = LMDataset(val_chunks)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Baseline
    print("="*70)
    print("BASELINE (Uniform Random Selection)")
    print("="*70)
    baseline_metrics = MetricsTracker()
    baseline_diversity = DiversityTracker(len(train_ds))
    baseline_model = train_baseline(train_ds, val_ds, baseline_metrics, baseline_diversity)
    baseline_metrics.save(f"{cfg.save_dir}/baseline_metrics.json")
    
    # Reset seed for fair comparison
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Router
    print("\n" + "="*70)
    print("ROUTER (Attention-based Curriculum)")
    print("="*70)
    router_metrics = MetricsTracker()
    router_diversity = DiversityTracker(len(train_ds))
    router_model, router_net = train_router(train_ds, val_ds, router_metrics, router_diversity)
    router_metrics.save(f"{cfg.save_dir}/router_metrics.json")
    
    # Compare
    compare_runs(baseline_metrics, router_metrics)
    
    print(f"✓ Results saved to {cfg.save_dir}/")

if __name__ == "__main__":
    main()