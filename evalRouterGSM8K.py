# eval_gsm8k.py
# Curriculum learning on GSM8K mathematical reasoning
# Router learns to focus on appropriate difficulty levels over time

import math, torch, random, json, re
from torch import nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
from collections import defaultdict

# Config
class Config:
    # Model
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 512  # Longer for math problems
    
    # Training
    batch = 8           # Smaller batch for longer sequences
    pool_mult = 4       # Larger pool for more diversity
    epochs = 15         # More epochs - math is harder
    lr_lm = 3e-4
    lr_router = 1e-3
    temp = 1.0
    lambda_ent = 2e-2   # Higher entropy reg for more buckets
    lambda_router = 0.1
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    save_dir = "results_gsm8k"
    log_every = 50
    eval_every = 200  # Evaluate more frequently

cfg = Config()
cfg.pool = cfg.pool_mult * cfg.batch

random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

# Tokenizer
tok = GPT2TokenizerFast.from_pretrained("gpt2")
if tok.pad_token is None: 
    tok.pad_token = tok.eos_token
vocab_size = len(tok)

print(f"Vocab size: {vocab_size}")

# -------------------------
# GSM8K Data Processing
# -------------------------

def extract_answer(answer_text):
    """Extract final numerical answer from GSM8K format"""
    # GSM8K format: "explanation here\n#### 42"
    if "####" in answer_text:
        return answer_text.split("####")[1].strip()
    return answer_text.strip()

def count_reasoning_steps(answer_text):
    """
    Count reasoning steps in GSM8K answer.
    Steps are marked by <<calculation>> in the answer.
    """
    return answer_text.count("<<")

def get_difficulty_bucket(num_steps):
    """
    Map reasoning steps to difficulty buckets:
    0: Easy (1-2 steps)
    1: Medium (3-4 steps)
    2: Hard (5-6 steps)
    3: Very Hard (7+ steps)
    """
    if num_steps <= 2:
        return 0
    elif num_steps <= 4:
        return 1
    elif num_steps <= 6:
        return 2
    else:
        return 3

class GSM8KDataset(Dataset):
    """
    GSM8K dataset for mathematical reasoning.
    Format: "Question: {q}\nAnswer: {a}"
    """
    def __init__(self, split="train"):
        print(f"Loading GSM8K {split} split...")
        ds = load_dataset("gsm8k", "main", split=split)
        
        self.samples = []
        self.difficulties = []
        
        for ex in ds:
            question = ex['question']
            answer = ex['answer']
            
            # Format: Question + Answer for next-token prediction
            text = f"Question: {question}\nAnswer: {answer}"
            
            # Tokenize
            tokens = tok(text, truncation=True, max_length=cfg.max_seq_len, 
                        add_special_tokens=True)['input_ids']
            
            if len(tokens) < 10:  # Skip very short samples
                continue
            
            # Count reasoning steps for difficulty
            num_steps = count_reasoning_steps(answer)
            difficulty = get_difficulty_bucket(num_steps)
            
            self.samples.append(tokens)
            self.difficulties.append(difficulty)
        
        # Print difficulty distribution
        diff_counts = defaultdict(int)
        for d in self.difficulties:
            diff_counts[d] += 1
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Difficulty distribution:")
        print(f"  Easy (1-2 steps):     {diff_counts[0]} samples")
        print(f"  Medium (3-4 steps):   {diff_counts[1]} samples")
        print(f"  Hard (5-6 steps):     {diff_counts[2]} samples")
        print(f"  Very Hard (7+ steps): {diff_counts[3]} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        difficulty = self.difficulties[idx]
        
        # Create input (all but last) and target (all but first)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y, difficulty

def collate_fn(batch):
    """Pad sequences to same length in batch"""
    xs, ys, diffs = zip(*batch)
    
    # Find max length in batch
    max_len = max(x.size(0) for x in xs)
    
    # Pad sequences
    xs_padded = []
    ys_padded = []
    
    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), tok.pad_token_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])  # -100 = ignore in loss
        xs_padded.append(x)
        ys_padded.append(y)
    
    return torch.stack(xs_padded), torch.stack(ys_padded), list(diffs)

def make_pool_loader(dataset, pool_size):
    """Yield random pools of samples"""
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for i in range(0, len(dataset), pool_size):
        pool_indices = indices[i:i+pool_size]
        if len(pool_indices) >= cfg.batch:  # Only yield if we have enough
            yield pool_indices

# -------------------------
# Model
# -------------------------
class MathGPT(nn.Module):
    """GPT-style model for mathematical reasoning"""
    def __init__(self, vocab, d_model=512, n_layers=6, n_heads=8, d_ff=2048, 
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=dropout, 
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.tok_emb.weight
        
    def _causal_mask(self, seq_len):
        """Generate causal attention mask"""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, 
                         device=self.tok_emb.weight.device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def forward(self, x):  # x: [B, L]
        B, L = x.shape
        
        # Embeddings
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        
        # Transformer with causal mask
        mask = self._causal_mask(L)
        h = self.transformer(h, mask=mask)
        h = self.ln_f(h)
        
        # LM head
        logits = self.lm_head(h)  # [B, L, V]
        return logits

# -------------------------
# Router
# -------------------------
class DifficultyRouter(nn.Module):
    """
    Router that learns to score samples based on their embeddings.
    Should learn to focus on appropriate difficulty over time.
    """
    def __init__(self, d_model=512, d_k=128):
        super().__init__()
        self.K = nn.Linear(d_model, d_k, bias=False)
        self.q = nn.Parameter(torch.randn(d_k))
        nn.init.normal_(self.q, std=0.02)
        
    def forward(self, feats):  # [M, d_model]
        K = self.K(feats)  # [M, d_k]
        scores = K @ self.q  # [M]
        return scores

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, dataset, max_samples=500):
    """
    Evaluate perplexity on validation set.
    For math, lower perplexity = better at modeling the reasoning.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_loss = 0.0
    total_tokens = 0
    num_samples = min(len(dataset), max_samples)
    
    for i in range(num_samples):
        x, y, _ = dataset[i]
        x = x.unsqueeze(0).to(model.lm_head.weight.device)
        y = y.unsqueeze(0).to(model.lm_head.weight.device)
        
        logits = model(x)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        # Count non-padding tokens
        valid_tokens = (y != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
    
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # Cap to prevent overflow
    
    return avg_loss, ppl

# -------------------------
# Metrics & Tracking
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

class DifficultyTracker:
    """Track which difficulty levels the router selects over time"""
    def __init__(self):
        self.step_difficulties = []
        self.difficulty_counts = defaultdict(int)
    
    def update(self, difficulties):
        """difficulties: list of difficulty levels for selected samples"""
        self.step_difficulties.append(difficulties)
        for d in difficulties:
            self.difficulty_counts[d] += 1
    
    def get_metrics(self):
        """Get distribution of selected difficulties"""
        if not self.step_difficulties:
            return {}
        
        # Last 100 steps
        recent = self.step_difficulties[-100:] if len(self.step_difficulties) >= 100 else self.step_difficulties
        
        # Count difficulty distribution
        diff_dist = defaultdict(int)
        total = 0
        for step_diffs in recent:
            for d in step_diffs:
                diff_dist[d] += 1
                total += 1
        
        # Normalize to fractions
        if total > 0:
            diff_dist = {k: v/total for k, v in diff_dist.items()}
        
        # Average difficulty (weighted by selection frequency)
        avg_diff = sum(k * v for k, v in diff_dist.items()) if diff_dist else 0
        
        return {
            'avg_difficulty': avg_diff,
            'frac_easy': diff_dist.get(0, 0),
            'frac_medium': diff_dist.get(1, 0),
            'frac_hard': diff_dist.get(2, 0),
            'frac_vhard': diff_dist.get(3, 0)
        }

# -------------------------
# Training: Router
# -------------------------
def train_router(train_ds, val_ds, metrics, diff_tracker):
    print(f"\nTraining with Router on {cfg.device}")
    
    model = MathGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, 
                    cfg.d_ff, cfg.max_seq_len, cfg.dropout).to(cfg.device)
    router = DifficultyRouter(d_model=cfg.d_model, d_k=128).to(cfg.device)
    
    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.01)
    opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    global_step = 0
    total_steps = cfg.epochs * (len(train_ds) // cfg.pool)
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        router.train()
        pool_loader = make_pool_loader(train_ds, cfg.pool)
        
        for pool_indices in pool_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            # Gentle exploration: 5% → 0%
            eps = 0.05 * (1 - global_step / max(total_steps, 1))
            
            # Load pool
            pool_samples = [train_ds[i] for i in pool_indices]
            xs, ys, diffs = collate_fn(pool_samples)
            
            X = xs.to(cfg.device)  # [M, L]
            Y = ys.to(cfg.device)  # [M, L]
            M = X.size(0)
            
            # Get sample embeddings for router (mean-pooled token embeddings)
            with torch.no_grad():
                tok_emb = model.tok_emb(X)  # [M, L, d]
                # Mask padding tokens
                pad_mask = (X != tok.pad_token_id).float().unsqueeze(-1)  # [M, L, 1]
                tok_emb_masked = tok_emb * pad_mask
                # Mean pool (excluding padding)
                feats = tok_emb_masked.sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)  # [M, d]
            
            # Router scoring
            scores = router(feats)  # [M]
            probs = torch.softmax(scores / cfg.temp, dim=0)  # [M]
            logp = torch.log(probs.clamp_min(1e-12))
            
            # Selection (epsilon-greedy)
            k = min(cfg.batch, M)
            if random.random() < eps:
                topk_idx = torch.randperm(M, device=cfg.device)[:k]
            else:
                topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            # Straight-Through for router gradient
            sel_mask_hard = torch.zeros(M, device=cfg.device)
            sel_mask_hard[topk_idx] = 1.0
            sel_mask_st = (sel_mask_hard - probs).detach() + probs
            
            # Selected batch
            X_sel = X[topk_idx]  # [k, L]
            Y_sel = Y[topk_idx]  # [k, L]
            diffs_sel = [diffs[i] for i in topk_idx.cpu().tolist()]
            
            # LM forward
            logits = model(X_sel)  # [k, L, V]
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            # Router loss (ST: align with selection + entropy reg)
            ce_align = -(sel_mask_st * logp).sum() / k
            ent = (probs * logp).sum()  # Negative entropy
            loss_router = ce_align + cfg.lambda_ent * ent
            
            # Optimize
            opt_lm.zero_grad(set_to_none=True)
            opt_router.zero_grad(set_to_none=True)
            (loss_lm + cfg.lambda_router * loss_router).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt_lm.step()
            opt_router.step()
            
            # Track difficulty selection
            diff_tracker.update(diffs_sel)
            
            # Log
            if global_step % cfg.log_every == 0:
                diff_metrics = diff_tracker.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router=loss_router.item(),
                    entropy=-ent.item(),
                    epsilon=eps,
                    **diff_metrics
                )
                print(f"[Router] epoch {epoch} step {global_step}: "
                      f"LM={loss_lm.item():.4f}  Router={loss_router.item():.4f}  "
                      f"H={-ent.item():.3f}  AvgDiff={diff_metrics['avg_difficulty']:.2f}  "
                      f"Easy={diff_metrics['frac_easy']:.2f} Med={diff_metrics['frac_medium']:.2f} "
                      f"Hard={diff_metrics['frac_hard']:.2f} VH={diff_metrics['frac_vhard']:.2f}")
            
            # Evaluate
            if global_step % cfg.eval_every == 0:
                val_loss, val_ppl = evaluate(model, val_ds)
                metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
                print(f"  → Val: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
        
        # End of epoch eval
        val_loss, val_ppl = evaluate(model, val_ds)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Router] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    return model, router

# -------------------------
# Training: Baseline
# -------------------------
def train_baseline(train_ds, val_ds, metrics, diff_tracker):
    print(f"\nTraining Baseline (uniform) on {cfg.device}")
    
    model = MathGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, 
                    cfg.d_ff, cfg.max_seq_len, cfg.dropout).to(cfg.device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pool_loader = make_pool_loader(train_ds, cfg.pool)
        
        for pool_indices in pool_loader:
            if len(pool_indices) < cfg.batch:
                continue
            global_step += 1
            
            # Uniform random selection
            selected_indices = random.sample(pool_indices, cfg.batch)
            selected_samples = [train_ds[i] for i in selected_indices]
            xs, ys, diffs = collate_fn(selected_samples)
            
            X = xs.to(cfg.device)
            Y = ys.to(cfg.device)
            
            # Forward
            logits = model(X)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            
            # Optimize
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            # Track difficulty
            diff_tracker.update(diffs)
            
            # Log
            if global_step % cfg.log_every == 0:
                diff_metrics = diff_tracker.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss.item(),
                    entropy=math.log(cfg.batch),  # Max entropy for uniform
                    **diff_metrics
                )
                print(f"[Baseline] epoch {epoch} step {global_step}: "
                      f"LM={loss.item():.4f}  AvgDiff={diff_metrics['avg_difficulty']:.2f}")
            
            # Evaluate
            if global_step % cfg.eval_every == 0:
                val_loss, val_ppl = evaluate(model, val_ds)
                metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
                print(f"  → Val: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
        
        # End of epoch eval
        val_loss, val_ppl = evaluate(model, val_ds)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Baseline] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    return model

# -------------------------
# Comparison
# -------------------------
def compare_runs(baseline_metrics, router_metrics):
    print("\n" + "="*70)
    print("COMPARISON REPORT - GSM8K Mathematical Reasoning")
    print("="*70)
    
    baseline_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()
    improvement = ((baseline_ppl - router_ppl) / baseline_ppl) * 100
    
    print(f"\n1. Final Validation Perplexity:")
    print(f"   Baseline: {baseline_ppl:.2f}")
    print(f"   Router:   {router_ppl:.2f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    # Convergence analysis
    def steps_to_ppl(metrics, target):
        for i, ppl in enumerate(metrics.history.get('val_ppl', [])):
            if ppl <= target:
                return metrics.history['step'][i]
        return None
    
    target = 50  # Reasonable target for GSM8K
    baseline_steps = steps_to_ppl(baseline_metrics, target)
    router_steps = steps_to_ppl(router_metrics, target)
    
    print(f"\n2. Steps to reach PPL ≤ {target}:")
    print(f"   Baseline: {baseline_steps if baseline_steps else 'Not reached'}")
    print(f"   Router:   {router_steps if router_steps else 'Not reached'}")
    if baseline_steps and router_steps:
        speedup = ((baseline_steps - router_steps) / baseline_steps) * 100
        print(f"   Speedup: {speedup:+.2f}%")
    
    # Difficulty progression
    def avg_last(metrics, key, n=10):
        vals = metrics.history.get(key, [])
        if len(vals) >= n:
            return sum(vals[-n:]) / n
        return 0
    
    print(f"\n3. Difficulty Selection (avg last 10 logs):")
    print(f"   Baseline Avg Difficulty: {avg_last(baseline_metrics, 'avg_difficulty'):.2f}")
    print(f"   Router Avg Difficulty:   {avg_last(router_metrics, 'avg_difficulty'):.2f}")
    
    print(f"\n4. Router Difficulty Distribution (final):")
    print(f"   Easy (1-2 steps):     {avg_last(router_metrics, 'frac_easy'):.1%}")
    print(f"   Medium (3-4 steps):   {avg_last(router_metrics, 'frac_medium'):.1%}")
    print(f"   Hard (5-6 steps):     {avg_last(router_metrics, 'frac_hard'):.1%}")
    print(f"   Very Hard (7+ steps): {avg_last(router_metrics, 'frac_vhard'):.1%}")
    
    # Verdict
    print(f"\n5. Verdict:")
    if improvement > 10:
        print("   ✓✓✓ Router provides strong improvement (>10%)")
    elif improvement > 5:
        print("   ✓✓ Router provides significant improvement (5-10%)")
    elif improvement > 2:
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
    print("="*70)
    print("GSM8K Curriculum Learning Experiment")
    print("="*70)
    
    # Load data
    train_ds = GSM8KDataset("train")
    val_ds = GSM8KDataset("test")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Baseline
    print("\n" + "="*70)
    print("BASELINE (Uniform Random Selection)")
    print("="*70)
    baseline_metrics = MetricsTracker()
    baseline_diff = DifficultyTracker()
    baseline_model = train_baseline(train_ds, val_ds, baseline_metrics, baseline_diff)
    baseline_metrics.save(f"{cfg.save_dir}/baseline_metrics.json")
    
    # Reset seed for fair comparison
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Router
    print("\n" + "="*70)
    print("ROUTER (Difficulty-Aware Curriculum)")
    print("="*70)
    router_metrics = MetricsTracker()
    router_diff = DifficultyTracker()
    router_model, router_net = train_router(train_ds, val_ds, router_metrics, router_diff)
    router_metrics.save(f"{cfg.save_dir}/router_metrics.json")
    
    # Compare
    compare_runs(baseline_metrics, router_metrics)
    
    print(f"✓ Results saved to {cfg.save_dir}/")
    print(f"✓ Baseline metrics: {cfg.save_dir}/baseline_metrics.json")
    print(f"✓ Router metrics: {cfg.save_dir}/router_metrics.json")

if __name__ == "__main__":
    main()