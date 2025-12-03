import math, torch, random, json
from torch import nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from pathlib import Path
from collections import defaultdict
import copy

# -------------------------
# Config
# -------------------------
class Config:
    # Data
    block = 256
    easy_samples = 30000   
    hard_samples = 5000    
    
    # Training
    batch = 16           # final selected batch size (k)
    pool_mult = 5        # candidate pool multiplier -> M = pool_mult * batch (increased)
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
    n_chunks = 8  # For hierarchical pooling
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    
    # Eval
    save_dir = "results_improved"
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

print(f"Device: {cfg.device}")
print(f"Vocab size: {vocab_size}")

# -------------------------
# Data - Mixed Difficulty Dataset
# -------------------------
def load_mixed_dataset():
    """Load TinyStories (easy) + OpenWebText (hard)"""
    print("\n" + "="*70)
    print("Loading Mixed Difficulty Dataset")
    print("="*70)
    
    # Easy samples: TinyStories (children's stories)
    print(f"Loading TinyStories (easy)... target: {cfg.easy_samples} samples")
    easy_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{cfg.easy_samples}]")
    
    # Hard samples: OpenWebText (Reddit, news)
    print(f"Loading OpenWebText (hard)... target: {cfg.hard_samples} samples")
    hard_ds = load_dataset("stas/openwebtext-10k", split="train")
    # Take subset if needed
    if len(hard_ds) > cfg.hard_samples:
        hard_ds = hard_ds.select(range(cfg.hard_samples))
    
    print(f"✓ Loaded {len(easy_ds)} easy samples")
    print(f"✓ Loaded {len(hard_ds)} hard samples")
    
    return easy_ds, hard_ds

def tokenize_and_chunk(text, max_length=cfg.block+1):
    """Tokenize text and create chunks"""
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    
    # Create chunks of max_length
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if len(chunk) == max_length:  # Only keep full chunks
            chunks.append(chunk)
    
    return chunks

def make_mixed_chunks(split="train"):
    """Create training/validation chunks from mixed dataset"""
    if split == "train":
        easy_ds, hard_ds = load_mixed_dataset()
        
        print("\nTokenizing and chunking...")
        easy_chunks = []
        for item in easy_ds:
            chunks = tokenize_and_chunk(item['text'])
            easy_chunks.extend(chunks)
        
        hard_chunks = []
        for item in hard_ds:
            chunks = tokenize_and_chunk(item['text'])
            hard_chunks.extend(chunks)
        
        print(f"✓ Created {len(easy_chunks)} easy chunks")
        print(f"✓ Created {len(hard_chunks)} hard chunks")
        
        # CRITICAL FIX: Balance the dataset after chunking
        # We want 70% easy, 30% hard for clear curriculum signal
        target_easy_ratio = 0.7
        
        if len(easy_chunks) > 0 and len(hard_chunks) > 0:
            total_chunks = len(easy_chunks) + len(hard_chunks)
            current_easy_ratio = len(easy_chunks) / total_chunks
            
            print(f"⚠ Current ratio - Easy: {current_easy_ratio:.2f}, Hard: {1-current_easy_ratio:.2f}")
            
            # Downsample to achieve target ratio
            target_total = min(50000, total_chunks)  # Cap at 50k for speed
            target_easy = int(target_total * target_easy_ratio)
            target_hard = target_total - target_easy
            
            # Sample with replacement if needed
            if len(easy_chunks) >= target_easy:
                easy_sampled = random.sample(easy_chunks, target_easy)
            else:
                easy_sampled = random.choices(easy_chunks, k=target_easy)
            
            if len(hard_chunks) >= target_hard:
                hard_sampled = random.sample(hard_chunks, target_hard)
            else:
                hard_sampled = random.choices(hard_chunks, k=target_hard)
            
            print(f"✓ Rebalanced to {len(easy_sampled)} easy, {len(hard_sampled)} hard")
            print(f"✓ New ratio - Easy: {len(easy_sampled)/(len(easy_sampled)+len(hard_sampled)):.2f}")
        else:
            easy_sampled = easy_chunks
            hard_sampled = hard_chunks
        
        # Label chunks with difficulty
        easy_labeled = [(chunk, 0) for chunk in easy_sampled]  # 0 = easy
        hard_labeled = [(chunk, 1) for chunk in hard_sampled]  # 1 = hard
        
        # Combine and shuffle
        all_chunks = easy_labeled + hard_labeled
        random.shuffle(all_chunks)
        
        print(f"✓ Final dataset: {len(all_chunks)} chunks")
        
        return all_chunks
    
    else:  # validation - use WikiText for consistency
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = tok.eos_token.join(ds["text"])
        ids = tok(text, add_special_tokens=False)["input_ids"]
        L = (len(ids) // (cfg.block + 1)) * (cfg.block + 1)
        ids = ids[:L]
        chunks = [ids[i:i+cfg.block+1] for i in range(0, L, cfg.block+1)]
        return [(chunk, -1) for chunk in chunks]  # -1 = unknown difficulty

class MixedLMDataset(Dataset):
    def __init__(self, labeled_chunks):
        self.x = [torch.tensor(c[:-1], dtype=torch.long) for c, _ in labeled_chunks]
        self.y = [torch.tensor(c[1:],  dtype=torch.long) for c, _ in labeled_chunks]
        self.difficulty = [d for _, d in labeled_chunks]
    
    def __len__(self): 
        return len(self.x)
    
    def __getitem__(self, i): 
        return self.x[i], self.y[i], self.difficulty[i]

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
        self.d_model = d_model
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
    
    def forward_to_hidden(self, x):
        """Forward pass returning hidden states (for feature extraction)"""
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        mask = self._causal_mask(L)
        h = self.tr(h, mask=mask)
        return h  # [B, L, d_model]

    def forward(self, x):
        h = self.forward_to_hidden(x)
        return self.lm_head(h)

# -------------------------
# Feature Extraction - Hierarchical + Text Stats
# -------------------------
def compute_text_statistics(X, pad_token_id=tok.pad_token_id):
    """
    Compute simple text statistics as explicit difficulty features.
    X: [M, L] tensor of token IDs
    Returns: [M, 4] tensor of features
    """
    M, L = X.shape
    stats = []
    
    for i in range(M):
        tokens = X[i]
        valid_mask = tokens != pad_token_id
        valid_tokens = tokens[valid_mask]
        
        if len(valid_tokens) == 0:
            stats.append([0, 0, 0, 0])
            continue
        
        # 1. Sequence length
        length = len(valid_tokens)
        
        # 2. Unique token ratio (vocabulary diversity)
        unique_ratio = len(valid_tokens.unique()) / length
        
        # 3. Average token ID (lower IDs = more common words)
        avg_token_id = valid_tokens.float().mean().item() / vocab_size  # Normalize
        
        # 4. Token ID std (variation in vocabulary)
        std_token_id = valid_tokens.float().std().item() / vocab_size
        
        stats.append([length / cfg.block, unique_ratio, avg_token_id, std_token_id])
    
    return torch.tensor(stats, device=X.device, dtype=torch.float32)

def extract_hierarchical_features(model, X):
    """
    Extract hierarchical features from model hidden states.
    X: [M, L] input tokens
    Returns: [M, n_chunks*d_model + 4] features
    """
    with torch.no_grad():
        # Get hidden states (model's actual understanding)
        h = model.forward_to_hidden(X)  # [M, L, d_model]
        M, L, d_model = h.shape
        
        # Hierarchical pooling - divide sequence into chunks
        n_chunks = cfg.n_chunks
        chunk_size = L // n_chunks
        
        chunk_features = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else L
            chunk = h[:, start:end, :]  # [M, chunk_size, d_model]
            chunk_mean = chunk.mean(dim=1)  # [M, d_model]
            chunk_features.append(chunk_mean)
        
        # Concatenate all chunk features
        learned_features = torch.cat(chunk_features, dim=-1)  # [M, n_chunks*d_model]
        
        # Add text statistics (explicit difficulty metrics)
        text_stats = compute_text_statistics(X)  # [M, 4]
        
        # Combine learned and statistical features
        features = torch.cat([learned_features, text_stats], dim=-1)
        
    return features  # [M, n_chunks*d_model + 4]

# -------------------------
# Router - Attention-based
# -------------------------
class AttentionRouter(nn.Module):
    def __init__(self, d_input, d_k=64):
        super().__init__()
        self.K = nn.Linear(d_input, d_k, bias=False)
        self.q = nn.Parameter(torch.randn(d_k))
        nn.init.normal_(self.q, std=0.02)
        
    def forward(self, feats):  # [M, d_input]
        K = self.K(feats)      # [M, d_k]
        scores = K @ self.q    # [M]
        return scores

# -------------------------
# Loss Computation
# -------------------------
def compute_loss_per_sample(model, X, Y, loss_fn):
    """Compute loss for each sample separately"""
    logits = model(X)  # [batch, L, vocab]
    # Compute per-sample loss
    losses = []
    for i in range(X.size(0)):
        loss = loss_fn(logits[i].reshape(-1, logits.size(-1)), Y[i].reshape(-1))
        losses.append(loss)
    return torch.stack(losses)  # [batch]

# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, ds, loss_fn):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i in range(len(ds)):
        x, y, _ = ds[i]
        x = x.unsqueeze(0).to(cfg.device)
        y = y.unsqueeze(0).to(cfg.device)
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
        self.difficulty_selections = defaultdict(int)  # Track easy vs hard
        
    def update(self, selected_indices, difficulties):
        self.step_selections.append(set(selected_indices))
        for idx in selected_indices:
            if 0 <= idx < self.dataset_size:
                self.selection_counts[idx] += 1
        
        # Track difficulty distribution
        for d in difficulties:
            self.difficulty_selections[d] += 1
    
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
        
        # Difficulty ratio
        total_sel = sum(self.difficulty_selections.values())
        easy_ratio = self.difficulty_selections[0] / max(total_sel, 1)
        hard_ratio = self.difficulty_selections[1] / max(total_sel, 1)
            
        return {
            'coverage': coverage,
            'balance_std': balance_std,
            'unique_ratio': unique_ratio,
            'easy_ratio': easy_ratio,
            'hard_ratio': hard_ratio
        }

# -------------------------
# Training: REINFORCE with Loss Improvement
# -------------------------
def train_router(train_ds, val_ds, metrics, diversity):
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    
    # Router input dimension: n_chunks * d_model + 4 (text stats)
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
            
            # Extract hierarchical features
            feats = extract_hierarchical_features(model, X)  # [M, d_router_input]
            
            # Router scoring
            scores = router(feats)
            
            # CURRICULUM BIAS: Encourage easy samples early in training   NOW DISABLED
            training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
            curriculum_strength = max(0, 1.0 - training_progress)  # 1.0 → 0.0
            
            # if curriculum_strength > 0:
            #     # Boost easy samples, penalize hard samples
            #     difficulty_tensor = torch.tensor([diffs[i] for i in range(M)], 
            #                                      device=scores.device, dtype=torch.float32)
            #     curriculum_bonus = curriculum_strength * 2.0 * (1 - difficulty_tensor)  # Easy=+2, Hard=0
            #     scores = scores + curriculum_bonus
            
            probs = torch.softmax(scores / cfg.temp, dim=0)
            
            # Hard select top-k
            k = cfg.batch
            topk_idx = torch.topk(probs, k=k, dim=0).indices
            
            # Selected batch
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # ===== LOSS IMPROVEMENT REWARD =====
            # Step 1: Measure loss BEFORE update
            with torch.no_grad():
                loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            # Step 2: LM forward and backward
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            opt_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            opt_lm.step()
            
            # Step 3: Measure loss AFTER update
            with torch.no_grad():
                loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            
            # Step 4: Reward = improvement
            improvement = loss_before - loss_after  # [k]
            
            improvement = improvement.clamp(min=0.0)
            
            baseline = improvement.mean()
            
            # Router loss (REINFORCE with improvement reward)
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((improvement - baseline) * torch.log(sel_probs)).mean()
            
            # Entropy regularization
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            loss_router = reinforce + cfg.lambda_ent * ent
            
            # Router update
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
                      f"Improve={improvement.mean().item():.4f}  "
                      f"H={-ent.item():.3f}  "
                      f"CurrBias={curriculum_strength:.2f}  "
                      f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}")
        
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
            batch_data = [train_ds[i] for i in selected_indices]
            xs, ys, diffs = zip(*batch_data)
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
            diversity.update(selected_indices, diffs)
            
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
                      f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}")
        
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
    
    # Difficulty progression (analyze easy vs hard ratio over time)
    if 'easy_ratio' in router_metrics.history:
        easy_ratios = router_metrics.history['easy_ratio']
        early_easy = sum(easy_ratios[:len(easy_ratios)//3]) / max(len(easy_ratios)//3, 1)
        late_easy = sum(easy_ratios[-len(easy_ratios)//3:]) / max(len(easy_ratios)//3, 1)
        
        print(f"\n2. Curriculum Progression (Router):")
        print(f"   Early training easy ratio: {early_easy:.2f}")
        print(f"   Late training easy ratio:  {late_easy:.2f}")
        print(f"   Shift toward hard samples: {(early_easy - late_easy):.2f}")
        
        if early_easy > late_easy + 0.05:
            print("   ✓ Clear curriculum: easy→hard progression observed")
        else:
            print("   ✗ No clear curriculum progression")
    
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
    print("="*70)
    print("IMPROVED CURRICULUM LEARNING EXPERIMENT")
    print("="*70)
    print(f"\nImprovements:")
    print("  1. Mixed dataset: TinyStories (easy) + OpenWebText (hard)")
    print("  2. Loss improvement reward signal")
    print("  3. Hierarchical hidden states + text statistics features")
    print()
    
    # Load data
    train_chunks = make_mixed_chunks("train")
    val_chunks = make_mixed_chunks("validation")
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)
    print(f"\n✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Baseline
    print("\n" + "="*70)
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
    print("ROUTER (Attention-based Curriculum with Loss Improvement)")
    print("="*70)
    router_metrics = MetricsTracker()
    router_diversity = DiversityTracker(len(train_ds))
    router_model, router_net = train_router(train_ds, val_ds, router_metrics, router_diversity)
    router_metrics.save(f"{cfg.save_dir}/router_metrics.json")
    
    # Compare
    compare_runs(baseline_metrics, router_metrics)
    
    print(f"✓ Results saved to {cfg.save_dir}/")
    print(f"✓ Metrics: baseline_metrics.json, router_metrics.json")

if __name__ == "__main__":
    main()