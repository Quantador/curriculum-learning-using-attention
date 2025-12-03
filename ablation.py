# eval_curriculum_ablation.py
# Curriculum learning with attention-based router + ablation study

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
    print("⚠️  wandb not installed. Run: pip install wandb")
    print("   Continuing without wandb logging...")

# -------------------------
# Config
# -------------------------
class Config:
    # Data
    block = 256
    easy_samples = 30000
    hard_samples = 5000
    
    # Training
    batch = 16            # final selected batch size (k)
    pool_mult = 5         # M = pool_mult * batch
    epochs = 10
    lr_lm = 3e-4
    lr_router = 1e-3
    temp = 1.0
    lambda_ent = 0.005
    lambda_router = 0.1   # (for extra regularizers if needed)
    
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
    
    # Wandb
    use_wandb = True
    wandb_project = "curriculum-learning"
    wandb_entity = None  # Set to your wandb username/team, or None
    
    # Eval / saving
    save_dir = "results_ablation"
    log_every = 100

cfg = Config()
cfg.pool = cfg.pool_mult * cfg.batch

# Seed
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
    """Load TinyStories (easy) + OpenWebText (hard)."""
    print("\n" + "="*70)
    print("Loading Mixed Difficulty Dataset")
    print("="*70)
    
    # Easy samples: TinyStories
    print(f"Loading TinyStories (easy)... target: {cfg.easy_samples} samples")
    easy_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{cfg.easy_samples}]")
    
    # Hard samples: OpenWebText
    print(f"Loading OpenWebText (hard)... target: {cfg.hard_samples} samples")
    hard_ds = load_dataset("stas/openwebtext-10k", split="train")
    if len(hard_ds) > cfg.hard_samples:
        hard_ds = hard_ds.select(range(cfg.hard_samples))
    
    print(f"✓ Loaded {len(easy_ds)} easy samples")
    print(f"✓ Loaded {len(hard_ds)} hard samples")
    return easy_ds, hard_ds

def tokenize_and_chunk(text, max_length=cfg.block+1):
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if len(chunk) == max_length:
            chunks.append(chunk)
    return chunks

def make_mixed_chunks(split="train"):
    """Create training/validation chunks from mixed dataset."""
    if split == "train":
        easy_ds, hard_ds = load_mixed_dataset()
        
        print("\nTokenizing and chunking...")
        easy_chunks, hard_chunks = [], []
        for item in easy_ds:
            easy_chunks.extend(tokenize_and_chunk(item["text"]))
        for item in hard_ds:
            hard_chunks.extend(tokenize_and_chunk(item["text"]))
        
        print(f"✓ Created {len(easy_chunks)} easy chunks")
        print(f"✓ Created {len(hard_chunks)} hard chunks")
        
        # Rebalance to 70% easy / 30% hard (up to 50k chunks total)
        target_easy_ratio = 0.7
        if len(easy_chunks) > 0 and len(hard_chunks) > 0:
            total_chunks = len(easy_chunks) + len(hard_chunks)
            current_easy_ratio = len(easy_chunks) / total_chunks
            print(f"⚠ Current ratio - Easy: {current_easy_ratio:.2f}, Hard: {1-current_easy_ratio:.2f}")
            
            target_total = min(50000, total_chunks)
            target_easy = int(target_total * target_easy_ratio)
            target_hard = target_total - target_easy
            
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
            easy_sampled, hard_sampled = easy_chunks, hard_chunks
        
        # Label chunks with difficulty
        easy_labeled = [(chunk, 0) for chunk in easy_sampled]  # 0 = easy
        hard_labeled = [(chunk, 1) for chunk in hard_sampled]  # 1 = hard
        
        all_chunks = easy_labeled + hard_labeled
        random.shuffle(all_chunks)
        
        print(f"✓ Final dataset: {len(all_chunks)} chunks")
        return all_chunks
    
    else:  # validation on WikiText-2
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
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        
    def _causal_mask(self, L):
        m = torch.ones(L, L, dtype=torch.bool, device=self.lm_head.weight.device).triu(1)
        return m
    
    def forward_to_hidden(self, x):
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
# Feature Extraction
# -------------------------
def compute_text_statistics(X, pad_token_id=tok.pad_token_id):
    """
    Compute explicit text statistics as difficulty proxies.
    X: [M, L]
    Returns: [M, 4]
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
        
        length = len(valid_tokens)
        unique_ratio = len(valid_tokens.unique()) / length
        avg_token_id = valid_tokens.float().mean().item() / vocab_size
        std_token_id = valid_tokens.float().std().item() / vocab_size
        stats.append([length / cfg.block, unique_ratio, avg_token_id, std_token_id])
    
    return torch.tensor(stats, device=X.device, dtype=torch.float32)

def extract_hierarchical_hidden(model, X, n_chunks=None):
    """
    Hierarchical pooling over Transformer hidden states.
    X: [M, L]
    Returns: [M, n_chunks*d_model]
    """
    if n_chunks is None:
        n_chunks = cfg.n_chunks
    with torch.no_grad():
        h = model.forward_to_hidden(X)  # [M, L, d_model]
        M, L, d_model = h.shape
        chunk_size = L // n_chunks
        chunk_features = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else L
            chunk = h[:, start:end, :]
            chunk_mean = chunk.mean(dim=1)
            chunk_features.append(chunk_mean)
        learned_features = torch.cat(chunk_features, dim=-1)  # [M, n_chunks*d_model]
    return learned_features

def build_router_features(model, X, use_hidden=True, use_text_stats=True):
    """
    Build router feature vectors from hidden states and/or text stats.
    Returns: [M, d_input]
    """
    feats = []
    if use_hidden:
        feats.append(extract_hierarchical_hidden(model, X))  # [M, n_chunks*d_model]
    if use_text_stats:
        feats.append(compute_text_statistics(X))             # [M, 4]
    if len(feats) == 0:
        raise ValueError("At least one of use_hidden/use_text_stats must be True.")
    return torch.cat(feats, dim=-1) if len(feats) > 1 else feats[0]

# -------------------------
# Routers
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

class MLPRouter(nn.Module):
    def __init__(self, d_input, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1)
        )
    
    def forward(self, feats):
        return self.net(feats).squeeze(-1)  # [M]

# -------------------------
# Loss & Evaluation
# -------------------------
def compute_loss_per_sample(model, X, Y, loss_fn):
    logits = model(X)  # [B, L, vocab]
    losses = []
    for i in range(X.size(0)):
        loss = loss_fn(logits[i].reshape(-1, logits.size(-1)), Y[i].reshape(-1))
        losses.append(loss)
    return torch.stack(losses)  # [B]

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
# Metrics & Diversity
# -------------------------
class MetricsTracker:
    def __init__(self, use_wandb=False):
        self.history = defaultdict(list)
        self.use_wandb = use_wandb
    
    def log(self, epoch, step, **kwargs):
        self.history["epoch"].append(epoch)
        self.history["step"].append(step)
        for k, v in kwargs.items():
            self.history[k].append(v)
        if self.use_wandb and WANDB_AVAILABLE:
            wandb_dict = {"epoch": epoch, "step": step}
            wandb_dict.update(kwargs)
            wandb.log(wandb_dict, step=step)
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(self.history), f, indent=2)
    
    def get_final_ppl(self):
        if "val_ppl" in self.history and len(self.history["val_ppl"]) > 0:
            return self.history["val_ppl"][-1]
        return float("inf")

class DiversityTracker:
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.selection_counts = torch.zeros(dataset_size)
        self.step_selections = []
        self.difficulty_selections = defaultdict(int)
    
    def update(self, selected_indices, difficulties):
        self.step_selections.append(set(selected_indices))
        for idx in selected_indices:
            if 0 <= idx < self.dataset_size:
                self.selection_counts[idx] += 1
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
        
        total_sel = sum(self.difficulty_selections.values())
        easy_ratio = self.difficulty_selections[0] / max(total_sel, 1)
        hard_ratio = self.difficulty_selections[1] / max(total_sel, 1)
        
        return {
            "coverage": coverage,
            "balance_std": balance_std,
            "unique_ratio": unique_ratio,
            "easy_ratio": easy_ratio,
            "hard_ratio": hard_ratio,
        }

# -------------------------
# Training: Baseline (Uniform)
# -------------------------
def train_baseline(train_ds, val_ds, metrics, diversity, run_name="baseline", use_wandb=False):
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config={
                "method": "baseline",
                "batch_size": cfg.batch,
                "pool_size": cfg.pool,
                "epochs": cfg.epochs,
                "lr_lm": cfg.lr_lm,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "block_size": cfg.block,
                "dataset_size": len(train_ds),
                "seed": cfg.seed,
            },
            reinit=True,
        )
    
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
            
            selected_indices = random.sample(pool_indices, cfg.batch)
            batch_data = [train_ds[i] for i in selected_indices]
            xs, ys, diffs = zip(*batch_data)
            X = torch.stack(xs, 0).to(cfg.device)
            Y = torch.stack(ys, 0).to(cfg.device)
            
            logits = model(X)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            diversity.update(selected_indices, diffs)
            
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss.item(),
                    entropy=math.log(cfg.batch),  # max entropy
                    **div_metrics,
                )
                print(f"[Baseline] epoch {epoch} step {global_step}: "
                      f"LM={loss.item():.4f}  "
                      f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}")
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Baseline] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model

# -------------------------
# Training: Router (with Ablation Flags)
# -------------------------
def train_router(
    train_ds,
    val_ds,
    metrics,
    diversity,
    run_name="router_full",
    use_wandb=False,
    # Ablation flags:
    use_hidden_feats=True,
    use_text_stats=True,
    router_type="attention",       # "attention" or "mlp"
    use_entropy=True,
    reward_type="improvement",     # "improvement" or "neg_loss_before"
    selection_strategy="topk",     # "topk" or "random"
    use_curriculum_bias=False,
):
    """Router training with REINFORCE and ablation controls."""
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config={
                "method": "router",
                "run_name": run_name,
                "batch_size": cfg.batch,
                "pool_size": cfg.pool,
                "pool_mult": cfg.pool_mult,
                "epochs": cfg.epochs,
                "lr_lm": cfg.lr_lm,
                "lr_router": cfg.lr_router,
                "temperature": cfg.temp,
                "lambda_ent": cfg.lambda_ent,
                "lambda_router": cfg.lambda_router,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_chunks": cfg.n_chunks,
                "block_size": cfg.block,
                "dataset_size": len(train_ds),
                "seed": cfg.seed,
                "reward_type": reward_type,
                "features_hidden": use_hidden_feats,
                "features_stats": use_text_stats,
                "router_type": router_type,
                "selection_strategy": selection_strategy,
                "curriculum_bias": use_curriculum_bias,
            },
            reinit=True,
        )
    
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    
    # Router input dimension
    d_router_input = 0
    if use_hidden_feats:
        d_router_input += cfg.n_chunks * cfg.d_model
    if use_text_stats:
        d_router_input += 4
    assert d_router_input > 0, "Router input dimension must be > 0."
    
    if router_type == "attention":
        router = AttentionRouter(d_input=d_router_input, d_k=64).to(cfg.device)
    elif router_type == "mlp":
        router = MLPRouter(d_input=d_router_input, d_hidden=256).to(cfg.device)
    else:
        raise ValueError(f"Unknown router_type: {router_type}")
    
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
            
            # Extract router features
            feats = build_router_features(
                model, X,
                use_hidden=use_hidden_feats,
                use_text_stats=use_text_stats,
            )  # [M, d_router_input]
            
            # Router scores
            scores = router(feats)  # [M]
            
            # Curriculum bias (optional): boost easy early, neutral later
            training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
            curriculum_strength = max(0.0, 1.0 - training_progress)
            if use_curriculum_bias:
                difficulty_tensor = torch.tensor(diffs, device=scores.device, dtype=torch.float32)
                # Easy (0) → +alpha, Hard (1) → 0
                alpha = 2.0
                curriculum_bonus = curriculum_strength * alpha * (1.0 - difficulty_tensor)
                scores = scores + curriculum_bonus
            
            # Selection probabilities
            probs = torch.softmax(scores / cfg.temp, dim=0)
            
            # Selection strategy
            k = cfg.batch
            if selection_strategy == "topk":
                topk_idx = torch.topk(probs, k=k, dim=0).indices
            elif selection_strategy == "random":
                # ignore scores, sample uniformly from pool
                topk_idx = torch.tensor(
                    random.sample(range(M), k=k),
                    device=probs.device,
                    dtype=torch.long,
                )
            else:
                raise ValueError(f"Unknown selection_strategy: {selection_strategy}")
            
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # --- Reward computation ---
            if reward_type == "improvement":
                # loss BEFORE update
                with torch.no_grad():
                    loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            elif reward_type == "neg_loss_before":
                with torch.no_grad():
                    loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
            else:
                raise ValueError(f"Unknown reward_type: {reward_type}")
            
            # LM forward + update
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            opt_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            opt_lm.step()
            
            # loss AFTER update (only needed for improvement reward)
            if reward_type == "improvement":
                with torch.no_grad():
                    loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)
                improvement = loss_before - loss_after
                improvement = improvement.clamp(min=0.0)
                reward = improvement
            elif reward_type == "neg_loss_before":
                # reward = - loss_before (favor easy / low-loss samples)
                reward = -loss_before
            
            baseline = reward.mean()
            
            # REINFORCE loss
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -((reward - baseline) * torch.log(sel_probs)).mean()
            
            # Entropy regularization
            ent = (probs * torch.log(probs.clamp_min(1e-12))).sum()
            if use_entropy:
                loss_router = reinforce + cfg.lambda_ent * ent
            else:
                loss_router = reinforce
            
            opt_router.zero_grad(set_to_none=True)
            loss_router.backward()
            opt_router.step()
            
            # Diversity
            selected_indices = [pool_indices[i] for i in topk_idx.cpu().tolist()]
            selected_diffs = [diffs[i] for i in topk_idx.cpu().tolist()]
            diversity.update(selected_indices, selected_diffs)
            
            # Logging
            if global_step % cfg.log_every == 0:
                div_metrics = diversity.get_metrics()
                metrics.log(
                    epoch=epoch,
                    step=global_step,
                    loss_lm=loss_lm.item(),
                    loss_router_reinforce=reinforce.item(),
                    entropy=-ent.item(),
                    avg_reward=reward.mean().item(),
                    curriculum_bias_strength=curriculum_strength if use_curriculum_bias else 0.0,
                    **div_metrics,
                )
                print(
                    f"[{run_name}] epoch {epoch} step {global_step}: "
                    f"LM={loss_lm.item():.4f}  "
                    f"Reward={reward.mean().item():.4f}  "
                    f"H={-ent.item():.3f}  "
                    f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}"
                )
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [{run_name}] epoch {epoch}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model, router

# -------------------------
# Comparison helper
# -------------------------
def avg_last(metrics, key, n=5):
    vals = metrics.history.get(key, [])
    if len(vals) >= n:
        return sum(vals[-n:]) / n
    return 0.0

# -------------------------
# Ablation Study Runner
# -------------------------
def run_ablation_study(train_ds, val_ds, use_wandb=False):
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Define ablation experiments
    # You can comment out some to save compute
    experiments = {
        "baseline": {
            "type": "baseline",
        },
        "router_full": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=True,
                router_type="attention",
                use_entropy=True,
                reward_type="improvement",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        "router_no_hidden": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=False,
                use_text_stats=True,
                router_type="attention",
                use_entropy=True,
                reward_type="improvement",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        "router_no_stats": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=False,
                router_type="attention",
                use_entropy=True,
                reward_type="improvement",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        "router_mlp": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=True,
                router_type="mlp",
                use_entropy=True,
                reward_type="improvement",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        "router_no_entropy": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=True,
                router_type="attention",
                use_entropy=False,
                reward_type="improvement",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        "router_random_selection": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=True,
                router_type="attention",
                use_entropy=True,
                reward_type="improvement",
                selection_strategy="random",  # ignores scores
                use_curriculum_bias=False,
            ),
        },
        "router_no_improvement": {
            "type": "router",
            "kwargs": dict(
                use_hidden_feats=True,
                use_text_stats=True,
                router_type="attention",
                use_entropy=True,
                reward_type="neg_loss_before",
                selection_strategy="topk",
                use_curriculum_bias=False,
            ),
        },
        # Example: curriculum bias on
        # "router_curriculum_bias": {
        #     "type": "router",
        #     "kwargs": dict(
        #         use_hidden_feats=True,
        #         use_text_stats=True,
        #         router_type="attention",
        #         use_entropy=True,
        #         reward_type="improvement",
        #         selection_strategy="topk",
        #         use_curriculum_bias=True,
        #     ),
        # },
    }
    
    results_summary = {}
    
    for name, cfg_exp in experiments.items():
        print("\n" + "="*70)
        print(f"Running experiment: {name}")
        print("="*70)
        
        # Reset seeds for fair comparison
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        metrics = MetricsTracker(use_wandb=use_wandb)
        diversity = DiversityTracker(len(train_ds))
        
        if cfg_exp["type"] == "baseline":
            model = train_baseline(
                train_ds, val_ds,
                metrics, diversity,
                run_name=name,
                use_wandb=use_wandb,
            )
        else:
            kwargs = cfg_exp.get("kwargs", {})
            model, router = train_router(
                train_ds, val_ds,
                metrics, diversity,
                run_name=name,
                use_wandb=use_wandb,
                **kwargs,
            )
        
        metrics_path = f"{cfg.save_dir}/{name}_metrics.json"
        metrics.save(metrics_path)
        print(f"✓ Saved metrics to {metrics_path}")
        
        final_ppl = metrics.get_final_ppl()
        coverage = avg_last(metrics, "coverage")
        entropy = avg_last(metrics, "entropy")
        easy_ratio = avg_last(metrics, "easy_ratio")
        hard_ratio = avg_last(metrics, "hard_ratio")
        
        results_summary[name] = dict(
            final_ppl=final_ppl,
            coverage=coverage,
            entropy=entropy,
            easy_ratio=easy_ratio,
            hard_ratio=hard_ratio,
        )
    
    # Save summary
    summary_path = f"{cfg.save_dir}/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n" + "="*70)
    print("Ablation study summary")
    print("="*70)
    for name, res in results_summary.items():
        print(f"{name}: "
              f"PPL={res['final_ppl']:.2f}, "
              f"Cov={res['coverage']:.3f}, "
              f"Ent={res['entropy']:.3f}, "
              f"Easy={res['easy_ratio']:.2f}, "
              f"Hard={res['hard_ratio']:.2f}")
    print(f"\n✓ Summary saved to {summary_path}")

# -------------------------
# Main
# -------------------------
def main():
    print("="*70)
    print("CURRICULUM LEARNING ABLATION STUDY")
    print("="*70)
    
    # Wandb
    use_wandb = cfg.use_wandb and WANDB_AVAILABLE
    if cfg.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb requested but not available. Install with: pip install wandb")
        print("   Continuing without wandb logging...\n")
    elif use_wandb:
        print(f"✓ wandb enabled - project: {cfg.wandb_project}\n")
    
    # Data
    train_chunks = make_mixed_chunks("train")
    val_chunks = make_mixed_chunks("validation")
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)
    print(f"\n✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Run ablation study
    run_ablation_study(train_ds, val_ds, use_wandb=use_wandb)

if __name__ == "__main__":
    main()
