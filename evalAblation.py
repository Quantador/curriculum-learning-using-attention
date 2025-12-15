# eval_curriculum_ablation.py
# Ablation study for curriculum learning via attention-based sample selection

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
    batch = 16           # final selected batch size (k)
    pool_mult = 5        # candidate pool multiplier -> M = pool_mult * batch
    epochs = 10
    lr_lm = 3e-4
    lr_router = 1e-3
    temp = 1.0
    lambda_ent = 0.005   # entropy regularization coefficient
    lambda_router = 0.1  # kept for potential extensions
    
    # Model
    d_model = 256
    n_layers = 4
    n_heads = 8
    d_ff = 1024
    
    # Features
    n_chunks = 8  # For hierarchical pooling
    
    # Router / Ablation knobs
    reward_type = "improvement"  # "improvement", "loss_after", "random"
    clip_negative_improvement = True
    use_reward_baseline = True
    feature_mode = "full"        # "full", "hidden_only", "text_only", "simple_mean"
    router_arch = "attention"    # "attention" or "mlp"
    selection_mode = "topk"      # "topk" or "sample"
    curriculum_bias = False
    curriculum_bias_strength = 2.0  # used if curriculum_bias=True
    
    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0
    
    # Wandb
    use_wandb = True
    wandb_project = "curriculum-learning"
    wandb_entity = None  # Set to your wandb username/team, or None
    
    # Eval / experiment bookkeeping
    save_dir = "results_ablation"
    log_every = 100
    experiment_name = "default"

cfg = Config()
cfg.pool = cfg.pool_mult * cfg.batch

# Set seed (will be reset per experiment as well)
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
    """Tokenize text and create chunks of max_length"""
    tokens = tok(text, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i+max_length]
        if len(chunk) == max_length:
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
        
        # Rebalance to ~70% easy, 30% hard (cap total at 50k)
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
            easy_sampled = easy_chunks
            hard_sampled = hard_chunks
        
        # Label chunks with difficulty
        easy_labeled = [(chunk, 0) for chunk in easy_sampled]  # 0 = easy
        hard_labeled = [(chunk, 1) for chunk in hard_sampled]  # 1 = hard
        
        all_chunks = easy_labeled + hard_labeled
        random.shuffle(all_chunks)
        
        print(f"✓ Final dataset: {len(all_chunks)} chunks")
        
        return all_chunks
    
    else:  # validation
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
# Feature Extraction
# -------------------------
def compute_text_statistics(X, pad_token_id=tok.pad_token_id):
    """
    Compute simple text statistics as explicit difficulty features.
    X: [M, L] tensor of token IDs
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

def extract_features(model, X):
    """
    Extract features according to cfg.feature_mode:
      - "full": hierarchical hidden + text stats
      - "hidden_only": hierarchical hidden only
      - "text_only": text stats only
      - "simple_mean": whole-sequence mean hidden (+ optional text stats)
    Returns: [M, D]
    """
    mode = cfg.feature_mode
    with torch.no_grad():
        M, L = X.shape
        
        use_hidden = mode in ("full", "hidden_only", "simple_mean")
        use_stats = mode in ("full", "text_only")
        
        if use_hidden:
            h = model.forward_to_hidden(X)  # [M, L, d_model]
            M_, L_, d_model = h.shape
            
            if mode == "simple_mean":
                # Single mean over full sequence
                learned_features = h.mean(dim=1)  # [M, d_model]
            else:
                # Hierarchical pooling into n_chunks
                n_chunks = cfg.n_chunks
                chunk_size = L_ // n_chunks
                chunk_features = []
                for i in range(n_chunks):
                    start = i * chunk_size
                    end = (i + 1) * chunk_size if i < n_chunks - 1 else L_
                    chunk = h[:, start:end, :]   # [M, chunk_size, d_model]
                    chunk_mean = chunk.mean(dim=1)
                    chunk_features.append(chunk_mean)
                learned_features = torch.cat(chunk_features, dim=-1)  # [M, n_chunks*d_model]
        else:
            learned_features = None
        
        if use_stats:
            text_stats = compute_text_statistics(X)  # [M, 4]
        else:
            text_stats = None
        
        if learned_features is not None and text_stats is not None:
            feats = torch.cat([learned_features, text_stats], dim=-1)
        elif learned_features is not None:
            feats = learned_features
        elif text_stats is not None:
            feats = text_stats
        else:
            raise ValueError("Feature mode produces no features.")
    
    return feats  # [M, D]

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
    def __init__(self, d_input, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, feats):  # [M, d_input]
        return self.net(feats).squeeze(-1)  # [M]

def build_router(d_input):
    if cfg.router_arch == "attention":
        return AttentionRouter(d_input=d_input, d_k=64)
    elif cfg.router_arch == "mlp":
        return MLPRouter(d_input=d_input, hidden=128)
    else:
        raise ValueError(f"Unknown router_arch: {cfg.router_arch}")

# -------------------------
# Loss Computation
# -------------------------
def compute_loss_per_sample(model, X, Y, loss_fn):
    """Compute loss for each sample separately"""
    logits = model(X)  # [batch, L, vocab]
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
    def __init__(self, use_wandb=False):
        self.history = defaultdict(list)
        self.use_wandb = use_wandb
        
    def log(self, epoch, step, **kwargs):
        self.history['epoch'].append(epoch)
        self.history['step'].append(step)
        for k, v in kwargs.items():
            self.history[k].append(v)
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb_dict = {'epoch': epoch, 'step': step}
            wandb_dict.update(kwargs)
            wandb.log(wandb_dict, step=step)
    
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
        
        recent = self.step_selections[-100:] if len(self.step_selections) >= 100 else self.step_selections
        if recent:
            unique_recent = len(set.union(*recent))
            total_selections = sum(len(s) for s in recent)
            unique_ratio = unique_recent / max(total_selections, 1)
        else:
            unique_ratio = 0
        
        total_sel = sum(self.difficulty_selections.values())
        easy_ratio = self.difficulty_selections[0] / max(total_sel, 1) if total_sel > 0 else 0
        hard_ratio = self.difficulty_selections[1] / max(total_sel, 1) if total_sel > 0 else 0
            
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
def train_router(train_ds, val_ds, metrics, diversity, use_wandb=False):
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.experiment_name,
            group="ablation",
            config={
                "method": "router",
                "experiment_name": cfg.experiment_name,
                "batch_size": cfg.batch,
                "pool_size": cfg.pool,
                "pool_mult": cfg.pool_mult,
                "epochs": cfg.epochs,
                "lr_lm": cfg.lr_lm,
                "lr_router": cfg.lr_router,
                "temperature": cfg.temp,
                "lambda_ent": cfg.lambda_ent,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_chunks": cfg.n_chunks,
                "block_size": cfg.block,
                "dataset_size": len(train_ds),
                "seed": cfg.seed,
                "reward_type": cfg.reward_type,
                "clip_negative_improvement": cfg.clip_negative_improvement,
                "use_reward_baseline": cfg.use_reward_baseline,
                "feature_mode": cfg.feature_mode,
                "router_arch": cfg.router_arch,
                "selection_mode": cfg.selection_mode,
                "curriculum_bias": cfg.curriculum_bias,
                "curriculum_bias_strength": cfg.curriculum_bias_strength
            },
            reinit=True
        )
    
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.block).to(cfg.device)
    
    router = None
    opt_router = None
    
    opt_lm = torch.optim.AdamW(model.parameters(), lr=cfg.lr_lm)
    loss_fn = nn.CrossEntropyLoss()
    
    global_step = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if router is not None:
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
            
            # Extract features
            feats = extract_features(model, X)  # [M, D]
            
            # Lazily build router once we know d_input
            if router is None:
                d_router_input = feats.size(-1)
                router = build_router(d_router_input).to(cfg.device)
                opt_router = torch.optim.AdamW(router.parameters(), lr=cfg.lr_router)
                router.train()
            
            # Router scoring
            scores = router(feats)  # [M]
            
            # Curriculum bias (optional)
            training_progress = global_step / (len(train_ds) // cfg.pool * cfg.epochs)
            curriculum_strength = max(0, 1.0 - training_progress) if cfg.curriculum_bias else 0.0
            
            if cfg.curriculum_bias and curriculum_strength > 0:
                difficulty_tensor = torch.tensor(
                    [d for d in diffs],
                    device=scores.device,
                    dtype=torch.float32
                )
                # Easy=0 -> bonus, Hard=1 -> no bonus
                curriculum_bonus = curriculum_strength * cfg.curriculum_bias_strength * (1.0 - difficulty_tensor)
                scores = scores + curriculum_bonus
            
            probs = torch.softmax(scores / cfg.temp, dim=0)  # over pool
            
            # Selection: top-k or sampling from categorical
            k = cfg.batch
            if cfg.selection_mode == "topk":
                topk_idx = torch.topk(probs, k=k, dim=0).indices
            elif cfg.selection_mode == "sample":
                topk_idx = torch.multinomial(probs, num_samples=k, replacement=False)
            else:
                raise ValueError(f"Unknown selection_mode: {cfg.selection_mode}")
            
            # Selected batch
            X_sel = X[topk_idx]
            Y_sel = Y[topk_idx]
            
            # ===== Reward definition =====
            with torch.no_grad():
                loss_before = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)  # [k]
            
            # LM update
            logits = model(X_sel)
            loss_lm = loss_fn(logits.reshape(-1, logits.size(-1)), Y_sel.reshape(-1))
            
            opt_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            opt_lm.step()
            
            with torch.no_grad():
                loss_after = compute_loss_per_sample(model, X_sel, Y_sel, loss_fn)  # [k]
            
            if cfg.reward_type == "improvement":
                reward = loss_before - loss_after       # positive = improved
            elif cfg.reward_type == "loss_after":
                reward = -loss_after                    # lower loss_after = higher reward
            elif cfg.reward_type == "random":
                reward = torch.randn_like(loss_after)   # sanity check
            else:
                raise ValueError(f"Unknown reward_type: {cfg.reward_type}")
            
            if cfg.clip_negative_improvement:
                reward = reward.clamp(min=0.0)
            
            if cfg.use_reward_baseline:
                baseline = reward.mean()
                advantage = reward - baseline
            else:
                advantage = reward
            
            sel_probs = probs[topk_idx].clamp_min(1e-12)
            reinforce = -(advantage * torch.log(sel_probs)).mean()
            
            # Entropy regularization (note: ent is negative)
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
                    entropy=-ent.item(),  # convert to positive entropy
                    avg_reward=reward.mean().item(),
                    curriculum_bias_strength=curriculum_strength,
                    **div_metrics
                )
                print(f"[Router:{cfg.experiment_name}] epoch {epoch} step {global_step}: "
                      f"LM={loss_lm.item():.4f}  "
                      f"Reward={reward.mean().item():.4f}  "
                      f"H={-ent.item():.3f}  "
                      f"CurrBias={curriculum_strength:.2f}  "
                      f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}")
        
        # Validation
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Router:{cfg.experiment_name}] epoch {epoch}: "
              f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model, router

# -------------------------
# Training: Baseline (uniform)
# -------------------------
def train_baseline(train_ds, val_ds, metrics, diversity, use_wandb=False):
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.experiment_name,
            group="ablation",
            config={
                "method": "baseline",
                "experiment_name": cfg.experiment_name,
                "batch_size": cfg.batch,
                "pool_size": cfg.pool,
                "epochs": cfg.epochs,
                "lr_lm": cfg.lr_lm,
                "d_model": cfg.d_model,
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "block_size": cfg.block,
                "dataset_size": len(train_ds),
                "seed": cfg.seed
            },
            reinit=True
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
                    entropy=math.log(cfg.batch),  # ideal uniform entropy
                    **div_metrics
                )
                print(f"[Baseline:{cfg.experiment_name}] epoch {epoch} step {global_step}: "
                      f"LM={loss.item():.4f}  "
                      f"Easy={div_metrics['easy_ratio']:.2f} Hard={div_metrics['hard_ratio']:.2f}")
        
        val_loss, val_ppl = evaluate(model, val_ds, loss_fn)
        metrics.log(epoch=epoch, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
        print(f"==> [Baseline:{cfg.experiment_name}] epoch {epoch}: "
              f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model

# -------------------------
# Comparison helper (baseline vs one router)
# -------------------------
def compare_runs(baseline_metrics, router_metrics, router_name, use_wandb=False):
    print("\n" + "="*70)
    print(f"COMPARISON REPORT: baseline vs {router_name}")
    print("="*70)
    
    baseline_ppl = baseline_metrics.get_final_ppl()
    router_ppl = router_metrics.get_final_ppl()
    improvement = ((baseline_ppl - router_ppl) / baseline_ppl) * 100
    
    print(f"\n1. Final Validation Perplexity:")
    print(f"   Baseline: {baseline_ppl:.2f}")
    print(f"   {router_name}: {router_ppl:.2f}")
    print(f"   Improvement: {improvement:+.2f}%")
    
    curriculum_progression = False
    early_easy, late_easy, shift = 0, 0, 0
    if 'easy_ratio' in router_metrics.history:
        easy_ratios = router_metrics.history['easy_ratio']
        if len(easy_ratios) > 0:
            third = max(len(easy_ratios) // 3, 1)
            early_easy = sum(easy_ratios[:third]) / third
            late_easy = sum(easy_ratios[-third:]) / third
            shift = early_easy - late_easy
            
            print(f"\n2. Curriculum Progression ({router_name}):")
            print(f"   Early training easy ratio: {early_easy:.2f}")
            print(f"   Late training easy ratio:  {late_easy:.2f}")
            print(f"   Shift toward hard samples: {shift:.2f}")
            
            if early_easy > late_easy + 0.05:
                print("   ✓ Clear curriculum: easy→hard progression observed")
                curriculum_progression = True
            else:
                print("   ✗ No clear curriculum progression")
    
    def avg_last(metrics, key, n=5):
        vals = metrics.history.get(key, [])
        if len(vals) >= n:
            return sum(vals[-n:]) / n
        return 0
    
    baseline_coverage = avg_last(baseline_metrics, 'coverage')
    router_coverage = avg_last(router_metrics, 'coverage')
    baseline_entropy = avg_last(baseline_metrics, 'entropy')
    router_entropy = avg_last(router_metrics, 'entropy')
    
    print(f"\n3. Sample Diversity (avg last 5 checkpoints):")
    print(f"   Baseline Coverage: {baseline_coverage:.3f}")
    print(f"   {router_name} Coverage:   {router_coverage:.3f}")
    print(f"   Baseline Entropy:  {baseline_entropy:.3f}")
    print(f"   {router_name} Entropy:    {router_entropy:.3f}")
    
    print(f"\n4. Verdict:")
    if improvement > 5 and router_ppl < baseline_ppl:
        verdict = "significant_improvement"
        print("   ✓✓ Router provides significant improvement (>5%)")
    elif improvement > 2 and router_ppl < baseline_ppl:
        verdict = "meaningful_improvement"
        print("   ✓ Router shows meaningful improvement (2-5%)")
    elif improvement > 0:
        verdict = "modest_improvement"
        print("   ~ Router shows modest improvement (<2%)")
    else:
        verdict = "no_improvement"
        print("   ✗ Router does not improve over baseline")
    
    print("\n" + "="*70 + "\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"comparison_{router_name}",
            group="ablation_comparison",
            reinit=True
        )
        
        wandb.log({
            "comparison/baseline_ppl": baseline_ppl,
            "comparison/router_ppl": router_ppl,
            "comparison/improvement_pct": improvement,
            "comparison/early_easy_ratio": early_easy,
            "comparison/late_easy_ratio": late_easy,
            "comparison/curriculum_shift": shift,
            "comparison/curriculum_progression": curriculum_progression,
            "comparison/baseline_coverage": baseline_coverage,
            "comparison/router_coverage": router_coverage,
            "comparison/baseline_entropy": baseline_entropy,
            "comparison/router_entropy": router_entropy,
            "comparison/verdict": verdict
        })
        
        wandb.finish()

# -------------------------
# Main: Ablation Runner
# -------------------------
def main():
    print("="*70)
    print("CURRICULUM LEARNING ABLATION EXPERIMENTS")
    print("="*70)
    print(f"Device: {cfg.device}")
    print()
    
    use_wandb = cfg.use_wandb and WANDB_AVAILABLE
    if cfg.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  wandb requested but not available. Install with: pip install wandb")
        print("   Continuing without wandb logging...\n")
    elif use_wandb:
        print(f"✓ wandb enabled - project: {cfg.wandb_project}\n")
    
    # Load data ONCE (same data for all experiments)
    train_chunks = make_mixed_chunks("train")
    val_chunks = make_mixed_chunks("validation")
    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)
    print(f"\n✓ Train: {len(train_ds)} samples, Val: {len(val_ds)} samples\n")
    
    Path(cfg.save_dir).mkdir(exist_ok=True)
    
    # Define ablation experiments
    # You can comment out experiments you don't want to run
    EXPERIMENTS = [
        # 0. Baseline
        {"name": "baseline_uniform", "kind": "baseline"},
        
        # 1. Full router (current best config)
        {"name": "router_full_improvement", "kind": "router"},
        
        # 2. Reward ablations
        {"name": "router_random_reward", "kind": "router", "reward_type": "random"},
        {"name": "router_loss_after", "kind": "router", "reward_type": "loss_after"},
        {"name": "router_no_clip", "kind": "router", "clip_negative_improvement": False},
        
        # 3. Feature ablations
        {"name": "router_text_stats_only", "kind": "router", "feature_mode": "text_only"},
        {"name": "router_hidden_only", "kind": "router", "feature_mode": "hidden_only"},
        
        # 4. Router architecture
        {"name": "router_mlp_router", "kind": "router", "router_arch": "mlp"},
        
        # 5. Exploration (temperature & entropy)
        {"name": "router_temp_0_5", "kind": "router", "temp": 0.5},
        {"name": "router_temp_2_0", "kind": "router", "temp": 2.0},
        {"name": "router_no_entropy", "kind": "router", "lambda_ent": 0.0},
        {"name": "router_high_entropy", "kind": "router", "lambda_ent": 0.02},
        
        # 6. Curriculum bias (hand-crafted)
        {"name": "router_curriculum_bias", "kind": "router", "curriculum_bias": True},
    ]
    
    baseline_metrics = None
    
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        kind = exp["kind"]  # "baseline" or "router"
        
        print("\n" + "="*70)
        print(f"RUNNING EXPERIMENT: {exp_name}  (kind={kind})")
        print("="*70 + "\n")
        
        # Reset config to defaults, then apply overrides
        base_cfg = Config()
        base_cfg.pool = base_cfg.pool_mult * base_cfg.batch
        
        # Copy fields into global cfg
        for k, v in base_cfg.__dict__.items():
            setattr(cfg, k, v)
        cfg.pool = cfg.pool_mult * cfg.batch
        
        # Apply experiment-specific overrides
        for k, v in exp.items():
            if k in ("name", "kind"):
                continue
            setattr(cfg, k, v)
        cfg.experiment_name = exp_name
        
        # Reset seeds for comparability
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        
        metrics = MetricsTracker(use_wandb=use_wandb)
        diversity = DiversityTracker(len(train_ds))
        
        if kind == "baseline":
            model = train_baseline(train_ds, val_ds, metrics, diversity, use_wandb=use_wandb)
            metrics.save(f"{cfg.save_dir}/{exp_name}_metrics.json")
            baseline_metrics = metrics
        elif kind == "router":
            model, router = train_router(train_ds, val_ds, metrics, diversity, use_wandb=use_wandb)
            metrics.save(f"{cfg.save_dir}/{exp_name}_metrics.json")
            if baseline_metrics is not None and exp_name == "router_full_improvement":
                compare_runs(baseline_metrics, metrics, router_name=exp_name, use_wandb=use_wandb)
        else:
            raise ValueError(f"Unknown experiment kind: {kind}")
    
    print("\nAll experiments finished.")
    print(f"Metrics saved under {cfg.save_dir}/")

if __name__ == "__main__":
    main()
