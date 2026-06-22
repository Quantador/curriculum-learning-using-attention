# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo layout and add comprehensive documentation across all source files to onboard a new ML research collaborator.

**Architecture:** Three sequential phases — (1) structural changes (archive, move, rename), (2) full commenting pass per file, (3) README. Each structural task is verified by running `smoke_test.py`. Commenting tasks add documentation only; smoke_test.py serves as the safety net.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `datasets`/`transformers`, W&B (optional)

## Global Constraints

- No logic changes to any Python file (import renames only in Task 3)
- All `clean/` files use flat imports (`from config import ...`) — no path changes needed when moving to root
- `smoke_test.py` must pass after every task
- Working directory for all commands: repo root `curriculum-learning-using-attention/`
- Note: `extract_hierarchical_features` IS a real function in `model.py` (line 136) — the import in `training.py` is correct, no fix needed

---

### Task 1: Archive legacy files

Move `TESTS/` (old prototypes) and `old_results/` (historical JSON metrics) to `archive/` so they are preserved but clearly not active code.

**Files:**
- Create: `archive/TESTS/` (moved from `TESTS/`)
- Create: `archive/old_results/` (moved from `old_results/`)

- [ ] **Step 1: Create archive directory and move directories**

```bash
mkdir archive
git mv TESTS archive/TESTS
git mv old_results archive/old_results
```

- [ ] **Step 2: Verify**

```bash
ls archive/
```
Expected: `TESTS  old_results`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: move TESTS/ and old_results/ to archive/ (legacy prototypes and old experiment data)"
```

---

### Task 2: Move clean/ contents to repo root

All working source lives in `clean/`. Moving it to root makes `python experiments.py`, `python compare.py`, etc. work from the project root without `cd clean/` first. All files use flat imports so no import paths change.

**Files:**
- Move: all files under `clean/` → root
- Delete: `clean/` directory

- [ ] **Step 1: Move Python source files**

```bash
git mv clean/config.py config.py
git mv clean/data.py data.py
git mv clean/model.py model.py
git mv clean/metrics.py metrics.py
git mv clean/training.py training.py
git mv clean/modelExperiments.py modelExperiments.py
git mv clean/trainingExperiments.py trainingExperiments.py
git mv clean/baselineVSrouter.py baselineVSrouter.py
git mv clean/test_run.py test_run.py
git mv clean/experiments.py experiments.py
```

- [ ] **Step 2: Move remaining files**

```bash
git mv clean/EXPERIMENTS.md EXPERIMENTS.md
git mv clean/run_experiment.ipynb run_experiment.ipynb
```

- [ ] **Step 3: Remove the now-empty clean/ directory**

```bash
git rm -r --cached clean/ 2>/dev/null || true
rmdir clean 2>/dev/null || true
```

- [ ] **Step 4: Verify root structure**

```bash
ls *.py
```
Expected: `baselineVSrouter.py  config.py  data.py  experiments.py  metrics.py  model.py  modelExperiments.py  test_run.py  trainingExperiments.py  training.py`

- [ ] **Step 5: Verify imports still work**

```bash
python -c "from config import ExperimentConfig; from model import TinyGPT; print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: move clean/ contents to repo root"
```

---

### Task 3: Rename files and update all imports

Five files get clearer names. Every import of an old name must be updated in the same commit to avoid a broken intermediate state.

**Files renamed:**
- `modelExperiments.py` → `router.py`
- `trainingExperiments.py` → `rl_training.py`
- `baselineVSrouter.py` → `compare.py`
- `test_run.py` → `smoke_test.py`
- `visu.py` → `visualize.py`

**Files with imports to update:**
- `experiments.py` (imports `modelExperiments` and `trainingExperiments`)
- `rl_training.py` (imports `modelExperiments`)
- `compare.py` (imports `modelExperiments` and `trainingExperiments`)
- `smoke_test.py` (imports `modelExperiments` and `trainingExperiments`)

- [ ] **Step 1: Rename files with git mv**

```bash
git mv modelExperiments.py router.py
git mv trainingExperiments.py rl_training.py
git mv baselineVSrouter.py compare.py
git mv test_run.py smoke_test.py
git mv visu.py visualize.py
```

- [ ] **Step 2: Update imports in experiments.py**

Change:
```python
from modelExperiments import *
from trainingExperiments import *
```
To:
```python
from router import *
from rl_training import *
```

- [ ] **Step 3: Update imports in rl_training.py**

Change:
```python
from modelExperiments import extract_router_features
```
To:
```python
from router import extract_router_features
```

- [ ] **Step 4: Update imports in compare.py**

Change:
```python
from modelExperiments import build_router, get_router_feature_dim
from trainingExperiments import train_aux_baseline
```
To:
```python
from router import build_router, get_router_feature_dim
from rl_training import train_aux_baseline
```

- [ ] **Step 5: Update imports in smoke_test.py**

Change:
```python
from modelExperiments import build_router, get_router_feature_dim
from trainingExperiments import train_aux_baseline
```
To:
```python
from router import build_router, get_router_feature_dim
from rl_training import train_aux_baseline
```

- [ ] **Step 6: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: rename files for clarity and update all imports"
```

---

### Task 4: Comment config.py

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add module docstring at the top of config.py (after the `# config.py` line)**

```python
"""
Central configuration for all curriculum learning experiments.

Two dataclasses:
  - Config: base training hyperparameters (model size, batch, lr, etc.)
  - ExperimentConfig: extends Config with all experiment-specific knobs —
    router architecture, training algorithm (REINFORCE/GRPO/PPO), reward
    signal, entropy formulation, dataset choices, coverage regularization,
    PPO/GRPO params, and feature caching settings.

Typical usage:
    cfg = ExperimentConfig()           # sensible defaults
    cfg = replace(cfg, epochs=5, ...)  # override via dataclasses.replace
"""
```

- [ ] **Step 2: Add class docstring to ExperimentConfig (immediately after `class ExperimentConfig(Config):`)**

```python
    """
    Extends Config with all experiment-specific hyperparameters.

    All fields have sensible defaults. Override via dataclasses.replace()
    to generate ablation configurations without mutating the base config.

    Key field groups:
      - Dataset: easy_dataset/hard_dataset for mixed mode, single_dataset
        for single-source mode; use_external_embeddings for pre-computed vectors
      - Router: router_architecture ('attention', 'mlp', 'linear'),
        router_n_heads (>1 enables MultiHeadAttentionRouter)
      - Training algorithm: training_algorithm ('reinforce', 'grpo', 'ppo')
      - Reward: reward_signal (8 options documented in field comments)
      - Entropy: entropy_type, entropy_schedule, use_entropy_targeting
      - Coverage: use_coverage_regularization, coverage_type
      - Caching: feature_cache_epochs (0 = disabled)
    """
```

- [ ] **Step 3: Verify**

```bash
python -c "from config import ExperimentConfig; print(ExperimentConfig.__doc__[:50])"
```
Expected: prints the start of the class docstring without error.

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "docs: add module and class docstrings to config.py"
```

---

### Task 5: Comment data.py

**Files:**
- Modify: `data.py`

- [ ] **Step 1: Add module docstring at the top of data.py (after `# data.py`)**

```python
"""
Dataset loading, tokenisation, and chunking for curriculum learning.

Two dataset modes:
  - Mixed-difficulty: load an easy + a hard HuggingFace dataset, tokenise,
    chunk into (block+1)-token sequences, label them 0 (easy) / 1 (hard).
    Use make_mixed_chunks().
  - Single-dataset: one source, no difficulty split (all labels 0). Supports
    optional pre-computed external embeddings (e.g. epfml/FineWeb-HQ).
    Use make_single_chunks().

Difficulty label convention: 0 = easy, 1 = hard, -1 = validation (no label).

Validation always uses WikiText-2 regardless of training dataset config,
keeping the eval set fixed across all experiments for fair comparison.

Key exports:
  get_tokenizer()       — shared GPT-2 BPE tokeniser
  make_mixed_chunks()   — builds labelled train/val chunks for mixed mode
  make_single_chunks()  — builds chunks for single-dataset mode
  MixedLMDataset        — PyTorch Dataset yielding (x, y, difficulty) triples
  make_index_loader()   — yields shuffled pool-sized index batches
"""
```

- [ ] **Step 2: Add comment explaining difficulty labels in MixedLMDataset.__init__**

After the line `self.difficulty = [d for _, d in labeled_chunks]`, add:

```python
        # Labels: 0 = easy, 1 = hard, -1 = validation set (no curriculum label).
```

- [ ] **Step 3: Add comment in make_mixed_chunks explaining the hardcoded validation set**

At the start of the `else` branch (validation path) in `make_mixed_chunks`, add:

```python
        # Validation always uses WikiText-2, not the configured training datasets.
        # This keeps the eval signal identical across all experiment variants.
```

- [ ] **Step 4: Update docstring of load_dataset_with_embeddings**

The function already has a one-line docstring. Expand it to:

```python
    """
    Load a HuggingFace dataset that has a pre-computed embedding column.

    Some datasets (e.g. epfml/FineWeb-HQ) store one embedding vector per
    sub-chunk of the document, yielding shape [n_sub_chunks, dim]. These
    are mean-pooled to a single document-level vector before being attached
    to each token chunk produced from that document.

    Returns (texts, embeddings) where embeddings[i] is a 1-D float32 tensor
    aligned with texts[i].
    """
```

- [ ] **Step 5: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 6: Commit**

```bash
git add data.py
git commit -m "docs: add module docstring and inline comments to data.py"
```

---

### Task 6: Comment model.py

**Files:**
- Modify: `model.py`

- [ ] **Step 1: Add module docstring at the top of model.py (after `# model.py`)**

```python
"""
Model and router architecture definitions.

TinyGPT:
  Small causal language model built on PyTorch's TransformerEncoder with a
  causal attention mask (upper-triangular -inf). Shares weights between the
  token embedding and the LM head (weight tying). Used as the student LM
  in all experiments.

Router architectures (all produce a scalar score [B] per sample in the pool):
  AttentionRouter          — single (projection, query) pair; the baseline router
  MultiHeadAttentionRouter — n independent heads, scores averaged across heads

Feature extraction utilities:
  compute_text_statistics()     — 4 cheap surface-level features: sequence fill
                                  ratio, lexical diversity, mean/std token id
  extract_hierarchical_hidden() — transformer hidden states, chunked & pooled
  extract_hierarchical_features() — combines the above two (legacy helper used
                                    by training.py's reference router loop)
"""
```

- [ ] **Step 2: Add docstring to TinyGPT class**

Insert immediately after `class TinyGPT(nn.Module):`:

```python
    """
    Small causal GPT-style language model (decoder-only transformer).

    Uses nn.TransformerEncoderLayer with an upper-triangular causal mask to
    simulate autoregressive decoding. Weight tying: lm_head.weight == tok_embed.weight,
    halving the effective parameter count and stabilising training.

    forward_to_hidden(x) exposes the transformer hidden states without computing
    logits — used by extract_hierarchical_hidden() for feature extraction without
    a second full forward pass.
    """
```

- [ ] **Step 3: Add docstring to AttentionRouter**

Insert after `class AttentionRouter(nn.Module):`:

```python
    """
    Single-head attention-based sample scorer.

    Learns a linear projection W ∈ R^{d_input × d_k} and a query vector
    q ∈ R^{d_k}. For a batch of feature vectors F ∈ R^{B × d_input}:
        scores = (F @ W^T) @ q  ∈ R^B

    Equivalent to a single-head cross-attention where F are the keys and q
    is the query. This is the default/baseline router architecture.
    """
```

- [ ] **Step 4: Add docstring to compute_text_statistics**

Replace the bare function definition with one that includes a docstring:

```python
def compute_text_statistics(
    X: torch.Tensor,
    pad_token_id: int,
    vocab_size: int,
    block: int,
) -> torch.Tensor:
    """
    Compute 4 cheap surface-level text features per sample.

    Returns a [B, 4] tensor. Column semantics:
      [0] relative_length — non-pad tokens / block  (sequence fill ratio)
      [1] unique_ratio    — unique tokens / sequence length  (lexical diversity)
      [2] avg_token       — mean token id / vocab_size  (normalized)
      [3] std_token       — std of token ids / vocab_size  (normalized)

    All values are in [0, 1]. These four statistics are fast to compute
    (no transformer forward pass) and capture coarse difficulty signals:
    longer, more diverse sequences with unusual token distributions tend to
    be harder for the model to predict.
    """
```

- [ ] **Step 5: Add docstring to extract_hierarchical_hidden**

```python
def extract_hierarchical_hidden(
    model: TinyGPT,
    X: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Extract chunked, mean-pooled hidden states from TinyGPT.

    The sequence of length L is divided into cfg.n_chunks equal segments.
    Each segment's hidden states are mean-pooled to a single d_model vector.
    The n_chunks vectors are concatenated to produce [B, n_chunks * d_model].

    Chunking captures positional structure: early chunks encode document
    start (typically more predictable), later chunks encode content density.
    This is richer than a single mean-pool over the whole sequence.

    Two modes (cfg.hierarchical_representation):
      'full'     — uses full transformer hidden states (one LM forward pass)
      'embedder' — uses only token + positional embeddings, no transformer
                   (~10× faster but loses contextual information)

    Always runs under torch.no_grad() — never affects LM gradients.
    """
```

- [ ] **Step 6: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 7: Commit**

```bash
git add model.py
git commit -m "docs: add module, class, and function docstrings to model.py"
```

---

### Task 7: Comment training.py

**Files:**
- Modify: `training.py`

- [ ] **Step 1: Add module docstring at the top of training.py (after `# training.py`)**

```python
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
```

- [ ] **Step 2: Add docstring to evaluate**

Insert after `def evaluate(`:

```python
    """
    Compute mean cross-entropy loss and perplexity over the full dataset.

    Iterates sample-by-sample (not batched) to avoid padding artefacts.
    Sets model to eval() before the loop and restores train() after.

    Returns (avg_loss, perplexity) where perplexity = exp(avg_loss).
    """
```

- [ ] **Step 3: Add docstring to train_baseline**

Insert after `def train_baseline(`:

```python
    """
    Train TinyGPT with uniform random batch selection (no curriculum).

    At each step, draws cfg.batch samples uniformly at random from a pool
    of cfg.pool candidates (pool_mult × batch). This is the control condition —
    it sets the performance floor that the router should beat.
    """
```

- [ ] **Step 4: Add docstring to train_router**

Insert after `def train_router(`:

```python
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
```

- [ ] **Step 5: Add inline comment explaining the entropy term in train_router**

After the line `ent = (probs * probs.clamp_min(1e-12).log()).sum()`, add:

```python
            # ent = sum(p * log p) = -H(p), the *negative* Shannon entropy.
            # Adding lambda_ent * ent to the loss penalises low-entropy distributions,
            # so minimising the total loss pushes the router toward diverse selection.
```

- [ ] **Step 6: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 7: Commit**

```bash
git add training.py
git commit -m "docs: add module docstring, function docstrings, and inline comments to training.py"
```

---

### Task 8: Comment router.py

**Files:**
- Modify: `router.py`

- [ ] **Step 1: Add module docstring at the top of router.py (after `from __future__ import annotations`)**

```python
"""
Router factory and feature extraction for curriculum learning experiments.

This module bridges the language model and the RL training loop: it defines
how samples are featurized and which router architecture scores them.

Router architectures (all nn.Module, produce [B] scalar scores):
  LinearRouter   — single linear layer, fewest parameters, fastest
  MLPRouter      — two-layer MLP with GELU, more expressive
  AuxNetRouter   — supervised alternative trained with MSE to predict
                   loss improvement (not policy gradient)

The primary router architectures (AttentionRouter, MultiHeadAttentionRouter)
are defined in model.py. build_router() here is the factory for all of them.

Feature extraction:
  extract_router_features() — concatenates up to three feature groups into
                              the vector fed to the router
  get_router_feature_dim()  — computes the expected input dimension so the
                              router can be instantiated before training starts
"""
```

- [ ] **Step 2: Replace AuxNetRouter docstring with an expanded version**

```python
    """
    Supervised alternative to the RL router — used as an ablation baseline.

    Trained with MSE loss to directly regress the observed per-sample
    loss-improvement signal, rather than via policy gradient. At inference
    time it scores samples identically to the attention router (top-k by
    predicted score), making it a controlled comparison:
      - Same features (output of extract_router_features)
      - Same selection logic (top-k in rl_training.train_aux_baseline)
      - Different training objective: MSE regression vs. REINFORCE

    Instantiate via build_router(arch='auxnet').
    Training loop: rl_training.train_aux_baseline().
    """
```

- [ ] **Step 3: Add docstring to extract_router_features**

```python
def extract_router_features(
    model: TinyGPT,
    X: torch.Tensor,
    cfg: ExperimentConfig,
    pad_token_id: int,
    vocab_size: int,
    external_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build the feature vector that the router scores each candidate sample with.

    Concatenates up to three optional feature groups in this order:
      1. Hierarchical hidden states  [B, n_chunks * d_model]
         Enabled by cfg.enable_text_hierarchical. Runs a transformer forward
         pass (or uses only embeddings if cfg.hierarchical_representation='embedder').
      2. Text statistics  [B, 4]
         Enabled by cfg.enable_text_stat. Cheap surface features: fill ratio,
         lexical diversity, normalised mean/std token id.
      3. Pre-computed external embeddings  [B, external_embedding_dim]
         Used when cfg.use_external_embeddings=True and external_emb is provided.

    If no group is enabled, returns random features as a fallback
    (router learns nothing — intended only for sanity-check baselines).

    Returns [B, F] where F == get_router_feature_dim(cfg).
    """
```

- [ ] **Step 4: Add docstring to get_router_feature_dim**

```python
def get_router_feature_dim(cfg: ExperimentConfig) -> int:
    """
    Compute the router's expected input dimensionality from config flags.

    Mirrors the concatenation order in extract_router_features():
      n_chunks * d_model   if enable_text_hierarchical  (hierarchical hidden)
      + 4                  if enable_text_stat           (text statistics)
      + external_dim       if use_external_embeddings    (external embeddings)

    When both hierarchical and stat flags are False, returns the full fallback
    dimension n_chunks * d_model + 4 to match the random-feature path in
    extract_router_features().
    """
```

- [ ] **Step 5: Add docstring to build_router**

```python
def build_router(
    d_input: int,
    arch: str = "attention",
    d_k: int = 128,
    d_hidden: int = 256,
    n_heads: int = 1,
) -> nn.Module | None:
    """
    Factory for all router architectures.

    Args:
        d_input:  Input feature dimensionality. Pass get_router_feature_dim(cfg).
        arch:     Architecture name:
                    'attention' — AttentionRouter (n_heads=1) or
                                  MultiHeadAttentionRouter (n_heads > 1)
                    'linear'   — single linear projection
                    'mlp'      — two-hidden-layer MLP with GELU
                    'auxnet'   — supervised AuxNetRouter (MSE training)
                    'random'   — returns None; training falls back to random scores
        d_k:      Key/query dimension for attention routers.
        d_hidden: Hidden dimension for MLP/auxnet routers.
        n_heads:  Attention heads (attention arch only; >1 enables multi-head).

    Returns an nn.Module or None (for arch='random').
    """
```

- [ ] **Step 6: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 7: Commit**

```bash
git add router.py
git commit -m "docs: add module docstring and function docstrings to router.py"
```

---

### Task 9: Comment rl_training.py

This is the most complex file and the most important one to document for a new collaborator.

**Files:**
- Modify: `rl_training.py`

- [ ] **Step 1: Add module docstring at the top of rl_training.py (after `from __future__ import annotations`)**

```python
"""
Advanced RL-based training loops for curriculum learning experiments.

This is the experiment-grade router training loop used by experiments.py
(ablation studies) and compare.py (single runs with all variants enabled).

Key features over the reference loop in training.py:
  - Three policy gradient algorithms: REINFORCE, GRPO, PPO
  - Eight reward signals: loss_improvement, neg_loss, relative_improvement,
    difficulty_weighted, uncertainty_reduction, gradient_norm,
    gradient_alignment, combined
  - Four entropy formulations: Shannon, Rényi, Tsallis, KL-uniform
  - SAC-style entropy targeting (auto-adjusts lambda_ent to hit a target entropy)
  - Coverage regularisation (penalises repeated sample selection)
  - Feature caching (amortises expensive transformer forward passes)
  - Supervised aux-net baseline (MSE alternative to policy gradient)

Entropy sign convention — READ THIS:
  All compute_*_entropy() functions return -H, the *negative* entropy.
  Adding `lambda_ent * entropy_term` to the router loss therefore
  penalises low-entropy distributions: minimising the total loss
  *maximises* entropy and encourages diverse sample selection.
  The logged 'entropy' value is always negated before display so the
  dashboard shows a positive, human-readable entropy number.

Entry points:
  train_router_experiments() — main RL training loop (REINFORCE/GRPO/PPO)
  train_aux_baseline()       — supervised MSE alternative
  compare_runs_experiments() — prints a performance comparison table
"""
```

- [ ] **Step 2: Add comment block before the inner training loop in train_router_experiments**

Find `for pool_indices in tqdm(idx_loader):` and insert a comment block before it:

```python
        # ── Per-step curriculum loop ──────────────────────────────────────────
        # Each iteration implements the core curriculum learning cycle:
        #   1. Sample M = cfg.pool candidate indices (pre-shuffled each epoch).
        #   2. Extract router features for all M samples.
        #   3. Router scores pool → softmax(/ temp) → select k = cfg.batch samples.
        #   4. LM forward + backward on selected batch.
        #   5. Compute reward signal (loss improvement, gradient norm, etc.).
        #   6. Router RL update (REINFORCE / GRPO / PPO + entropy regularisation).
        # ─────────────────────────────────────────────────────────────────────
```

- [ ] **Step 3: Add comment explaining why the feature cache skips epoch 0**

Find the block:
```python
        if (
            cfg.feature_cache_epochs > 0
            and cfg.enable_text_hierarchical
            and epoch > 0
```
Insert a comment immediately before this `if` block:

```python
        # The feature cache is never built at epoch 0: the model's weights are
        # randomly initialised, so the hidden states are noise. Caching garbage
        # features would waste memory and mislead the router. Rebuilding every
        # feature_cache_epochs epochs (starting at epoch 1) keeps the cache
        # fresh as the model's representations improve.
```

- [ ] **Step 4: Add comment explaining gradient reward timing**

Find the call to `compute_gradient_reward(...)` and insert a comment before it:

```python
                # Gradient reward is computed AFTER loss_lm.backward() populates
                # .grad on all parameters but BEFORE opt_lm.step() zeroes them.
                # This window is the only point where the raw batch gradients exist.
```

- [ ] **Step 5: Add comment explaining the entropy negation in the logging block**

Find `"entropy": -entropy.item(),` and add an inline comment:

```python
                    "entropy": -entropy.item(),  # entropy is -H; negate to log positive H
```

- [ ] **Step 6: Add docstring to get_scheduled_value**

```python
def get_scheduled_value(
    schedule: str,
    initial: float,
    minimum: float,
    progress: float,
    step: int = 0,
    cycle_length: int = 1000,
) -> float:
    """
    Return a scheduled hyperparameter value at the given training progress.

    Used for temperature annealing (cfg.temp_schedule) and entropy coefficient
    annealing (cfg.entropy_schedule). All schedules interpolate from `initial`
    at progress=0.0 to `minimum` at progress=1.0.

    Schedules:
      'fixed'             — constant initial throughout training
      'linear_decay'      — linear interpolation from initial to minimum
      'cosine_decay'      — cosine annealing (smooth S-curve decay)
      'exponential_decay' — fast initial drop, slower tail
      'cyclic'            — cosine warm restarts every cycle_length steps
      'adaptive'          — placeholder, returns initial (not implemented)
    """
```

- [ ] **Step 7: Add docstring to reinforce_update**

Insert after `def reinforce_update(`:

```python
    """
    Standard REINFORCE (vanilla policy gradient) router update.

    Advantage = reward - baseline (reduces gradient variance).
    Policy loss = -mean(advantage * log_prob_of_selected_samples).
    Total loss = policy_loss + lambda_ent * entropy_term.

    See module docstring for the entropy sign convention.

    Returns (loss_router, reinforce_loss, entropy) where entropy = -H.
    """
```

- [ ] **Step 8: Add docstring to grpo_update**

Insert after `def grpo_update(`:

```python
    """
    Group Relative Policy Optimization (GRPO) router update.

    Instead of a single global baseline, advantages are normalised within
    small groups of group_size samples:
        advantage_i = (r_i - group_mean) / group_std
    This provides lower-variance gradient estimates when rewards vary
    substantially across samples, without needing a learned value function.

    Returns (loss_router, grpo_loss, entropy) where entropy = -H.
    """
```

- [ ] **Step 9: Add docstring to ppo_update**

Insert after `def ppo_update(`:

```python
    """
    Proximal Policy Optimization (PPO) router update.

    Runs cfg.ppo_epochs inner update steps with the clipped surrogate:
        L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    where ratio = new_log_prob / old_log_prob and ε = cfg.ppo_clip.
    Clipping prevents destructively large policy updates in a single step.

    Advantages are normalised across the selected batch before clipping.
    Coverage loss is applied only on the first inner epoch to avoid
    double-counting the coverage penalty.

    Returns averaged (loss_router, policy_loss, entropy) over inner steps,
    where entropy = -H.
    """
```

- [ ] **Step 10: Add docstring to build_feature_cache**

```python
def build_feature_cache(
    model: TinyGPT,
    train_ds: MixedLMDataset,
    cfg: ExperimentConfig,
) -> torch.Tensor:
    """
    Pre-compute and cache hierarchical hidden features for the full training set.

    Running a full transformer forward pass over M pool samples at every step
    is the dominant cost when enable_text_hierarchical=True. This function
    amortises that cost by running the model once over the entire dataset,
    storing the result as fp16 on CPU, and reusing it for feature_cache_epochs
    epochs before rebuilding.

    Cache validity: if the stored shape or dtype does not match expectations
    (e.g. after changing d_model or n_chunks), the cache is discarded and rebuilt.

    Disk persistence: if cfg.feature_cache_path is non-empty, the cache is saved
    as a .pt file and loaded on the next call instead of recomputing.

    Returns: [N, n_chunks * d_model] fp16 CPU tensor.
    """
```

- [ ] **Step 11: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 12: Commit**

```bash
git add rl_training.py
git commit -m "docs: add comprehensive module docstring and inline comments to rl_training.py"
```

---

### Task 10: Comment experiments.py

**Files:**
- Modify: `experiments.py`

- [ ] **Step 1: Add module docstring at the top of experiments.py (after `# run_experiment.py`)**

```python
"""
Ablation study orchestration for curriculum learning experiments.

Three experiment modes:
  1. One-factor-at-a-time ablation (--all flag, default):
     For each field in EXPERIMENTAL_FIELDS, generates one config per
     alternative value. Every config is identical to the baseline except
     for exactly ONE field change, isolating the effect of each design choice.
     Produces ~37 experiments total.

  2. Full grid search (--combinations flag):
     All combinations of all field values. Grows exponentially — use only
     for a small subset of fields (e.g. 2-3 fields at most).

  3. Predefined profiles (--profile flag):
     Curated subsets defined in EXPERIMENT_PROFILES for focused runs
     (e.g. 'final_presentation', 'additional_experiments').

EXPERIMENTAL_FIELDS format:
  { "field_name": (baseline_value, [alternative_values]), ... }

Results are saved to cfg.save_dir/<experiment_name>.json.
See EXPERIMENTS.md for the full CLI reference and field descriptions.
"""
```

- [ ] **Step 2: Add comment before EXPERIMENTAL_FIELDS**

```python
# Maps each experimental dimension to (baseline_value, [alternative_values]).
# The baseline_value is used in the control experiment (experiment_name="baseline").
# Each alternative generates one experiment that changes only this single field.
# This one-factor-at-a-time design lets us isolate the effect of each choice.
```

- [ ] **Step 3: Add docstring to generate_experiment_configs**

```python
    """
    Generate ablation (one-factor-at-a-time) experiment configurations.

    For each field in experimental_fields, generates one ExperimentConfig per
    alternative value. Each config is identical to the baseline except for
    exactly ONE field — this isolates each design choice cleanly.

    Example: baseline uses (ppo, loss_improvement, topk). Varying
    training_algorithm yields two configs: one with 'grpo', one with
    'reinforce', both with all other fields at baseline values.

    Args:
        base_cfg:            Starting config (defaults to ExperimentConfig()).
        experimental_fields: {field: (baseline, [alternatives])} mapping.
        include_baseline:    Whether to prepend the all-baseline config first.

    Returns a list of ExperimentConfig with descriptive experiment_name fields.
    """
```

- [ ] **Step 4: Add docstring to generate_combination_configs**

```python
    """
    Generate all combinations of experimental field values (full grid search).

    Produces the Cartesian product of all field value lists. Grows exponentially:
    3 fields × 3 values each = 27 experiments; 10 fields = potentially thousands.
    Only use this for small, targeted subsets of fields.

    Args:
        base_cfg:            Starting config (defaults to ExperimentConfig()).
        experimental_fields: {field: [values]} mapping (flat lists, no baseline tuple).

    Returns a list of ExperimentConfig, one per combination.
    """
```

- [ ] **Step 5: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 6: Commit**

```bash
git add experiments.py
git commit -m "docs: add module docstring and function docstrings to experiments.py"
```

---

### Task 11: Comment compare.py, smoke_test.py, metrics.py, visualize.py

**Files:**
- Modify: `compare.py`, `smoke_test.py`, `metrics.py`, `visualize.py`

- [ ] **Step 1: Add module docstring to compare.py (at the top, after `# baselineVSrouter.py`)**

```python
"""
Entry point for a single three-way comparison run.

Trains and evaluates three models on the same dataset and reports final
validation perplexity for each:
  - Baseline:  uniform random batch selection (training.train_baseline)
  - RL Router: attention router trained with REINFORCE (training.train_router)
  - Aux-net:   supervised MSE router (rl_training.train_aux_baseline)
               only runs when cfg.run_aux_baseline=True

Results are printed by compare_runs() and saved to cfg.save_dir/.
For running many ablation experiments sequentially, see experiments.py.
"""
```

- [ ] **Step 2: Add module docstring to smoke_test.py (replace or extend the existing one)**

The file already has a triple-quoted module docstring. Replace it with:

```python
"""
Smoke tests for all major training code paths.

Runs 1 epoch on tiny in-memory configurations (small model, few samples,
no W&B) to verify that every code path executes end-to-end without error.
These are crash tests, not correctness tests — any result > 1.0 perplexity
is accepted.

Run after any structural change:
    python smoke_test.py

Tests:
  1. Default mixed datasets (TinyStories + OpenWebText2)
  2. Custom datasets (WikiText easy + ML-ArXiv hard)
  3. Multi-head router (n_heads = 2 and 4)
  4. Single-dataset mode (OpenWebText2, no easy/hard split)
  5. Auxiliary network baseline (supervised MSE router)
  6. Full three-way comparison (baseline + router + aux-net)
"""
```

- [ ] **Step 3: Add module docstring to metrics.py (at the top, after `# metrics.py`)**

```python
"""
Metric tracking and logging utilities.

MetricsTracker:
  Accumulates scalar metrics as Python lists in self.history.
  Optionally logs each step to Weights & Biases (if installed and enabled).
  Serialises to / loads from JSON for persistence between training runs.
  Used by every training loop in this project.

DiversityTracker:
  Records which samples (by dataset index) were selected at each step.
  Computes coverage (fraction of dataset ever seen), easy/hard selection
  ratio (curriculum direction), unique_ratio (short-window diversity), and
  balance_std (selection uniformity across the dataset).
  Logged every cfg.log_every steps via MetricsTracker.log().
"""
```

- [ ] **Step 4: Add docstring to MetricsTracker.log**

```python
        """Log scalar metrics to history and optionally to W&B.

        Non-numeric values are silently skipped (e.g. epoch strings).
        """
```

- [ ] **Step 5: Add docstring to MetricsTracker.load**

```python
        """Load a MetricsTracker from a JSON file previously saved by .save()."""
```

- [ ] **Step 6: Add docstring to DiversityTracker.get_metrics**

```python
        """
        Return a dict of diversity metrics accumulated since instantiation.

        Keys:
          coverage      — fraction of dataset samples selected at least once [0,1]
          balance_std   — std of selection counts (lower = more uniform coverage)
          unique_ratio  — fraction of unique samples in the last `window` steps [0,1]
          easy_ratio    — fraction of selected samples with difficulty=0 [0,1]
          hard_ratio    — fraction of selected samples with difficulty=1 [0,1]
        """
```

- [ ] **Step 7: Update module docstring in visualize.py**

The file already has a module docstring. Replace it with:

```python
"""
Visualization utilities for curriculum learning experiments.

Loads baseline and router metrics JSON files (written by MetricsTracker.save)
from a results directory and generates five comparison plots:
  1. Validation perplexity over epochs (baseline vs router)
  2. Easy vs hard sample selection ratio over training (curriculum progression)
  3. Average loss improvement per step (router learning signal quality)
  4. Dataset coverage and selection entropy over training
  5. Final performance bar chart with improvement annotation

Usage:
    python visualize.py --results_dir results/

Plots are saved as PNG files under results/plots/.
A text summary report is also written to results/plots/summary_report.txt.
"""
```

- [ ] **Step 8: Run smoke_test.py**

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

- [ ] **Step 9: Commit**

```bash
git add compare.py smoke_test.py metrics.py visualize.py
git commit -m "docs: add module docstrings and function docstrings to compare.py, smoke_test.py, metrics.py, visualize.py"
```

---

### Task 12: Write README.md

**Files:**
- Modify: `README.md` (currently a one-line title)

- [ ] **Step 1: Write the full README**

Replace the entire contents of `README.md` with:

````markdown
# Curriculum Learning Using Attention

A research framework for training a small language model (TinyGPT) with
**curriculum learning**: instead of drawing random batches, a learned router
scores a candidate pool of samples at each training step and selects those
predicted to yield the highest learning signal.

The router is trained with **reinforcement learning** — it receives a reward
(e.g. per-sample loss improvement after the LM update) and optimises its
selection policy via REINFORCE, GRPO, or PPO. A supervised aux-net baseline
(MSE regression on the same reward signal) is included for controlled comparison.

The project is structured as a flexible ablation framework: every major design
choice (router architecture, reward signal, training algorithm, entropy
formulation, dataset difficulty) is swappable via config flags and can be
evaluated in a systematic one-factor-at-a-time study.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Pool of M candidates  (M = pool_mult × batch_size)     │
│  drawn from the training dataset each step              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  Feature extraction     │
         │  · hierarchical hidden  │   ← transformer hidden states,
         │  · text statistics      │     chunked & mean-pooled
         │  · external embeddings  │   ← optional pre-computed vectors
         └──────────┬──────────────┘
                    │  [M, F]
                    ▼
         ┌─────────────────────────┐
         │  Router                 │   ← AttentionRouter / MLP / Linear
         │  → scalar score/sample  │
         └──────────┬──────────────┘
                    │  scores [M]
                    ▼
         ┌─────────────────────────┐
         │  Selection strategy     │   ← top-k / sample / ε-greedy
         └──────────┬──────────────┘
                    │  selected batch [B]
                    ▼
         ┌─────────────────────────┐
         │  TinyGPT forward +      │
         │  backward (LM update)   │
         └──────────┬──────────────┘
                    │  loss before / after
                    ▼
         ┌─────────────────────────┐
         │  Reward signal          │   ← loss improvement, gradient norm, ...
         └──────────┬──────────────┘
                    │  reward [B]
                    ▼
         ┌─────────────────────────┐
         │  Router RL update       │   ← REINFORCE / GRPO / PPO
         └─────────────────────────┘
```

---

## File Map

| File | Role |
| --- | --- |
| `config.py` | `Config` and `ExperimentConfig` dataclasses — all hyperparameters |
| `data.py` | Dataset loading, tokenisation, chunking; `MixedLMDataset` PyTorch dataset |
| `model.py` | `TinyGPT` (causal LM), router architectures, feature extraction utilities |
| `metrics.py` | `MetricsTracker` (logging + JSON), `DiversityTracker` (coverage stats) |
| `training.py` | Reference training loops: `train_baseline`, `train_router` (simplified) |
| `router.py` | Router factory `build_router()`, `extract_router_features()`, `AuxNetRouter` |
| `rl_training.py` | Full RL training loop with all configurable algorithms and schedules |
| `experiments.py` | Ablation orchestration: generates and runs one-factor-at-a-time configs |
| `compare.py` | Entry point for a single baseline vs router vs aux-net comparison |
| `smoke_test.py` | 6 end-to-end smoke tests — run after any structural change |
| `visualize.py` | Plots validation PPL, selection ratios, coverage, entropy, etc. |
| `EXPERIMENTS.md` | Detailed CLI reference for `experiments.py` (all flags and field values) |

**`training.py` vs `rl_training.py`:** `training.py` has the simplified
reference loops (fixed REINFORCE, loss-improvement reward, top-k, Shannon
entropy). `rl_training.py` is the experiment-grade version where every design
choice is configurable. Both are used by `compare.py`; only `rl_training.py`
is used by `experiments.py`.

**`compare.py` vs `experiments.py`:** `compare.py` runs a single experiment
(baseline + router side by side). `experiments.py` generates many configs
(ablation or grid search) and runs them sequentially.

---

## Setup

**Requirements:** Python 3.10+, CUDA optional but recommended for training.

```bash
pip install -r requirement.txt
```

The first run will stream dataset shards from HuggingFace Hub. Subsequent runs
use the local HuggingFace cache (`~/.cache/huggingface/`).

---

## How to Run

### 1. Verify the install

```bash
python smoke_test.py
```
Expected: `All 6 tests passed.`

### 2. Single comparison run (baseline vs router vs aux-net)

```bash
python compare.py
```
Trains all three models on the dataset configured in `ExperimentConfig`
(default: single-dataset mode on FineWeb with external embeddings).
Results are saved to `results/` and printed to stdout.

### 3. Ablation study

```bash
# See all experiments that would run without actually running them
python experiments.py --list

# Run the full ablation suite (~37 experiments, one-factor-at-a-time)
python experiments.py --all

# Run only the reward signal ablation
python experiments.py --field reward_signal

# Run multiple fields
python experiments.py --field training_algorithm --field reward_signal

# Run a curated subset (see EXPERIMENTAL_PROFILES in experiments.py)
python experiments.py --profile final_presentation

# Full grid search over two fields (combinatorial — use with caution)
python experiments.py --combinations --field training_algorithm --field reward_signal
```

See `EXPERIMENTS.md` for the complete CLI reference and descriptions of every
experimental field and its options.
````

- [ ] **Step 2: Verify the file was written correctly**

```bash
python -c "
with open('README.md') as f:
    content = f.read()
assert '## Architecture' in content
assert '## File Map' in content
assert '## Setup' in content
assert '## How to Run' in content
print(f'README OK ({len(content.splitlines())} lines)')
"
```
Expected: `README OK (N lines)` where N > 80.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: write README with project overview, architecture, file map, setup, and usage"
```

---

## Self-Review

**Spec coverage:**

- ✅ `TESTS/` and `old_results/` moved to `archive/` — Task 1
- ✅ `clean/` contents moved to repo root — Task 2
- ✅ All five file renames — Task 3
- ✅ All import statements updated after renames — Task 3
- ✅ `config.py` commented — Task 4
- ✅ `data.py` commented — Task 5
- ✅ `model.py` commented — Task 6
- ✅ `training.py` commented — Task 7
- ✅ `router.py` commented — Task 8
- ✅ `rl_training.py` commented (priority spots: RL loop, entropy sign, cache epoch-0, gradient timing) — Task 9
- ✅ `experiments.py` commented — Task 10
- ✅ `compare.py`, `smoke_test.py`, `metrics.py`, `visualize.py` commented — Task 11
- ✅ README sections 1–5 — Task 12
- ✅ `extract_hierarchical_features` exists in `model.py` (line 136) — no import fix needed; removed from plan

**Placeholder scan:** No TBD, TODO, or vague steps. Every step contains the actual docstring or comment text to write.

**Type consistency:** No cross-task type dependencies — all tasks add documentation to existing interfaces.
