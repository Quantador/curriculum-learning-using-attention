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
