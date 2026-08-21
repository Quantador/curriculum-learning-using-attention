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
| `models/model.py` | `TinyGPT` (causal LM), router architectures, feature extraction utilities |
| `utils/metrics.py` | `MetricsTracker` (logging + JSON), `DiversityTracker` (coverage stats) |
| `training.py` | Reference training loops: `train_baseline`, `train_router` (simplified) |
| `models/router.py` | Router factory `build_router()`, `extract_router_features()`, `AuxNetRouter` |
| `rl_training.py` | Full RL training loop with all configurable algorithms and schedules |
| `parallel_experiments.py` | Ablation orchestration: generates one-factor-at-a-time/grid-search configs and either runs one in-process or schedules many across one GPU |
| `compare.py` | Entry point for a single baseline vs router vs aux-net comparison |
| `tests/smoke_test.py` | 6 end-to-end smoke tests — run after any structural change |
| `visualize.py` | Plots validation PPL, selection ratios, coverage, entropy, etc. |
| `EXPERIMENTS.md` | Detailed CLI reference for `parallel_experiments.py` (all flags and field values) |

**`training.py` vs `rl_training.py`:** `training.py` has the simplified
reference loops (fixed REINFORCE, loss-improvement reward, top-k, Shannon
entropy). `rl_training.py` is the experiment-grade version where every design
choice is configurable. Both are used by `compare.py`; only `rl_training.py`
is used by `parallel_experiments.py`.

**`compare.py` vs `parallel_experiments.py`:** `compare.py` runs a single
experiment (baseline + router side by side). `parallel_experiments.py`
generates many configs (ablation or grid search) and either runs a bare
single config in-process or schedules the whole sweep across one GPU,
launching each experiment as its own subprocess (see `EXPERIMENTS.md`).

---

## Setup

**Requirements:** Python 3.12 (pinned by `pyproject.toml`'s `requires-python`),
CUDA optional but recommended for training.

Dependencies live in `pyproject.toml`, resolved through `uv.lock`:

```bash
uv sync
```

`uv sync` installs the locked versions exactly. After editing `pyproject.toml`'s
dependency list, run `uv lock` to refresh `uv.lock` -- `uv lock --check` fails if
the two have drifted, which is what catches a dependency that was declared but
never locked (`lm-eval` was in exactly that state until it was locked in).

The first run will stream dataset shards from HuggingFace Hub. Subsequent runs
use the local HuggingFace cache (`~/.cache/huggingface/`).

---

## How to Run

### 1. Verify the install

```bash
python tests/smoke_test.py
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
python parallel_experiments.py --list

# Run the full ablation suite (~37 experiments, one-factor-at-a-time,
# scheduled across one GPU)
python parallel_experiments.py --all

# Run only the reward signal ablation
python parallel_experiments.py --field reward_signal

# Run multiple fields
python parallel_experiments.py --field training_algorithm --field reward_signal

# Run a curated subset (see EXPERIMENT_PROFILES in parallel_experiments.py)
python parallel_experiments.py --profile final_presentation

# Full grid search over two fields (combinatorial — use with caution)
python parallel_experiments.py --combinations --field training_algorithm --field reward_signal
```

See `EXPERIMENTS.md` for the complete CLI reference and descriptions of every
experimental field and its options.
