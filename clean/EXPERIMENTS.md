# Experiments Guide

This document describes how to use `experiments.py` to run curriculum learning experiments with various configurations.

## Quick Start

```bash
# Run default single experiment
python experiments.py

# List all available ablation experiments
python experiments.py --list

# Run all ablation experiments
python experiments.py --all
```

## Command Line Options

| Flag | Description |
|------|-------------|
| `--all` | Run all ablation experiments (one-factor-at-a-time) |
| `--combinations` | Run full grid search of all combinations (warning: many experiments!) |
| `--field FIELD` | Run experiments for specific field(s) only (can be repeated) |
| `--no-baseline` | Skip the baseline experiment |
| `--list` | List experiments without running them |

## Experiment Modes

### 1. Single Experiment (Default)

Runs one experiment with the default `ExperimentConfig` settings.

```bash
python experiments.py
```

### 2. Ablation Study (Recommended)

Runs one-factor-at-a-time experiments where each experiment varies only ONE field from the baseline. This is the recommended approach for systematic evaluation.

```bash
python experiments.py --all
```

**Current ablation experiments (~26 total):**
- 1 baseline
- 2 router architectures (attention, linear)
- 2 router features (disable text_stat, disable hierarchical)
- 2 training algorithms (grpo, ppo)
- 5 reward signals (neg_loss, relative_improvement, difficulty_weighted, uncertainty_reduction, combined)
- 2 selection strategies (sample, epsilon_greedy)
- 2 baseline types (moving_avg, none)
- 2 temperature schedules (linear_decay, cosine_decay)
- 1 entropy schedule (linear_decay)
- 2 easy datasets (Children-Stories, WikiText)
- 3 hard datasets (scientific_papers, ML-ArXiv, fineweb-edu)

### 3. Field-Specific Ablation

Run ablations for specific experimental fields only.

```bash
# Only training algorithm variants
python experiments.py --field training_algorithm

# Multiple fields
python experiments.py --field training_algorithm --field reward_signal

# Skip baseline when running specific fields
python experiments.py --field reward_signal --no-baseline
```

### 4. Full Grid Search

Run all combinations of experimental values (use with caution - exponential growth!).

```bash
# Full grid search (thousands of experiments!)
python experiments.py --combinations

# Grid search for specific fields only
python experiments.py --combinations --field training_algorithm --field reward_signal
```

## Experimental Fields

### Router Architecture
| Value | Description |
|-------|-------------|
| `mlp` (baseline) | Multi-layer perceptron router |
| `attention` | Attention-based router |
| `linear` | Simple linear router |

### Router Features
| Field | Baseline | Alternative |
|-------|----------|-------------|
| `enable_text_stat` | True | False |
| `enable_text_hierarchical` | True | False |

### Training Algorithm
| Value | Description |
|-------|-------------|
| `reinforce` (baseline) | Standard REINFORCE with baseline |
| `grpo` | Group Relative Policy Optimization |
| `ppo` | Proximal Policy Optimization |

### Reward Signal
| Value | Formula | Description |
|-------|---------|-------------|
| `loss_improvement` (baseline) | `(loss_before - loss_after).clamp(0)` | Reward learning progress |
| `neg_loss` | `-loss_after` | Prefer easier samples |
| `relative_improvement` | `improvement / loss_before` | Normalized improvement |
| `difficulty_weighted` | `improvement * (1 + difficulty)` | Reward harder samples more |
| `uncertainty_reduction` | `entropy_before - entropy_after` | Reward confidence gain |
| `combined` | Weighted sum | Multi-objective reward |

### Selection Strategy
| Value | Description |
|-------|-------------|
| `topk` (baseline) | Select top-k highest scoring samples |
| `sample` | Sample from probability distribution |
| `epsilon_greedy` | Random with probability epsilon, else topk |

### Baseline Type (Variance Reduction)
| Value | Description |
|-------|-------------|
| `batch_mean` (baseline) | Per-batch mean reward |
| `moving_avg` | Exponential moving average |
| `none` | No baseline subtraction |

### Temperature Schedule
| Value | Description |
|-------|-------------|
| `fixed` (baseline) | Constant temperature |
| `linear_decay` | Linear decay from `temp` to `temp_min` |
| `cosine_decay` | Cosine annealing from `temp` to `temp_min` |

### Entropy Schedule
| Value | Description |
|-------|-------------|
| `fixed` (baseline) | Constant entropy coefficient |
| `linear_decay` | Linear decay from `lambda_ent` to `lambda_ent_min` |
| `cosine_decay` | Cosine annealing |
| `exponential_decay` | Faster initial decay |
| `cyclic` | Warm restarts for periodic exploration |

### Entropy Formulation
| Value | Description | Parameters |
|-------|-------------|------------|
| `shannon` (baseline) | Standard entropy: `-sum(p * log(p))` | - |
| `renyi` | Rényi entropy: `(1/(1-α)) * log(sum(p^α))` | `entropy_alpha` |
| `tsallis` | Tsallis entropy: `(1/(q-1)) * (1 - sum(p^q))` | `entropy_q` |
| `kl_uniform` | KL divergence from uniform distribution | - |

### Entropy Targeting (SAC-style)
When `use_entropy_targeting=True`, the entropy coefficient is automatically adjusted to maintain a target entropy level:
- If current entropy < target → increase `lambda_ent` (encourage exploration)
- If current entropy > target → decrease `lambda_ent` (allow exploitation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_entropy_ratio` | 0.5 | Target = ratio × max_entropy |
| `entropy_lr` | 1e-3 | Learning rate for coefficient adjustment |

### Coverage Regularization
Encourages the router to select diverse samples over time by penalizing repeated selection.

| Type | Description |
|------|-------------|
| `count` | Inverse of selection count (less selected = higher bonus) |
| `recency` | Inverse of recency (less recently selected = higher bonus) |
| `uncertainty` | Higher average loss = more uncertain = higher bonus |

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_coverage` | 0.01 | Weight for coverage regularization |
| `coverage_decay` | 0.99 | Decay factor for recency tracking |
| `coverage_temperature` | 1.0 | Temperature for coverage bonus |

### Datasets

#### Easy Datasets
| Value | Description |
|-------|-------------|
| `roneneldan/TinyStories` (baseline) | Synthetic children's stories |
| `ajibawa-2023/Children-Stories-Collection` | Real children's literature |
| `Salesforce/wikitext` | Clean Wikipedia articles |

#### Hard Datasets
| Value | Description |
|-------|-------------|
| `Geralt-Targaryen/openwebtext2` (baseline) | General web text |
| `armanc/scientific_papers` | ArXiv + PubMed papers |
| `CShorten/ML-ArXiv-Papers` | ML research papers |
| `HuggingFaceFW/fineweb-edu` | Educational web content |

## Examples

### Preview Experiments
```bash
# See what would run without actually running
python experiments.py --list
python experiments.py --list --field training_algorithm
python experiments.py --list --combinations --field training_algorithm --field baseline_type
```

### Common Experiment Sets
```bash
# Compare training algorithms
python experiments.py --field training_algorithm

# Compare reward signals
python experiments.py --field reward_signal

# Compare datasets
python experiments.py --field easy_dataset --field hard_dataset

# Full ablation study
python experiments.py --all

# Training algorithm x reward signal grid
python experiments.py --combinations --field training_algorithm --field reward_signal
```

## Output

Results are saved to:
- `results/<experiment_name>/` - Directory for each experiment
- `results/<experiment_name>.json` - Metrics file

Metrics include:
- `loss_lm` - Language model loss
- `loss_router` - Router loss
- `policy_loss` - Policy gradient loss
- `entropy` - Router entropy
- `avg_reward` - Average reward
- `val_loss` - Validation loss
- `val_ppl` - Validation perplexity

## Configuration

All experimental parameters can be modified in `config.py` under `ExperimentConfig`. Key parameters:

```python
# Training
epochs: int = 5
batch: int = 16
lr_lm: float = 3e-4
lr_router: float = 1e-3

# Temperature
temp: float = 1.0
temp_min: float = 0.1

# Entropy
lambda_ent: float = 0.005
lambda_ent_min: float = 0.001

# PPO specific
ppo_clip: float = 0.2
ppo_epochs: int = 4

# GRPO specific
grpo_group_size: int = 4

# Combined reward weights
reward_weight_improvement: float = 1.0
reward_weight_difficulty: float = 0.5
reward_weight_uncertainty: float = 0.3
```

## Adding New Experiments

To add a new experimental field:

1. Add the field to `ExperimentConfig` in `config.py`
2. Add the field to `EXPERIMENTAL_FIELDS` in `experiments.py`:
   ```python
   "new_field": ("baseline_value", ["alt1", "alt2"]),
   ```
3. Implement the logic in `trainingExperiments.py`
