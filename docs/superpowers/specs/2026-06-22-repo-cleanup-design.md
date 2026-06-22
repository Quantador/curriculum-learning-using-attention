# Repo Cleanup Design — 2026-06-22

## Goal

Prepare the repository for a new collaborator (ML researcher intern, solid Python/PyTorch background).
The cleanup covers: archiving dead code, restructuring the file layout, renaming files for clarity,
and adding comprehensive comments and a clear README.

## Approach

**Approach B** was selected: structural changes + README (sections 1–5) + full commenting pass on all files.
No logic changes — documentation only beyond the file moves and renames.

---

## Section 1: Repository Structure

### Moves

| From | To |
| --- | --- |
| `TESTS/` | `archive/TESTS/` |
| `old_results/` | `archive/old_results/` |
| `clean/*.py` | root |
| `clean/EXPERIMENTS.md` | root |
| `clean/run_experiment.ipynb` | root |

### File Renames

| Old name | New name | Reason |
| --- | --- | --- |
| `clean/modelExperiments.py` | `router.py` | Defines router architectures and feature extraction — nothing experiment-specific |
| `clean/trainingExperiments.py` | `rl_training.py` | The RL-based training loop (REINFORCE/GRPO/PPO) — distinct from `training.py` |
| `clean/baselineVSrouter.py` | `compare.py` | Entry point for a single baseline vs router comparison run |
| `clean/test_run.py` | `smoke_test.py` | Standard name for fast sanity-check tests |
| `visu.py` | `visualize.py` | Clearer name |

### Files That Keep Their Names

`config.py`, `data.py`, `model.py`, `metrics.py`, `training.py`, `experiments.py`, `EXPERIMENTS.md`

### Final Root Structure

```text
curriculum-learning-using-attention/
├── README.md
├── requirement.txt
├── .gitignore
├── config.py
├── data.py
├── model.py
├── metrics.py
├── training.py
├── router.py
├── rl_training.py
├── experiments.py
├── compare.py
├── smoke_test.py
├── visualize.py
├── EXPERIMENTS.md
├── run_experiment.ipynb
└── archive/
    ├── TESTS/
    └── old_results/
```

### Import Updates Required

All files that import from the renamed modules need their import statements updated:

- `from modelExperiments import ...` → `from router import ...`
- `from trainingExperiments import ...` → `from rl_training import ...`

Files affected: `compare.py`, `experiments.py`, `smoke_test.py`, `rl_training.py` itself.

---

## Section 2: README Structure

The README covers sections 1–5. Sections on design decisions and extension guides are intentionally
omitted — not enough information yet to write them accurately.

### 1. What this is (3–4 sentences)

Research project: training a small language model (TinyGPT) with curriculum learning. Instead of
random batches, a learned router selects which training samples to present at each step. The router
is trained with reinforcement learning using the LM's learning signal as reward. States the research
question clearly.

### 2. Architecture diagram

ASCII diagram of the core loop:

```text
Pool of candidates → Router scores → Selection strategy → LM forward+backward
                                                              ↓
                                          Router RL update ← Reward signal
```

### 3. File map

One line per file. Explicitly distinguishes:

- `training.py` vs `rl_training.py` (simple baseline loops vs full RL experiment loops)
- `compare.py` vs `experiments.py` (single run vs ablation study)

### 4. Setup

`pip install -r requirement.txt`, Python version, CUDA notes.

### 5. How to run

Three entry points with copy-paste examples:

- `smoke_test.py` — verify install
- `compare.py` — single baseline vs router run
- `experiments.py --list`, `--all`, `--profile`, `--field` — ablation studies

---

## Section 3: Commenting Strategy

Full commenting pass across all files. No logic changes — documentation only.

### Every file gets

- Module-level docstring: what the file does, what's in it, how it relates to other files
- Function and class docstrings for everything that doesn't already have one
- Inline comments on non-obvious logic

### Priority spots (non-obvious logic that needs explanation)

**`rl_training.py`**

- High-level block comment before the main pool→select→LM→reward→router loop
- Why entropy is negated throughout (`-H` convention so minimizing loss = maximizing entropy)
- Why the feature cache is never built at epoch 0 (weights too noisy for useful features)
- Why `compute_gradient_reward()` reads gradients before `opt_lm.step()`
- How the three RL algorithms (REINFORCE/GRPO/PPO) diverge from the same base

**`model.py`**

- Router architecture variants and what differentiates them
- The `extract_hierarchical_hidden()` chunking strategy

**`router.py`**

- `get_router_feature_dim()`: why the fallback is `n_chunks * d_model + 4` (the 4 text statistics)
- The three feature groups and when each is active

**`experiments.py`**

- The one-factor-at-a-time ablation logic in `generate_experiment_configs()`
- How `EXPERIMENTAL_FIELDS` drives config generation

**Remaining files** (`config.py`, `data.py`, `metrics.py`, `training.py`, `compare.py`,
`smoke_test.py`, `visualize.py`): module docstrings + function docstrings where missing.

---

## Pre-existing Issue to Fix During Pass

`training.py` imports `extract_hierarchical_features` from `model.py`, but the function is
actually named `extract_hierarchical_hidden` in `model.py`. This is a one-line import fix
(not a logic change) that should be corrected during the commenting pass so the file is
importable.

---

## What Is NOT Changing

- No logic changes anywhere (except the import name fix above)
- No refactoring of code structure within files
- `EXPERIMENTS.md` stays as-is (already thorough)
- No sections 6/7 in README (design decisions, extension guide) — insufficient info
