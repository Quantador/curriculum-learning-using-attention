from typing import Any, Dict, List, Tuple
from pathlib import Path 

# Maps each experimental dimension to (baseline_value, [alternative_values]).
# The baseline_value is used in the control experiment (experiment_name="baseline").
# Each alternative generates one experiment that changes only this single field.
# This one-factor-at-a-time design lets us isolate the effect of each choice.
EXPERIMENTAL_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Router architecture
    "router_architecture": ("attention", ["mlp", "linear"]),

    # Router features
    "enable_text_stat": (True, [False]),
    "enable_text_hierarchical": (True, [False]),
    "hierarchical_representation": ("full", ["embedder"]),

    # Training algorithm
    "training_algorithm": ("ppo", ["grpo", "reinforce"]),
    "reward_signal": ("loss_improvement", [
        "neg_loss",
        "relative_improvement",
        "difficulty_weighted",
        "uncertainty_reduction",
        "gradient_norm",
        "gradient_alignment",
        "combined",
    ]),

    # Selection strategy
    "selection_strategy": ("topk", ["sample", "epsilon_greedy"]),

    # Baseline for variance reduction
    "baseline_type": ("batch_mean", ["moving_avg", "none"]),

    # Temperature schedule
    "temp_schedule": ("fixed", ["linear_decay", "cosine_decay"]),

    # Entropy schedule
    "entropy_schedule": ("fixed", ["linear_decay", "cosine_decay", "exponential_decay", "cyclic"]),

    # Entropy formulation
    "entropy_type": ("shannon", ["renyi", "tsallis", "kl_uniform"]),

    # Entropy targeting (SAC-style automatic adjustment)
    "use_entropy_targeting": (False, [True]),

    # Coverage regularization
    "use_coverage_regularization": (False, [True]),
    "coverage_type": ("count", ["recency", "uncertainty"]),

    # Dataset combinations (easy_dataset, hard_dataset)
    # Easy options: TinyStories, Children-Stories, SimpleWikipedia, WikiText
    # Hard options: OpenWebText2, ArXiv, Code, FineWeb-Edu
    "easy_dataset": ("roneneldan/TinyStories", [
        "ajibawa-2023/Children-Stories-Collection",
        "Salesforce/wikitext",
    ]),
    "hard_dataset": ("Geralt-Targaryen/openwebtext2", [
        "armanc/scientific_papers",
        "CShorten/ML-ArXiv-Papers",
        "HuggingFaceFW/fineweb-edu",
    ]),

    # Feature caching
    "feature_cache_epochs": (0, [1, 2]),
}
# New experiments added for NeurIPS:
#   1. Multi-head attention router (n_heads = 2 and 4)
#   2. Harder easy datasets (WikiText, Children-Stories)
#   3. Harder hard datasets (scientific papers, ML-ArXiv, FineWeb-Edu)
# Note: single-dataset mode and aux-net baseline use a different training loop
# and must be run via compare.py with use_single_dataset / run_aux_baseline.
ADDITIONAL_EXPERIMENTS_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    "router_n_heads": (1, [2, 4]),
    "easy_dataset": (
        "roneneldan/TinyStories",
        ["ajibawa-2023/Children-Stories-Collection", "Salesforce/wikitext"],
    ),
    "hard_dataset": (
        "Geralt-Targaryen/openwebtext2",
        ["armanc/scientific_papers", "CShorten/ML-ArXiv-Papers", "HuggingFaceFW/fineweb-edu"],
    ),
}

# Predefined experiment profiles (subset of ablations)
FINAL_PRESENTATION_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Neg loss, gradient magnitude
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["neg_loss", "gradient_norm", "greats_score"]),
    # GRPO, PPO
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo", "reinforce"]),
    # Shannon fixed vs Shannon with decay (linear)
    "entropy_schedule": (EXPERIMENTAL_FIELDS["entropy_schedule"][0], ["linear_decay"]),
    # Coverage bonus
    "use_coverage_regularization": (EXPERIMENTAL_FIELDS["use_coverage_regularization"][0], [True]),
    # Top-k (baseline), sampling, sigma-greedy (epsilon_greedy)
    "selection_strategy": (EXPERIMENTAL_FIELDS["selection_strategy"][0], ["sample", "epsilon_greedy"]),
}

COMPARE_GRPO_VS_GREATS: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["greats_score"]),
    # GRPO, PPO
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo"]),
}
COMPARE_REWARD_SIGNALS: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["neg_loss", "gradient_norm", "greats_score"])
}
FEATURE_CACHE_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # baseline=0 (no cache) is the reference; 2 is the experiment
    "feature_cache_epochs": (0, [2]),
}

LOSS_IMPROVEMENT_LOGGING: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["loss_improvement"])
}

COMPARE_USE_ORIGINAL_SEQUENCE: Dict[str, tuple[Any, List[Any]]] = {
    "use_original_sequence": (False, [True])
}
EXPERIMENT_PROFILES: Dict[str, Dict[str, tuple[Any, List[Any]]]] = {
    "final_presentation": FINAL_PRESENTATION_FIELDS,
    "final-presentation": FINAL_PRESENTATION_FIELDS,  # alias
    "feature_cache": FEATURE_CACHE_FIELDS,
    "additional_experiments": ADDITIONAL_EXPERIMENTS_FIELDS,
    "additional-experiments": ADDITIONAL_EXPERIMENTS_FIELDS,  # alias
    "grpo_vs_greats": COMPARE_GRPO_VS_GREATS,
    "compare_reward_signals": COMPARE_REWARD_SIGNALS,
    "loss_improvement_logging": LOSS_IMPROVEMENT_LOGGING,
    "compare_use_original_sequence": COMPARE_USE_ORIGINAL_SEQUENCE,
    "compare-use-original-sequence": COMPARE_USE_ORIGINAL_SEQUENCE,  # alias
}

SCRATCH_DIR = Path("results/_parallel_run")
CONTEXT_OVERHEAD_BYTES = 400 * 1024 * 1024  # per-process CUDA context overhead
PER_PROC_BUFFER = 1.15  # safety factor over the measured probe peak

# Persists across sweeps (unlike SCRATCH_DIR, which is only reused, not
# versioned): one subdirectory per distinct dataset signature (see
# utils/shared_dataset.dataset_signature), so re-running the same dataset
# config skips re-tokenizing even across separate script invocations.
DATASET_CACHE_DIR = Path("results/dataset_cache")
