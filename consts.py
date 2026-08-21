from typing import Any, Dict, List, Tuple
from pathlib import Path 


# Registry mapping HuggingFace dataset names to their split/text-column metadata.
# Easier datasets produce simpler, shorter text; harder ones contain dense or
# domain-specific language.
#
# Easy:   roneneldan/TinyStories, ajibawa-2023/Children-Stories-Collection,
#         Salesforce/wikitext
# Medium: Geralt-Targaryen/openwebtext2, HuggingFaceFW/fineweb-edu, allenai/c4
# Hard:   armanc/scientific_papers, CShorten/ML-ArXiv-Papers
# Unstructured (single-dataset): HuggingFaceFW/fineweb
DATASET_REGISTRY: dict[str, dict] = {
    "roneneldan/TinyStories":                   {"split": "train", "text_col": "text"},
    "ajibawa-2023/Children-Stories-Collection": {"split": "train", "text_col": "text"},
    "Salesforce/wikitext":                      {"split": "train", "name": "wikitext-103-raw-v1", "text_col": "text"},
    "Geralt-Targaryen/openwebtext2":            {"split": "train", "text_col": "text"},
    "armanc/scientific_papers":                 {"split": "train", "text_col": "abstract"},
    "CShorten/ML-ArXiv-Papers":                 {"split": "train", "text_col": "abstract"},
    "HuggingFaceFW/fineweb-edu":                {"split": "train", "text_col": "text"},
    "HuggingFaceFW/fineweb":                    {"split": "train", "name": "sample-10BT", "text_col": "text"},
    # Bigger sample of the same underlying repo, for token budgets past 10B --
    # a distinct registry key (not a mutation of the entry above) so existing
    # configs pinned to the 10BT scope are unaffected. "repo_id" decouples the
    # registry key from the literal load_dataset() argument -- see
    # tokenization.py's registry_entry(path).get("repo_id", path) call sites.
    "HuggingFaceFW/fineweb-100BT":               {"repo_id": "HuggingFaceFW/fineweb", "split": "train", "name": "sample-100BT", "text_col": "text"},
    "allenai/c4":                               {"split": "train", "name": "en", "text_col": "text"},
    "DKYoon/SlimPajama-6B":                     {"split": "train", "text_col": "text"}
}

# Maps each experimental dimension to (baseline_value, [alternative_values]).
# The baseline_value is used in the control experiment (experiment_name="experiment_baseline").
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

    # Feature caching
    "feature_cache_epochs": (0, [1, 2]),
    "use_original_sequence": (False, [True]),
}
# New experiments added for NeurIPS:
#   1. Multi-head attention router (n_heads = 2 and 4)
#   2. Harder easy datasets (WikiText, Children-Stories)
#   3. Harder hard datasets (scientific papers, ML-ArXiv, FineWeb-Edu)
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
    "use_original_sequence": (EXPERIMENTAL_FIELDS["use_original_sequence"][0], [True])
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

CHECK_GREATS: Dict[str, tuple[Any, List[Any]]] = {
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["greats_score"])
}

# Baseline: router updates (REINFORCE/GRPO/PPO) for the whole run (current
# default, router_freeze_progress=None). Alternative: stop updating the
# router after 30% of training progress, but keep using its now-frozen
# weights to select samples for the remaining 70% -- tests whether continued
# router training helps past that point, or an early-converged router is
# already "good enough". See config.py's router_freeze_progress docstring.
COMPARE_ROUTER_FREEZE: Dict[str, tuple[Any, List[Any]]] = {
    "router_freeze_progress": (None, [0.3])
}

# Baseline: router reward/update every LM training step (current default,
# router_update_every=1). Alternatives: only every 2nd/5th step -- the router
# still scores/selects the pool every step with its current weights, this
# just throttles how often it pays for the extra loss_after forward pass and
# actually learns from a reward. See config.py's router_update_every
# docstring.
COMPARE_ROUTER_UPDATE_EVERY: Dict[str, tuple[Any, List[Any]]] = {
    "router_update_every": (1, [2, 5])
}

COMPARE_TRANSFORMER_LAYER: Dict[str, tuple[Any, List[Any]]] = {
    "hierarchical_layer_index": (0, [2, 4, 6, 8, 10, 12])
}

SCHEDULE_ABLATION: Dict[str, tuple[Any, List[Any]]] = {
    "curriculum_ratio_schedule": ("linear_decay", ["cosine_decay", "exponential_decay", "cyclic"])
}

# "" = no sentence-embedder feature (router falls back to the base
# hierarchical/stat features only). Every alternative below is a 768-dim
# model, matching ExperimentConfig.sentence_embedder_dim's default -- so this
# one-factor-at-a-time ablation never needs a second field to track the
# embedding dimension per model (see utils/sentence_embedder.py).
SENTENCE_EMBEDDER_ABLATION: Dict[str, tuple[Any, List[Any]]] = {
    "sentence_embedder_model": ("", [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "thenlper/gte-base",
        "intfloat/e5-base-v2",
    ]),
}

# One-factor-at-a-time comparison of every distinct way extract_router_features()
# (models/router.py) can build the router's input vector:
#   - enable_text_hierarchical=False  -> drop the hierarchical hidden-state group,
#                                         router sees only text stats
#   - enable_text_stat=False          -> drop the cheap text-statistics group,
#                                         router sees only the hierarchical group
#   - hierarchical_representation:
#       full      (baseline) -> final transformer hidden state
#       embedder              -> token+positional embeddings only, no transformer layers
#       layer                 -> hidden state after hierarchical_layer_index layers
#                                 (base config below pins hierarchical_layer_index=6
#                                 so this alternative is valid without touching that field)
#   - use_original_sequence=True      -> bypass every feature group; router scores
#                                         the raw token ids directly
#   - router_feature_source:
#       features      (baseline) -> the concatenated groups above
#       own_embeddings           -> the router owns token+positional embedding
#                                    tables and learns them from the routing
#                                    objective, never touching the LM. Pools
#                                    identically to hierarchical_representation=
#                                    'embedder', so the pair isolates learned-
#                                    by-the-router vs. frozen-from-the-LM
#   - sentence_embedder_model set     -> concatenate a frozen precomputed sentence
#                                         embedding onto the hierarchical+text-stat
#                                         baseline features (additive; only one
#                                         representative model here -- see
#                                         SENTENCE_EMBEDDER_ABLATION above to compare
#                                         models against each other)
#   - sentence_embedder_alone (combo) -> same sentence embedding, but with
#                                         enable_text_hierarchical/enable_text_stat
#                                         both off -- isolates the embedder's own
#                                         signal instead of adding it on top
# use_external_embeddings (config.py) is intentionally excluded: it's rejected by
# tokenization.py's datatrove path (NotImplementedError) and can't currently run.
ROUTER_FEATURE_ABLATION: Dict[str, tuple[Any, List[Any]]] = {
    "hierarchical_representation": ("full", ["embedder", "layer"]),
    "use_original_sequence": (False, [True]),
    "router_feature_source": ("features", [
        {
            "_name": "own_embeddings",
            "router_feature_source": "own_embeddings",
            # The router's own tables replace every other group, so the
            # baseline's feature flags have to come off or config validation
            # would be describing a router input that is never built.
            "enable_text_hierarchical": False,
            "enable_text_stat": False,
        },
    ]),
    "sentence_embedder_model": ("", [
        "intfloat/e5-base-v2",
        {
            "_name": "sentence_embedder_alone",
            "sentence_embedder_model": "intfloat/e5-base-v2",
            "enable_text_hierarchical": False,
            "enable_text_stat": False,
        },
    ]),
}

# How the router learns (training_algorithm) and what it learns from
# (reward_signal), with the router's INPUT held fixed at own_embeddings by
# configs/own_embeddings_rl_ablation.yaml -- that input won the
# router_feature_ablation sweep, so this sweep stops varying it and asks the
# next question instead: given the best features, which objective actually
# beats uniform random selection?
#
# random_batch_baseline is the reference every arm here is judged against and
# rides along automatically (see generate_experiment_configs) -- it trains the
# same LM on the same per-step batch size with no router at all, so the only
# difference is which samples were picked.
#
# reward_signal alternatives are the ones with a mechanism for beating random,
# rather than the full list from EXPERIMENTAL_FIELDS:
#   gradient_norm      -- prefer batches with large ||grad||, i.e. where the LM
#                         still has something to learn
#   gradient_alignment -- <g_t, g_ema>: prefer batches pulling in the same
#                         direction as recent progress
#   greats_score       -- GREATS ghost gradient-dot-product against a val batch
#                         (influence-function style; the only arm here whose
#                         reward references held-out data)
#   neg_loss           -- prefer easy samples; included as the directional
#                         opposite of gradient_norm, so a win either way is
#                         informative about what the router should chase
OWN_EMBEDDINGS_RL_ABLATION: Dict[str, tuple[Any, List[Any]]] = {
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo", "reinforce"]),
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], [
        "gradient_norm",
        "gradient_alignment",
        "neg_loss",
        {
            # greats_score hands every sample in the batch the SAME scalar, so
            # the default baseline_type='batch_mean' subtracts the reward from
            # itself and every advantage is identically zero -- the router
            # would train on no signal at all and silently look like a null
            # result. moving_avg is the baseline that survives a constant
            # reward (see reward_signal's docstring in config.py).
            "_name": "greats_score",
            "reward_signal": "greats_score",
            "baseline_type": "moving_avg",
        },
    ]),
}

# Narrow 3-arm cut of OWN_EMBEDDINGS_RL_ABLATION: does greats_score's ghost
# gradient-dot-product reward (plus its second-order diversity term) beat
# plain loss_improvement, both under PPO with own_embeddings features? Run
# with --profile own_embeddings_loss_vs_greats_diversity --no-baseline against
# configs/own_embeddings_rl_ablation.yaml (unchanged) to get exactly:
#   - experiment_baseline:     ppo, loss_improvement, own_embeddings
#   - random_batch_baseline:   same, uniform random selection, no router
#   - greats_score_diversity:  ppo, greats_score+diversity, own_embeddings
# training_algorithm is pinned to "ppo" with an EMPTY alternatives list
# (unlike OWN_EMBEDDINGS_RL_ABLATION above, which sweeps it) -- baseline_values
# still picks up "ppo" for every reference/arm config, but the empty list
# means the one-factor-at-a-time loop contributes zero grpo/reinforce arms.
OWN_EMBEDDINGS_LOSS_VS_GREATS_DIVERSITY: Dict[str, tuple[Any, List[Any]]] = {
    "training_algorithm": ("ppo", []),
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], [  # "loss_improvement"
        {
            # Same moving_avg rationale as OWN_EMBEDDINGS_RL_ABLATION's
            # greats_score combo above, plus the redundancy penalty
            # (config.py greats_diversity_term docstring), which needs
            # greats_log_grad_norms=True for the per-sample gradient norms it
            # sums.
            "_name": "greats_score_diversity",
            "reward_signal": "greats_score",
            "baseline_type": "moving_avg",
            "greats_diversity_term": True,
            "greats_log_grad_norms": True,
        },
    ]),
}

# The minimal 2-arm question: with the router's input fixed at own_embeddings
# (by the config, not this profile), does a PPO router trained on
# loss_improvement beat picking the same number of samples at random? Run with
# --profile own_embeddings_ppo_vs_random --no-baseline against
# configs/own_embeddings_ppo_vs_random.yaml to get exactly:
#   - experiment_baseline:   ppo, loss_improvement, own_embeddings (the router)
#   - random_batch_baseline: same LM/data/batch size, uniform random selection,
#                            no router at all (training.train_baseline())
# BOTH fields are pinned with EMPTY alternatives lists, so the
# one-factor-at-a-time loop in generate_experiment_configs() contributes zero
# arms of its own and only the two unconditional reference runs are generated
# (see _reference_configs). Pinning them here rather than relying on
# ExperimentConfig's dataclass defaults matters: training_algorithm defaults to
# "reinforce", not "ppo".
OWN_EMBEDDINGS_PPO_VS_RANDOM: Dict[str, tuple[Any, List[Any]]] = {
    "training_algorithm": ("ppo", []),
    "reward_signal": ("loss_improvement", []),
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
    "transformer-layer-ablation": COMPARE_TRANSFORMER_LAYER,
    "transformer_layer_ablation": COMPARE_TRANSFORMER_LAYER,
    "schedule_ablation": SCHEDULE_ABLATION,
    "sentence_embedder_ablation": SENTENCE_EMBEDDER_ABLATION,
    "sentence-embedder-ablation": SENTENCE_EMBEDDER_ABLATION,  # alias
    "router_feature_ablation": ROUTER_FEATURE_ABLATION,
    "compare_router_freeze": COMPARE_ROUTER_FREEZE,
    "compare-router-freeze": COMPARE_ROUTER_FREEZE,  # alias
    "compare_router_update_every": COMPARE_ROUTER_UPDATE_EVERY,
    "compare-router-update-every": COMPARE_ROUTER_UPDATE_EVERY,  # alias
    "check_greats": CHECK_GREATS,
    "own_embeddings_rl_ablation": OWN_EMBEDDINGS_RL_ABLATION,
    "own-embeddings-rl-ablation": OWN_EMBEDDINGS_RL_ABLATION,  # alias
    "own_embeddings_loss_vs_greats_diversity": OWN_EMBEDDINGS_LOSS_VS_GREATS_DIVERSITY,
    "own-embeddings-loss-vs-greats-diversity": OWN_EMBEDDINGS_LOSS_VS_GREATS_DIVERSITY,  # alias
    "own_embeddings_ppo_vs_random": OWN_EMBEDDINGS_PPO_VS_RANDOM,
    "own-embeddings-ppo-vs-random": OWN_EMBEDDINGS_PPO_VS_RANDOM,  # alias
}

SCRATCH_DIR = Path("results/_parallel_run")
CONTEXT_OVERHEAD_BYTES = 400 * 1024 * 1024  # per-process CUDA context overhead
PER_PROC_BUFFER = 1.15  # safety factor over the measured probe peak

# Persists across sweeps (unlike SCRATCH_DIR, which is only reused, not
# versioned): one subdirectory per distinct dataset signature (see
# utils/shared_dataset.dataset_signature), each holding a datatrove-tokenized
# corpus laid out as <split>/<domain>/*.ds (see tokenization.py). Written only
# by build_dataset_cache.py; sweeps and workers read it and never build it.
DATASET_CACHE_DIR = Path("results/dataset_cache")
