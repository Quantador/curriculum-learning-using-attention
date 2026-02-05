# run_experiment.py
from __future__ import annotations

import os
import torch
from dataclasses import replace
from itertools import product
from typing import Dict, List, Any

from config import ExperimentConfig
from data import get_tokenizer, make_mixed_chunks, MixedLMDataset
from model import TinyGPT, AttentionRouter
from metrics import MetricsTracker, DiversityTracker

from modelExperiments import *
from trainingExperiments import *


# Define the experimental fields and their possible values
# Format: field_name -> (baseline_value, [alternative_values])
EXPERIMENTAL_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Router architecture
    "router_architecture": ("attention", ["mlp", "linear"]),

    # Router features
    "enable_text_stat": (True, [False]),
    "enable_text_hierarchical": (True, [False]),
    "hierarchical_representation": ("full", ["embedder"]),

    # Training algorithm
    "training_algorithm": ("reinforce", ["grpo", "ppo"]),
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
}

# Predefined experiment profiles (subset of ablations)
FINAL_PRESENTATION_FIELDS: Dict[str, tuple[Any, List[Any]]] = {
    # Neg loss, gradient magnitude
    "reward_signal": (EXPERIMENTAL_FIELDS["reward_signal"][0], ["neg_loss", "gradient_norm"]),
    # GRPO, PPO
    "training_algorithm": (EXPERIMENTAL_FIELDS["training_algorithm"][0], ["grpo", "ppo"]),
    # Shannon fixed vs Shannon with decay (linear)
    "entropy_schedule": (EXPERIMENTAL_FIELDS["entropy_schedule"][0], ["linear_decay"]),
    # Coverage bonus
    "use_coverage_regularization": (EXPERIMENTAL_FIELDS["use_coverage_regularization"][0], [True]),
    # Top-k (baseline), sampling, sigma-greedy (epsilon_greedy)
    "selection_strategy": (EXPERIMENTAL_FIELDS["selection_strategy"][0], ["sample", "epsilon_greedy"]),
}

EXPERIMENT_PROFILES: Dict[str, Dict[str, tuple[Any, List[Any]]]] = {
    "final_presentation": FINAL_PRESENTATION_FIELDS,
    "final-presentation": FINAL_PRESENTATION_FIELDS,  # alias
}


def get_profile_fields(profile: str | None) -> Dict[str, tuple[Any, List[Any]]] | None:
    if not profile:
        return None
    if profile in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[profile]
    normalized = profile.replace("-", "_").replace(" ", "_")
    if normalized in EXPERIMENT_PROFILES:
        return EXPERIMENT_PROFILES[normalized]
    raise ValueError(
        f"Unknown profile '{profile}'. Available: {', '.join(sorted(set(EXPERIMENT_PROFILES.keys())))}"
    )


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment():
    cfg = ExperimentConfig()
    set_seed(cfg.seed)

    tokenizer = get_tokenizer()

    print("\n=== Building datasets ===")
    train_chunks = make_mixed_chunks("train", cfg, tokenizer)
    val_chunks = make_mixed_chunks("validation", cfg, tokenizer)

    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)

    # Read baseline and router metrics from previous runs
    base_metrics = MetricsTracker.load(f"results/baseline_metrics.json")
    router_metrics = MetricsTracker.load(f"results/router_metrics.json")
    
    print(f"\n=== Training experiment: {cfg.experiment_name} ===")
    
    set_seed(cfg.seed)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = build_router(
        d_input=get_router_feature_dim(cfg),
        arch=cfg.router_architecture,
        d_k=128,
    )

    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router_experiments(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=experiment_metrics,
        diversity=router_div,
    )

    experiment_metrics.save(f"{cfg.save_dir}/{cfg.experiment_name}.json")


    print("\n=== Comparing runs ===")
    compare_runs_experiments(
        base_metrics,
        router_metrics,
        experiment_metrics,
    )

    return experiment_metrics


def run_single_experiment(cfg: ExperimentConfig, tokenizer, train_ds, val_ds, base_metrics, router_metrics):
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*60}")
    print(f"=== Running experiment: {cfg.experiment_name} ===")
    print(f"{'='*60}")
    print(f"  router_architecture: {cfg.router_architecture}")
    print(f"  enable_text_stat: {cfg.enable_text_stat}")
    print(f"  enable_text_hierarchical: {cfg.enable_text_hierarchical}")
    print(f"  hierarchical_representation: {cfg.hierarchical_representation}")
    print(f"  training_algorithm: {cfg.training_algorithm}")
    print(f"  reward_signal: {cfg.reward_signal}")
    print(f"  selection_strategy: {cfg.selection_strategy}")
    print(f"  baseline_type: {cfg.baseline_type}")
    print(f"  temp_schedule: {cfg.temp_schedule}")
    print(f"  entropy_schedule: {cfg.entropy_schedule}")

    set_seed(cfg.seed)

    # Ensure save directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)

    model_router = TinyGPT(vocab_size=tokenizer.vocab_size, cfg=cfg)
    router = build_router(
        d_input=get_router_feature_dim(cfg),
        arch=cfg.router_architecture,
        d_k=128,
    )

    experiment_metrics = MetricsTracker(cfg.experiment_name, use_wandb=cfg.use_wandb)
    router_div = DiversityTracker(len(train_ds))

    model_router, router = train_router_experiments(
        cfg=cfg,
        model=model_router,
        router=router,
        train_ds=train_ds,
        val_ds=val_ds,
        tokenizer=tokenizer,
        metrics=experiment_metrics,
        diversity=router_div,
    )

    experiment_metrics.save(f"{cfg.save_dir}/{cfg.experiment_name}.json")

    print("\n=== Comparing runs ===")
    compare_runs_experiments(
        base_metrics,
        router_metrics,
        experiment_metrics,
    )

    return experiment_metrics


def get_baseline_config() -> Dict[str, Any]:
    """Get the baseline values for all experimental fields."""
    return {field: values[0] for field, values in EXPERIMENTAL_FIELDS.items()}


def generate_experiment_configs(
    base_cfg: ExperimentConfig | None = None,
    experimental_fields: Dict[str, tuple[Any, List[Any]]] | None = None,
    include_baseline: bool = True,
) -> List[ExperimentConfig]:
    """
    Generate ablation-style experimental configurations.

    Instead of full grid search, this generates experiments where each one
    varies only ONE field from the baseline (one-factor-at-a-time).

    Args:
        base_cfg: Base configuration to use. If None, uses default ExperimentConfig.
        experimental_fields: Dict mapping field names to (baseline, [alternatives]).
                           If None, uses EXPERIMENTAL_FIELDS.
        include_baseline: Whether to include the baseline config as first experiment.

    Returns:
        List of ExperimentConfig instances for ablation study.
    """
    if base_cfg is None:
        base_cfg = ExperimentConfig()

    if experimental_fields is None:
        experimental_fields = EXPERIMENTAL_FIELDS

    configs = []

    # Get baseline values
    baseline_values = {field: values[0] for field, values in experimental_fields.items()}

    # Optionally add baseline experiment
    if include_baseline:
        baseline_cfg = replace(
            base_cfg,
            experiment_name="baseline",
            save_dir="results/baseline",
            **baseline_values
        )
        configs.append(baseline_cfg)

    # Generate one experiment per alternative value (one-factor-at-a-time)
    for field_name, (_, alternatives) in experimental_fields.items():
        for alt_value in alternatives:
            # Start from baseline, change only this one field
            overrides = baseline_values.copy()
            overrides[field_name] = alt_value

            # Generate descriptive experiment name
            experiment_name = f"{field_name}={alt_value}"

            new_cfg = replace(
                base_cfg,
                experiment_name=experiment_name,
                save_dir=f"results/{experiment_name}",
                **overrides
            )
            configs.append(new_cfg)

    return configs


def generate_combination_configs(
    base_cfg: ExperimentConfig | None = None,
    experimental_fields: Dict[str, List[Any]] | None = None,
) -> List[ExperimentConfig]:
    """
    Generate all combinations of experimental configurations (full grid search).

    Use this when you want to test all possible combinations of field values.

    Args:
        base_cfg: Base configuration to use. If None, uses default ExperimentConfig.
        experimental_fields: Dict mapping field names to list of values to try.

    Returns:
        List of ExperimentConfig instances for each combination.
    """
    if base_cfg is None:
        base_cfg = ExperimentConfig()

    if experimental_fields is None:
        # Convert EXPERIMENTAL_FIELDS to flat list format
        experimental_fields = {
            field: [baseline] + alts
            for field, (baseline, alts) in EXPERIMENTAL_FIELDS.items()
        }

    field_names = list(experimental_fields.keys())
    field_values = list(experimental_fields.values())

    configs = []
    for combination in product(*field_values):
        overrides = dict(zip(field_names, combination))
        name_parts = [f"{k}={v}" for k, v in overrides.items()]
        experiment_name = "_".join(name_parts)

        new_cfg = replace(
            base_cfg,
            experiment_name=experiment_name,
            save_dir=f"results/{experiment_name}",
            **overrides
        )
        configs.append(new_cfg)

    return configs


def run_all_experiments(
    experimental_fields: Dict[str, tuple[Any, List[Any]]] | None = None,
    use_combinations: bool = False,
    include_baseline: bool = True,
):
    """
    Run ablation-style experiments (one-factor-at-a-time by default).

    Args:
        experimental_fields: Dict mapping field names to (baseline, [alternatives]).
                           If None, uses EXPERIMENTAL_FIELDS.
        use_combinations: If True, run full grid search instead of ablation.
        include_baseline: Whether to include baseline experiment.
    """
    base_cfg = ExperimentConfig()
    set_seed(base_cfg.seed)

    tokenizer = get_tokenizer()

    print("\n=== Building datasets ===")
    train_chunks = make_mixed_chunks("train", base_cfg, tokenizer)
    val_chunks = make_mixed_chunks("validation", base_cfg, tokenizer)

    train_ds = MixedLMDataset(train_chunks)
    val_ds = MixedLMDataset(val_chunks)

    # Read baseline and router metrics from previous runs
    base_metrics = MetricsTracker.load("results/baseline_metrics.json")
    router_metrics = MetricsTracker.load("results/router_metrics.json")

    # Generate experiment configurations
    if use_combinations:
        if experimental_fields is None:
            flat_fields = {f: [b] + a for f, (b, a) in EXPERIMENTAL_FIELDS.items()}
        else:
            flat_fields = {f: [b] + a for f, (b, a) in experimental_fields.items()}
        configs = generate_combination_configs(base_cfg, flat_fields)
    else:
        configs = generate_experiment_configs(
            base_cfg,
            experimental_fields,
            include_baseline=include_baseline
        )

    print(f"\n=== Running {len(configs)} experiments ===")
    for i, cfg in enumerate(configs, 1):
        print(f"  {i}. {cfg.experiment_name}")

    all_results = {}
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment: {cfg.experiment_name}")
        metrics = run_single_experiment(
            cfg=cfg,
            tokenizer=tokenizer,
            train_ds=train_ds,
            val_ds=val_ds,
            base_metrics=base_metrics,
            router_metrics=router_metrics,
        )
        all_results[cfg.experiment_name] = metrics

    print(f"\n{'='*60}")
    print(f"=== All {len(configs)} experiments completed ===")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run curriculum learning experiments")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all ablation experiments (one-factor-at-a-time)"
    )
    parser.add_argument(
        "--combinations",
        action="store_true",
        help="Run full grid search of all combinations (warning: many experiments!)"
    )
    parser.add_argument(
        "--field",
        type=str,
        action="append",
        help="Run experiments for specific field(s) only. Can be specified multiple times. "
             f"Options: {', '.join(EXPERIMENTAL_FIELDS.keys())}"
    )
    parser.add_argument(
        "--profile",
        type=str,
        help="Run a predefined experiment profile (e.g., final_presentation)"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip the baseline experiment"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments that would be run without actually running them"
    )
    args = parser.parse_args()

    if args.list:
        # Just list the experiments
        if args.profile and args.field:
            print("Error: --profile and --field cannot be used together.")
            exit(1)

        try:
            profile_fields = get_profile_fields(args.profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            exit(1)
        if args.field:
            selected_fields = {k: v for k, v in EXPERIMENTAL_FIELDS.items() if k in args.field}
        elif profile_fields is not None:
            selected_fields = profile_fields
        else:
            selected_fields = EXPERIMENTAL_FIELDS

        if args.combinations:
            flat_fields = {f: [b] + a for f, (b, a) in selected_fields.items()}
            configs = generate_combination_configs(experimental_fields=flat_fields)
        else:
            configs = generate_experiment_configs(
                experimental_fields=selected_fields,
                include_baseline=not args.no_baseline
            )

        print(f"\n=== {len(configs)} experiments would be run ===\n")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.experiment_name}")
        print()

    elif args.all or args.field or args.combinations or args.profile:
        if args.profile and args.field:
            print("Error: --profile and --field cannot be used together.")
            exit(1)

        try:
            profile_fields = get_profile_fields(args.profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            exit(1)
        if args.field:
            selected_fields = {k: v for k, v in EXPERIMENTAL_FIELDS.items() if k in args.field}
            if not selected_fields:
                print(f"Error: No valid fields specified. Available: {list(EXPERIMENTAL_FIELDS.keys())}")
                exit(1)
        elif profile_fields is not None:
            selected_fields = profile_fields
        else:
            selected_fields = None

        run_all_experiments(
            experimental_fields=selected_fields,
            use_combinations=args.combinations,
            include_baseline=not args.no_baseline
        )
    else:
        run_experiment()
