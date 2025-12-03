
# ablation_study.py
# Comprehensive ablation study and hyperparameter sweep for curriculum learning
# Tests each component's contribution and finds optimal hyperparameters

import os
import json
import copy
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Import the original training code
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

# We'll modify the original config for each experiment
from evalRouterWandB import (
    Config, make_mixed_chunks, MixedLMDataset,
    train_baseline, train_router, MetricsTracker, DiversityTracker,
    cfg as original_cfg
)

import torch
import random
import wandb

# =====================================================================
# EXPERIMENT CONFIGURATIONS
# =====================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment variant"""
    name: str
    description: str
    
    # Component ablations
    use_hierarchical_features: bool = True
    use_text_statistics: bool = True
    use_loss_improvement_reward: bool = True
    use_curriculum_bias: bool = True
    
    # Feature variants
    n_chunks: int = 8  # For hierarchical pooling
    
    # Hyperparameters
    pool_mult: int = 5
    batch_size: int = 16
    temperature: float = 1.0
    lambda_ent: float = 0.005
    lambda_router: float = 0.1
    lr_lm: float = 3e-4
    lr_router: float = 1e-3
    
    # Training
    epochs: int = 10
    seed: int = 0

# =====================================================================
# ABLATION STUDY: Test each component's contribution
# =====================================================================

def get_ablation_experiments() -> List[ExperimentConfig]:
    """Define ablation study experiments"""
    
    base = ExperimentConfig(
        name="full_model",
        description="Full model with all components"
    )
    
    ablations = [
        base,
        
        # Component ablations
        ExperimentConfig(
            name="no_hierarchical",
            description="Remove hierarchical features (only text stats)",
            use_hierarchical_features=False,
            use_text_statistics=True,
            use_loss_improvement_reward=True,
            use_curriculum_bias=True
        ),
        
        ExperimentConfig(
            name="no_text_stats",
            description="Remove text statistics (only hierarchical)",
            use_hierarchical_features=True,
            use_text_statistics=False,
            use_loss_improvement_reward=True,
            use_curriculum_bias=True
        ),
        
        ExperimentConfig(
            name="no_features",
            description="Remove both feature types (random features)",
            use_hierarchical_features=False,
            use_text_statistics=False,
            use_loss_improvement_reward=True,
            use_curriculum_bias=True
        ),
        
        ExperimentConfig(
            name="no_loss_improvement",
            description="Use fixed reward instead of loss improvement",
            use_hierarchical_features=True,
            use_text_statistics=True,
            use_loss_improvement_reward=False,
            use_curriculum_bias=True
        ),
        
        ExperimentConfig(
            name="no_curriculum_bias",
            description="No easy→hard curriculum bias",
            use_hierarchical_features=True,
            use_text_statistics=True,
            use_loss_improvement_reward=True,
            use_curriculum_bias=False
        ),
        
        # Combined ablations
        ExperimentConfig(
            name="minimal",
            description="Minimal model (no features, no curriculum, fixed reward)",
            use_hierarchical_features=False,
            use_text_statistics=False,
            use_loss_improvement_reward=False,
            use_curriculum_bias=False
        ),
    ]
    
    return ablations

# =====================================================================
# HYPERPARAMETER SWEEP: Find optimal settings
# =====================================================================

def get_hyperparameter_sweep() -> List[ExperimentConfig]:
    """Define hyperparameter sweep experiments"""
    
    experiments = []
    
    # 1. Pool multiplier sweep (how many candidates to consider)
    for pool_mult in [3, 5, 8, 10]:
        experiments.append(ExperimentConfig(
            name=f"pool_mult_{pool_mult}",
            description=f"Pool multiplier = {pool_mult}",
            pool_mult=pool_mult
        ))
    
    # 2. Temperature sweep (exploration vs exploitation)
    for temp in [0.5, 1.0, 2.0, 3.0]:
        experiments.append(ExperimentConfig(
            name=f"temperature_{temp}",
            description=f"Temperature = {temp}",
            temperature=temp
        ))
    
    # 3. Entropy regularization sweep
    for lambda_ent in [0.0, 0.001, 0.005, 0.01, 0.02]:
        experiments.append(ExperimentConfig(
            name=f"entropy_{lambda_ent}",
            description=f"Lambda entropy = {lambda_ent}",
            lambda_ent=lambda_ent
        ))
    
    # 4. Batch size sweep
    for batch_size in [8, 16, 32]:
        experiments.append(ExperimentConfig(
            name=f"batch_{batch_size}",
            description=f"Batch size = {batch_size}",
            batch_size=batch_size,
            pool_mult=5  # Keep pool_mult*batch constant
        ))
    
    # 5. Learning rate sweep
    for lr_router in [5e-4, 1e-3, 2e-3, 5e-3]:
        experiments.append(ExperimentConfig(
            name=f"lr_router_{lr_router}",
            description=f"Router LR = {lr_router}",
            lr_router=lr_router
        ))
    
    # 6. Feature granularity sweep (hierarchical chunks)
    for n_chunks in [4, 8, 16, 32]:
        experiments.append(ExperimentConfig(
            name=f"chunks_{n_chunks}",
            description=f"Hierarchical chunks = {n_chunks}",
            n_chunks=n_chunks
        ))
    
    return experiments

# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation study and hyperparameter sweep")
    parser.add_argument('--mode', choices=['ablation', 'hyperparam', 'both'], 
                       default='ablation', help='Which experiments to run')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save-dir', default='ablation_results', help='Save directory')
    args = parser.parse_args()
    
    print("Ablation study script ready!")
    print(f"Mode: {args.mode}")
    print(f"Save dir: {args.save_dir}")
    print(f"Wandb: {not args.no_wandb}")

if __name__ == "__main__":
    main()