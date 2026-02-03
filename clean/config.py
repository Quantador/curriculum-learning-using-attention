# config.py
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Data
    block: int = 256
    easy_samples: int = 10_00
    hard_samples: int = 2_00
    max_chunks: int = 20_00

    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    n_chunks: int = 8
    # Hierarchical feature source
    # options: full (transformer hidden), embedder (token+pos embeddings)
    hierarchical_representation: str = "full"

    # Training
    batch: int = 16
    pool_mult: int = 5
    epochs: int = 1
    lr_lm: float = 3e-4
    lr_router: float = 1e-3
    temp: float = 1.0
    lambda_ent: float = 0.005
    lambda_router: float = 0.1

    # System
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    use_wandb: bool = True
    wandb_project: str = "curriculum-learning"
    wandb_entity: str | None = None
    save_dir: str = "results"
    log_every: int = 100

    @property
    def pool(self) -> int:
        return self.pool_mult * self.batch
    
    
    
    
@dataclass

class ExperimentConfig(Config):
    experiment_name: str = "mlp_router_experiment"
    
    wanb_project: str = "curriculum-learning-"+experiment_name
    
    save_dir: str = "results/" + experiment_name


    # Data mixing
    easy_proportion: float = 0.7  # Proportion of easy samples in mixed chunks
    hard_proportion: float = 0.3  # Proportion of hard samples in mixed chunks

    # Dataset options by difficulty:
    # Easy:   roneneldan/TinyStories, ajibawa-2023/Children-Stories-Collection,
    #         wikipedia (simple), Salesforce/wikitext
    # Medium: Geralt-Targaryen/openwebtext2, bookcorpus, HuggingFaceFW/fineweb-edu, allenai/c4
    # Hard:   armanc/scientific_papers, CShorten/ML-ArXiv-Papers, bigcode/the-stack
    easy_dataset: str = "roneneldan/TinyStories"
    hard_dataset: str = "Geralt-Targaryen/openwebtext2"
    
    
    # Router architecture
    router_architecture: str = "mlp"  # options: attention, linear, mlp

    # Router features
    enable_text_stat: bool = True
    enable_text_hierarchical: bool = True

    # Training algorithm
    training_algorithm: str = "reinforce"  # options: reinforce, grpo, ppo

    # Reward signal options:
    #   - loss_improvement: (loss_before - loss_after).clamp(0) - reward progress
    #   - neg_loss: -loss_after - prefer easier samples
    #   - relative_improvement: (loss_before - loss_after) / loss_before - normalized
    #   - difficulty_weighted: improvement * difficulty - reward harder samples more
    #   - uncertainty_reduction: entropy_before - entropy_after - reward confidence gain
    #   - combined: weighted sum of multiple signals
    reward_signal: str = "loss_improvement"

    # Weights for combined reward signal
    reward_weight_improvement: float = 1.0
    reward_weight_difficulty: float = 0.5
    reward_weight_uncertainty: float = 0.3

    # Selection strategy
    selection_strategy: str = "topk"  # options: topk, sample, epsilon_greedy
    epsilon_greedy: float = 0.1  # epsilon for epsilon_greedy selection

    # Baseline for variance reduction (REINFORCE)
    baseline_type: str = "batch_mean"  # options: batch_mean, moving_avg, none
    baseline_momentum: float = 0.99  # momentum for moving_avg baseline

    # Temperature schedule
    temp_schedule: str = "fixed"  # options: fixed, linear_decay, cosine_decay
    temp_min: float = 0.1  # minimum temperature for decay schedules

    # Entropy coefficient schedule
    # Options: fixed, linear_decay, cosine_decay, exponential_decay, cyclic, adaptive
    entropy_schedule: str = "fixed"
    lambda_ent_min: float = 0.001  # minimum entropy coefficient for decay
    entropy_cycle_length: int = 1000  # steps per cycle for cyclic schedule

    # Entropy formulation
    # Options: shannon, renyi, tsallis, kl_uniform
    entropy_type: str = "shannon"
    entropy_alpha: float = 2.0  # Rényi entropy parameter (alpha > 0, != 1)
    entropy_q: float = 2.0  # Tsallis entropy parameter (q > 0)

    # Entropy targeting (SAC-style)
    # When enabled, automatically adjusts lambda_ent to maintain target entropy
    use_entropy_targeting: bool = False
    target_entropy_ratio: float = 0.5  # target = ratio * max_entropy
    entropy_lr: float = 1e-3  # learning rate for entropy coefficient

    # Coverage regularization
    # Encourages the router to select diverse samples over time
    use_coverage_regularization: bool = False
    coverage_type: str = "count"  # options: count, recency, uncertainty
    lambda_coverage: float = 0.01  # weight for coverage regularization
    coverage_decay: float = 0.99  # decay factor for recency-based coverage
    coverage_temperature: float = 1.0  # temperature for coverage bonus

    # PPO specific
    ppo_clip: float = 0.2  # PPO clipping parameter
    ppo_epochs: int = 4  # number of PPO update epochs per batch

    # GRPO specific
    grpo_group_size: int = 4  # number of groups for GRPO
    
    
