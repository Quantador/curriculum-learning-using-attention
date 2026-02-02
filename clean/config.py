# config.py
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Data
    block: int = 256
    easy_samples: int = 10_000
    hard_samples: int = 2_000
    max_chunks: int = 20_000

    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    n_chunks: int = 8

    # Training
    batch: int = 16
    pool_mult: int = 5
    epochs: int = 5
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
    
    easy_dataset: str = "roneneldan/TinyStories"
    hard_dataset: str = "Geralt-Targaryen/openwebtext2"
    
    
    # router choices
    router_architecture: str = "mlp"  # options: attention, linear, mlp
    
    enable_text_stat: bool = True
    enable_text_hierarchical: bool = True
    
    
    reward_signal: str = "loss_improvement"  # options: loss_improvement, neg_loss
    
    