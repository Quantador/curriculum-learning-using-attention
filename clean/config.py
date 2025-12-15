# config.py
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Data
    block: int = 256
    easy_samples: int = 30000
    hard_samples: int = 5000
    max_chunks: int = 50000

    # Model
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1024
    n_chunks: int = 8

    # Training
    batch: int = 16
    pool_mult: int = 5
    epochs: int = 10
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
