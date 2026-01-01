# config.py
from dataclasses import dataclass, field
import torch

@dataclass
class Config:
    # Data
    block: int = 256
    easy_samples: int = 100_000
    hard_samples: int = 5_000
    max_chunks: int = 200_000
    
    # Mix proportions
    easy_ratio: float = 0.7   # easy share in train mix (e.g., 0.7 means 70/30)

    # Data sources
    easy_source: str = "tinystories"         # currently only tinystories supported by default
    hard_source: str = "openwebtext2"        # {"openwebtext2","fineweb_edu","wikipedia","c4"}

    # cap per-source chunks (for fast sweeps)
    max_easy_chunks: int | None = None
    max_hard_chunks: int | None = None

    # Timing / profiling
    benchmark_steps: int = 200   # steps to benchmark overhead
    profile_warmup: int = 20     # ignore first warmup steps


    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
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
    
    # Ablations / Router config
    router_arch: str = "attention"   # {"attention","linear","mlp"}
    feature_mode: str = "full"       # {"full","no_hidden","no_stats","no_hier","random"}
    reward_mode: str = "progress"    # {"progress","neg_loss"}


    # GRPO/GRPU-like advantage normalization for router updates
    advantage_norm: str = "center"   # {"none", "center", "zscore"}
    advantage_clip: float | None = 5.0 
    advantage_eps: float = 1e-8
    
    
    # ---- Entropy annealing ----
    entropy_schedule: str = "constant"   # {"constant","linear","exp"}
    entropy_final_mult: float = 0.0      # final lambda = lambda_ent * entropy_final_mult
    entropy_warmup_steps: int = 0        # optional: keep lambda_ent constant for first N steps
    entropy_anneal_steps: int = 0        # if 0 => infer from epochs/steps or don't anneal

    # ---- Count-based novelty bonus ----
    novelty_bonus: float = 0.0           # eta0 (start small: 0.01–0.05 relative to reward scale)
    novelty_mode: str = "inv_sqrt"       # {"inv_sqrt","exp","first_time"}
    novelty_anneal: str = "constant"     # {"constant","linear","exp"}
    novelty_final_mult: float = 0.0      # final eta = novelty_bonus * novelty_final_mult



    # System
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    use_wandb: bool = True
    wandb_project: str = "curriculum-learning-ablation"
    wandb_entity: str | None = None
    save_dir: str = "results"
    log_every: int = 100

    @property
    def pool(self) -> int:
        return self.pool_mult * self.batch
