# config.py
"""
Central configuration for all curriculum learning experiments.

Two dataclasses:
  - Config: base training hyperparameters (model size, batch, lr, etc.)
  - ExperimentConfig: extends Config with all experiment-specific knobs —
    router architecture, training algorithm (REINFORCE/GRPO/PPO), reward
    signal, entropy formulation, dataset choices, coverage regularization,
    PPO/GRPO params, and feature caching settings.

Typical usage:
    cfg = ExperimentConfig()           # sensible defaults
    cfg = replace(cfg, epochs=5, ...)  # override via dataclasses.replace
    cfg = load_config_from_yaml("configs/my_run.yaml")  # override via YAML file
"""
from dataclasses import dataclass, field, fields, replace
import os
import torch
import yaml
from transformers import AutoConfig

@dataclass
class Config:
    # Data
    block: int = 256
    easy_samples: int = 100_000
    hard_samples: int = 20_000
    max_chunks: int = 500_000

    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    n_chunks: int = 8
    # Hierarchical feature source
    # options: full (transformer hidden), embedder (token+pos embeddings)
    hierarchical_representation: str = "full"

    # Student LM architecture, built via model.build_model():
    #   'tiny_gpt'      — small from-scratch TransformerEncoder (default)
    #   'hf_pretrained' — HuggingFace architecture named by hf_model_name
    #                     (e.g. "Qwen/Qwen3-1.7B"), randomly initialized and
    #                     trained from scratch, not fine-tuned from checkpoint
    # When 'hf_pretrained' is used, get_tokenizer(cfg.hf_model_name) must be
    # used too, since token ids must match the model's vocabulary.
    model_type: str = "tiny_gpt"
    hf_model_name: str = "Qwen/Qwen3-1.7B"

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

    # Distributed (DDP). Defaults are the single-process case; train_ddp.py
    # overrides these after torch.distributed.init_process_group().
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    # Logging
    use_wandb: bool = True
    wandb_project: str = "curriculum-learning-final"
    wandb_entity: str | None = None
    save_dir: str = "results"
    log_every: int = 100

    # Set automatically by load_config_from_yaml() to the source YAML path;
    # not a hyperparameter. When set, training loops upload this file to the
    # W&B run (see wandb.save() calls in training.py / rl_training.py) so the
    # exact override file used for the run is attached alongside its metrics.
    config_path: str | None = None

    def __post_init__(self):
        # d_model/n_layers/n_heads/d_ff describe TinyGPT's architecture, but
        # for 'hf_pretrained' the real architecture comes from the checkpoint
        # itself (see HFCausalLM in models/model.py, which reads hf_config.*
        # and ignores these fields entirely). Code that runs before the model
        # is built — e.g. get_router_feature_dim() in models/router.py, which
        # sizes the router from cfg.n_chunks * cfg.d_model — has no other way
        # to know the checkpoint's real hidden size, so these are overwritten
        # here to keep them truthful rather than left at the tiny_gpt defaults.
        if self.model_type == "hf_pretrained":
            hf_cfg = AutoConfig.from_pretrained(self.hf_model_name)
            self.d_model = hf_cfg.hidden_size
            self.n_layers = hf_cfg.num_hidden_layers
            self.n_heads = hf_cfg.num_attention_heads
            self.d_ff = getattr(hf_cfg, "intermediate_size", self.d_ff)

    @property
    def pool(self) -> int:
        return self.pool_mult * self.batch
    
    
    
    
@dataclass

class ExperimentConfig(Config):
    """
    Extends Config with all experiment-specific hyperparameters.

    All fields have sensible defaults. Override via dataclasses.replace()
    to generate ablation configurations without mutating the base config.

    Key field groups:
      - Dataset: easy_dataset/hard_dataset for mixed mode, single_dataset
        for single-source mode; use_external_embeddings for pre-computed vectors
      - Router: router_architecture ('attention', 'mlp', 'linear'),
        router_n_heads (>1 enables MultiHeadAttentionRouter)
      - Training algorithm: training_algorithm ('reinforce', 'grpo', 'ppo')
      - Reward: reward_signal (8 options documented in field comments)
      - Entropy: entropy_type, entropy_schedule, use_entropy_targeting
      - Coverage: use_coverage_regularization, coverage_type
      - Caching: feature_cache_epochs (0 = disabled)
    """
    experiment_name: str = "presentation_experiment"

    # None = derive from experiment_name in __post_init__ below. Fields are
    # computed once at class-definition time from the *default* experiment_name,
    # so a plain string default here would silently ignore any override of
    # experiment_name (constructor kwarg, dataclasses.replace(), or YAML).
    wandb_project: str | None = None
    save_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project is None:
            self.wandb_project = f"curriculum-learning-{self.experiment_name}"
        if self.save_dir is None:
            self.save_dir = f"results/{self.experiment_name}"

        if self.reward_signal == "greats_score":
            # GREATS-style ghost gradient-dot-product scoring (GhostSuite/ghostEngines,
            # via GhostEngineManager). Only supported for GPT-2-family HF checkpoints:
            # the scorer's per-sample-gradient hooks match nn.Linear / nn.Embedding /
            # nn.LayerNorm / HF Conv1D by EXACT type, not isinstance. TinyGPT's attention
            # is nn.MultiheadAttention (its in_proj_weight has no leaf-module hook target
            # at all, and out_proj is a Linear *subclass* that fails the exact-type
            # check), and Qwen3 uses a custom RMSNorm plus its own Linear stack — both
            # leave most/all of the model's gradient invisible to the scorer.
            if self.model_type != "hf_pretrained":
                raise ValueError(
                    "reward_signal='greats_score' requires model_type='hf_pretrained' "
                    "with a GPT-2-family checkpoint; GhostSuite's ghost gradient-dot-"
                    f"product hooks can't see {self.model_type!r}'s attention layers."
                )
            hf_arch = AutoConfig.from_pretrained(self.hf_model_name).model_type
            if hf_arch != "gpt2":
                raise ValueError(
                    f"reward_signal='greats_score' only supports GPT-2-family "
                    f"checkpoints; hf_model_name={self.hf_model_name!r} resolves to "
                    f"architecture {hf_arch!r}. GhostSuite's hooks match nn.Linear/"
                    "nn.Embedding/nn.LayerNorm/HF Conv1D by exact type, so e.g. "
                    "Qwen3's RMSNorm layers are invisible to the scorer. Pick a "
                    "GPT-2 checkpoint (gpt2, gpt2-medium, gpt2-large, ...) or a "
                    "different reward_signal."
                )


    # Data mixing
    easy_proportion: float = 0.7  # Proportion of easy samples in mixed chunks
    hard_proportion: float = 0.3  # Proportion of hard samples in mixed chunks

    # Dataset options by difficulty (see DATASET_REGISTRY in data.py):
    # Easy:         roneneldan/TinyStories
    #               ajibawa-2023/Children-Stories-Collection
    #               Salesforce/wikitext  (wikitext-103-raw-v1)
    # Medium/Hard:  Geralt-Targaryen/openwebtext2
    #               HuggingFaceFW/fineweb-edu
    #               allenai/c4
    # Hard:         armanc/scientific_papers
    #               CShorten/ML-ArXiv-Papers
    # Unstructured: HuggingFaceFW/fineweb  (use with use_single_dataset=True)
    easy_dataset: str = "roneneldan/TinyStories"
    hard_dataset: str = "Geralt-Targaryen/openwebtext2"

    # Single-dataset mode (no easy/hard split).
    # When True, trains on one dataset only; easy/hard fields above are ignored.
    use_single_dataset: bool = True
    single_dataset: str = "HuggingFaceFW/fineweb"
    single_dataset_samples: int = 120_000
    single_dataset_val_split: float = 0.05

    # External pre-computed embeddings (e.g. epfml/FineWeb-HQ).
    # The HuggingFace dataset must have a 'text' and an 'embeddings' column.
    # Only used when use_single_dataset=True.
    use_external_embeddings: bool = False
    external_embeddings_dataset: str = "epfml/FineWeb-HQ"
    external_embedding_dim: int = 768
    
    # Router architecture
    router_architecture: str = "attention"  # options: attention, linear, mlp
    router_n_heads: int = 1  # >1 enables MultiHeadAttentionRouter

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
    #   - gradient_norm: ||∇θ L_LM(S_t)|| - batch gradient magnitude
    #   - gradient_alignment: <g_t, g_ema> - alignment with EMA gradient
    #   - combined: weighted sum of multiple signals
    #   - greats_score: sum of ghost gradient-dot-product scores <g_i, g_val> over
    #     the selected batch (one scalar shared by every sample) - GPT-2-family HF
    #     checkpoint only (validated in __post_init__). A constant reward across
    #     the batch makes baseline_type='batch_mean' always cancel to zero
    #     advantage; use baseline_type='moving_avg' instead.
    reward_signal: str = "loss_improvement"

    # GhostSuite/ghostEngines scoring knobs, used only when reward_signal='greats_score'.
    greats_val_batch_size: int = 16
    greats_score_metric: str = "dot"  # options: dot, cosine (cosine forces greats_log_grad_norms)
    greats_log_grad_norms: bool = False
    greats_score_exclude_params: list[str] = field(default_factory=list)

    # Weights for combined reward signal
    reward_weight_improvement: float = 1.0
    reward_weight_difficulty: float = 0.5
    reward_weight_uncertainty: float = 0.3

    # Gradient-based reward settings
    gradient_ema_momentum: float = 0.9
    gradient_reward_clip: float | None = 10.0

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

    # Feature caching
    # 0 = disabled (recompute every step), n = rebuild cache every n epochs
    # Cache is never built during epoch 0 (features are noise early on)
    feature_cache_epochs: int = 0
    feature_cache_batch_size: int = 64
    # Path to save/load the cache on disk as a .pt file (fp16)
    # Empty string = keep in CPU RAM only, no disk persistence
    feature_cache_path: str = ""

    # Auxiliary network baseline.
    # When True, a supervised MLP is trained alongside the RL router.
    # It regresses directly on the observed loss-improvement signal and
    # selects samples by predicted improvement — a direct supervised
    # alternative to policy-gradient curriculum learning.
    run_aux_baseline: bool = False
    aux_net_hidden: int = 256


def load_config_from_yaml(path: str, cfg: ExperimentConfig | None = None) -> ExperimentConfig:
    """
    Apply field overrides from a YAML file on top of `cfg` (defaults to
    ExperimentConfig() if not given). YAML keys must match ExperimentConfig
    field names exactly, e.g.:

        epochs: 5
        training_algorithm: grpo
        lambda_ent: 0.01

    Only fields already present on `cfg` are accepted; an unrecognised key
    raises ValueError immediately rather than silently doing nothing (the
    likely outcome of a typo'd field name).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    if cfg is None:
        cfg = ExperimentConfig()
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}

    # save_dir/wandb_project are lazily derived from experiment_name in
    # __post_init__, but only when still None; by this point cfg already has
    # them resolved to concrete strings (from the ExperimentConfig() default
    # above, or from the caller-supplied cfg). If the YAML overrides
    # experiment_name without also overriding these, force them back to None
    # so __post_init__ re-derives from the new name instead of keeping the
    # stale resolved value from the old one.
    if "experiment_name" in overrides:
        overrides.setdefault("save_dir", None)
        overrides.setdefault("wandb_project", None)

    valid_fields = {f.name for f in fields(cfg)}
    unknown = set(overrides) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown config field(s) in {path}: {sorted(unknown)}. "
            f"Valid fields: {sorted(valid_fields)}"
        )
    cfg = replace(cfg, **overrides)
    # Always set from the real path, overriding any (unlikely) config_path
    # key the YAML file itself tried to set.
    return replace(cfg, config_path=path)

