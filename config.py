# config.py
"""
Central configuration for all curriculum learning experiments.

Two dataclasses:
  - Config: base training hyperparameters (model size, global_batch_size, lr, etc.)
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
import yaml
from transformers import AutoConfig

@dataclass
class Config:
    # Data
    block: int = 256
    max_chunks: int = 500_000 # Can be overriden by -1 

    # Model
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    n_chunks: int = 8
    # Hierarchical feature source
    # options: full (final transformer hidden state), embedder (token+pos
    # embeddings only, no transformer layers), layer (hidden state after
    # hierarchical_layer_index transformer layers)
    hierarchical_representation: str = "full"
    # Required when hierarchical_representation='layer'. 0 = embeddings only
    # (same as 'embedder'), n_layers = final layer (same as 'full'); anything
    # in between reads out an intermediate layer's hidden state.
    hierarchical_layer_index: int | None = None

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
    global_batch_size: int = 16
    pool_mult: int = 5
    epochs: int = 10
    # Stop training once total_tokens_seen reaches this many tokens, cutting
    # a run short mid-epoch if needed (logging still fires normally for the
    # terminating step/epoch). None = unlimited, bounded only by cfg.epochs
    # as before.
    max_tokens: int | None = None
    lr_lm: float = 3e-4 # Peak LR for the LM optimizer -- see lr_schedule below.
    # LM optimizer: 'adamw' (default, current behavior) or 'muon' (OPUS-style hybrid --
    # Muon on 2D matrix weights inside model.transformer_blocks(), AdamW on everything else;
    # see utils/muon_optimizer.py). lr_lm is reused as the AdamW sub-group's LR when
    # lm_optimizer='muon'; lr_muon is the Muon sub-group's LR (unused otherwise).
    lm_optimizer: str = "adamw"
    lr_muon: float = 0.02
    # LM learning-rate schedule, applied as a multiplier on lr_lm (and lr_muon's sub-group
    # when lm_optimizer='muon'; lr_router is unaffected):
    #   'fixed' -- constant at the configured peak throughout training (current behavior).
    #   'wsd'   -- Warmup-Stable-Decay (Hägele et al., NeurIPS 2024 "Scaling Laws and
    #              Compute-Optimal Training Beyond Fixed Training Durations"): linear warmup,
    #              held at peak through the stable phase, then their (1-sqrt) cooldown down to
    #              lr_min_ratio * peak. See utils/lr_scheduler.py.
    lr_schedule: str = "fixed"
    lr_warmup_steps: float = 300  # number of steps spent on linear warmup
    lr_decay_frac: float = 0.2   # fraction of total steps spent cooling down; the paper finds
    # the benefit plateaus here, though 0.05 with the (1-sqrt) shape still nearly matches a
    # length-matched cosine schedule if the cooldown's own compute cost needs to stay small.
    lr_min_ratio: float = 0.0    # floor as a fraction of peak lr at the end of decay
    # Global gradient-norm clip applied to the LM's parameters before opt_lm.step(), for any
    # lm_optimizer. None = no clipping (current behavior).
    grad_clip_norm: float | None = None
    lr_router: float = 1e-3
    temp: float = 1.0
    lambda_ent: float = 0.005
    lambda_router: float = 0.1

    # System
    seed: int = 0
    device: str = ""
    # Background worker processes for the training/eval DataLoaders (data.py's
    # make_pool_loader/make_baseline_loader, and evaluate()/evaluate_per_domain()
    # in training.py). 0 = load in the main process (safe default -- sweeps in
    # parallel_experiments.py run several experiment subprocesses concurrently
    # on one GPU without budgeting CPU workers, so raising this multiplies
    # across however many are running at once). Raise it for standalone runs
    # to overlap next-batch loading with GPU compute.
    dataloader_num_workers: int = 0

    # Distributed. Defaults are the single-process case; train_ddp.py
    # overrides these after torch.distributed.init_process_group().
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    # How to parallelize the LM across those ranks. Ignored at world_size 1.
    #   'DDP'  — replicate the whole model on every rank. Simplest and
    #            fastest whenever a replica fits.
    #   'FSDP' — FSDP2 (fully_shard): shard parameters, gradients and
    #            optimizer state across ranks. The option for models whose
    #            DDP replica does NOT fit — GPT2-XL is ~1.5B params, so an
    #            fp32 replica plus AdamW's two moments is ~24 GB per rank
    #            before activations.
    # See utils/distributed_utils.wrap_model().
    distributed: str = "DDP"

    # Parameter dtype for FSDP2's MixedPrecisionPolicy ('bf16' | 'fp16' |
    # 'none'); gradient reduction stays fp32 regardless. Ignored unless
    # distributed == 'FSDP'.
    # NOTE this changes numerics: FSDP runs at the 'bf16' default are not
    # directly comparable with existing fp32 DDP ablation results. Set
    # 'none' when a comparison has to be like-for-like.
    fsdp_mixed_precision: str = "bf16"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "curriculum-learning-final"
    wandb_entity: str | None = None
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

        if self.distributed not in ("DDP", "FSDP"):
            raise ValueError(
                f"distributed={self.distributed!r} is not supported (expected "
                f"'DDP' or 'FSDP')."
            )

        if self.fsdp_mixed_precision not in ("bf16", "fp16", "none"):
            raise ValueError(
                f"fsdp_mixed_precision={self.fsdp_mixed_precision!r} is not "
                f"supported (expected 'bf16', 'fp16' or 'none')."
            )

        if self.lm_optimizer not in ("adamw", "muon"):
            raise ValueError(
                f"lm_optimizer={self.lm_optimizer!r} is not supported (expected 'adamw' or 'muon')."
            )

        if self.lr_schedule not in ("fixed", "wsd"):
            raise ValueError(
                f"lr_schedule={self.lr_schedule!r} is not supported (expected 'fixed' or 'wsd')."
            )

        if self.lr_schedule == "wsd" and not (0.0 <= self.lr_warmup_steps and 0.0 < self.lr_decay_frac
                                               and self.lr_decay_frac < 1.0):
            raise ValueError(
                "lr_schedule='wsd' requires 0 <= lr_warmup_frac, 0 < lr_decay_frac, and "
                f"lr_decay_frac < 1.0 (got warmup_frac={self.lr_warmup_steps}, "
                f"decay_frac={self.lr_decay_frac})."
            )

        if self.lm_optimizer == "muon" and self.distributed == "FSDP":
            raise ValueError(
                "lm_optimizer='muon' is not supported with distributed='FSDP': FSDP2 shards "
                "parameters/gradients as DTensors, and Muon's Newton-Schulz orthogonalization "
                "(X @ X.T over a row-shard) is not equivalent to the same operation on the full "
                "matrix. checkpoint saving (_save_checkpoint) also has no FSDP2 unwrap path. Use "
                "distributed='DDP', or implement FSDP2-aware support for both before lifting this."
            )

        if self.hierarchical_representation == "layer":
            if self.hierarchical_layer_index is None:
                raise ValueError(
                    "hierarchical_representation='layer' requires "
                    "hierarchical_layer_index to be set."
                )
            if not (0 <= self.hierarchical_layer_index <= self.n_layers):
                raise ValueError(
                    f"hierarchical_layer_index={self.hierarchical_layer_index} "
                    f"out of range for n_layers={self.n_layers} (expected 0.."
                    f"{self.n_layers})."
                )

        if getattr(self, "use_curriculum_ratio_schedule", False):
            if not (0.0 < self.curriculum_ratio_min <= self.curriculum_ratio_initial <= 1.0):
                raise ValueError(
                    "use_curriculum_ratio_schedule requires "
                    "0 < curriculum_ratio_min <= curriculum_ratio_initial <= 1 "
                    f"(got min={self.curriculum_ratio_min}, "
                    f"initial={self.curriculum_ratio_initial})."
                )

    @property
    def pool(self) -> int:
        return self.pool_mult * self.global_batch_size

    @property
    def per_rank_batch_size(self) -> int:
        """global_batch_size split evenly across ranks -- the number of
        samples each GPU actually draws/selects per step under DDP.
        world_size=1 (the default) leaves this equal to global_batch_size.
        Any remainder (global_batch_size not a multiple of world_size) is
        dropped, same truncate-to-fit approach PooledBatchSampler already
        uses for pool sharding."""
        per_rank = self.global_batch_size // self.world_size
        if per_rank < 1:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} is smaller than "
                f"world_size={self.world_size}: each rank would get 0 samples per step."
            )
        return per_rank

    @property
    def per_rank_pool_size(self) -> int:
        """cfg.pool (pool_mult * global_batch_size) split evenly across
        ranks -- the number of candidate samples each GPU actually pulls per
        step for pool-based feature extraction (train_router_experiments/
        train_aux_baseline's make_pool_loader) and pool-windowed baseline
        training (train_baseline's make_baseline_loader/PooledBatchSampler).
        Dividing both pool and batch by the same world_size preserves the
        pool_mult ratio (per_rank_pool_size / per_rank_batch_size ==
        pool_mult) at any GPU count, exactly like per_rank_batch_size.
        world_size=1 (the default) leaves this equal to cfg.pool."""
        per_rank = self.pool // self.world_size
        if per_rank < 1:
            raise ValueError(
                f"pool={self.pool} is smaller than world_size={self.world_size}: "
                f"each rank would get 0 candidates per step."
            )
        return per_rank

    
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

    # None = derive from experiment_name in __post_init__ below. Computed
    # once at class-definition time from the *default* experiment_name, so a
    # plain string default here would silently ignore any override of
    # experiment_name (constructor kwarg, dataclasses.replace(), or YAML).
    wandb_project: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project is None:
            self.wandb_project = f"{self.experiment_name}"

        if self.reward_signal in ("difficulty_weighted", "combined") and len(self.dataset_list) != 2:
            raise ValueError("In order to use difficulty scoring, you need to use 2 datasets, the first one "
                             "being the easy one and the second one the hard one.")

        if self.distributed == "FSDP":
            # Both of these read raw per-rank .grad tensors off the LM, which
            # FSDP2 does not leave lying around: gradients are reduce-scattered
            # into DTensor shards during backward, so no rank ever holds this
            # rank's own complete gradient. Rejected outright rather than
            # silently scored against the wrong tensor -- under DDP these
            # signals deliberately use no_sync() to keep .grad local (see
            # rl_training.train_router_experiments), and there is no FSDP2
            # equivalent that preserves that meaning.
            if self.reward_signal in ("gradient_norm", "gradient_alignment"):
                raise ValueError(
                    f"reward_signal={self.reward_signal!r} is not supported with "
                    "distributed='FSDP': it needs this rank's own unreduced "
                    "gradients, but FSDP2 reduce-scatters them into shards during "
                    "backward. Use distributed='DDP', or a reward_signal that "
                    "doesn't read .grad (e.g. 'loss_improvement')."
                )
            if self.reward_signal == "greats_score":
                raise ValueError(
                    "reward_signal='greats_score' is not supported with "
                    "distributed='FSDP': GhostSuite's per-sample-gradient hooks "
                    "walk named_modules() and assume ordinary local parameter "
                    "tensors, but FSDP2 replaces them with sharded DTensors. Use "
                    "distributed='DDP'."
                )


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

        if self.greats_diversity_term:
            if self.reward_signal != "greats_score":
                raise ValueError(
                    "greats_diversity_term=True requires reward_signal='greats_score' -- it only "
                    "modifies that reward's computation."
                )
            if not self.greats_log_grad_norms:
                raise ValueError(
                    "greats_diversity_term=True requires greats_log_grad_norms=True: the "
                    "redundancy penalty needs per-sample train-gradient norms (sum_i ||g_i||^2), "
                    "which the engine only computes when log_grad_norms is enabled."
                )

        if self.router_freeze_progress is not None and not (0.0 <= self.router_freeze_progress <= 1.0):
            raise ValueError(
                f"router_freeze_progress={self.router_freeze_progress} must be in "
                "[0, 1] (a fraction of total training progress), or None to disable."
            )

        if self.router_update_every < 1:
            raise ValueError(
                f"router_update_every={self.router_update_every} must be >= 1 "
                "(1 = update every step)."
            )

        if self.use_original_sequence and self.feature_cache_epochs > 0:
            # build_feature_cache() (rl_training.py) always caches
            # extract_hierarchical_hidden() output, so the cached-features
            # branch in the training loop would silently ignore
            # use_original_sequence and feed the router hierarchical hidden
            # states instead of the raw token sequence.
            raise ValueError(
                "use_original_sequence=True is incompatible with "
                f"feature_cache_epochs={self.feature_cache_epochs} (>0): the "
                "feature cache only ever stores hierarchical hidden states, "
                "so caching would silently override use_original_sequence. "
                "Set feature_cache_epochs=0 or use_original_sequence=False."
            )

        if self.router_feature_source not in ("features", "own_embeddings"):
            raise ValueError(
                f"router_feature_source={self.router_feature_source!r} must be "
                "'features' or 'own_embeddings'."
            )

        if self.router_feature_source == "own_embeddings":
            if self.feature_cache_epochs > 0:
                # Same trap as use_original_sequence above, plus a worse one:
                # the router's tables are *trained*, so their output changes
                # every step and could never be cached even in principle.
                raise ValueError(
                    "router_feature_source='own_embeddings' is incompatible "
                    f"with feature_cache_epochs={self.feature_cache_epochs} "
                    "(>0): the router's embeddings are learned, so their "
                    "output changes every step and cannot be cached. "
                    "Set feature_cache_epochs=0."
                )
            if self.use_original_sequence:
                raise ValueError(
                    "router_feature_source='own_embeddings' and "
                    "use_original_sequence=True both claim the router's input: "
                    "the former embeds the token ids, the latter feeds them as "
                    "raw floats. Set use_original_sequence=False."
                )
            if self.block % self.n_chunks != 0:
                raise ValueError(
                    f"router_feature_source='own_embeddings' needs block="
                    f"{self.block} divisible by n_chunks={self.n_chunks} "
                    "(the router chunk-pools its embeddings the same way "
                    "extract_hierarchical_hidden does)."
                )
            if self.router_architecture == "random":
                raise ValueError(
                    "router_feature_source='own_embeddings' has no meaning "
                    "with router_architecture='random' (no router is built, so "
                    "there are no embeddings to learn). Set one or the other."
                )

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

    split_dataset: bool = False # Needs to be true when used with SlimPajama. Also overrides dataset proportions.
    split_column: str = "" # Make sure a split column is specified to create different datasets
    dataset_list: list[str] = field(default_factory=list)  # All datasets used
    dataset_proportions: list[str] = field(default_factory=list) # The proportion for datasets in order

    # --- tokenization (build time; see tokenization.py / build_dataset_cache.py) ---
    # These are the ONLY dataset fields baked into the on-disk cache key
    # (utils.shared_dataset.dataset_signature), together with dataset_list /
    # split_dataset / split_column. Changing them means re-tokenizing.
    #
    # Tokenizer the cache is written with. Must match the student LM's
    # vocabulary: keep "gpt2" for tiny_gpt / GPT-2 checkpoints, set it to
    # hf_model_name when training a model with a different vocabulary.
    tokenizer_name: str = "gpt2"
    # Source rows to read per split during tokenization (-1 = the whole split).
    # This is the cap that used to be conflated with max_chunks: max_chunks now
    # only limits how many (block+1)-token windows training uses, and is applied
    # at read time, so it can be changed without rebuilding the cache.
    max_documents: int = -1
    # Rows streamed when auto-discovering domains for split_dataset mode.
    domain_discovery_rows: int = 100_000

    single_dataset_val_split: float = 0.05 # Confused what this is 

    # External pre-computed embeddings (e.g. epfml/FineWeb-HQ).
    # The HuggingFace dataset must have a 'text' and an 'embeddings' column.
    # Only used when len(dataset_list)=1.
    use_external_embeddings: bool = False
    external_embeddings_dataset: str = "epfml/FineWeb-HQ"
    external_embedding_dim: int = 768

    # Sentence-embedder router features: a frozen sentence-transformers model
    # encodes each training window's decoded text once, up front (see
    # utils.sentence_embedder.build_sentence_embeddings), and the result is
    # concatenated onto the router's other features every step via
    # TokenizedCorpus.embeddings -- same plug point train_router_experiments /
    # train_aux_baseline (rl_training.py) already use for use_external_embeddings
    # above, which this option is independent of.
    # Empty model name = disabled.
    sentence_embedder_model: str = ""
    sentence_embedder_dim: int = 768
    # These are small (100-400M param) encoders relative to a modern GPU, so
    # this can go much higher than a training batch size would; 512 is a safe
    # default and can be raised further on GPUs with more headroom.
    sentence_embedder_batch_size: int = 512
    # Optional .pt path to persist/reload the embedding cache. Safe to reuse
    # across runs and even rebuild indefinitely: the encoder is frozen, so a
    # window's embedding never changes, unlike the periodically-rebuilt
    # hierarchical feature cache (rl_training.build_feature_cache).
    sentence_embedder_cache_path: str = ""

    # Router architecture
    router_architecture: str = "attention"  # options: attention, linear, mlp
    router_n_heads: int = 1  # >1 enables MultiHeadAttentionRouter
    router_n_layers: int = 1 # 1 -> Only one attention layer to catch correlations inbetween the batch. 
    
    # Router features
    enable_text_stat: bool = True
    enable_text_hierarchical: bool = True
    use_original_sequence: bool = False # This uses the original tokens sequence, not passed through the model.

    # Where the router's input representation comes from.
    #   'features'       — the concatenated feature groups above, built by
    #                      extract_router_features() (default, current behavior)
    #   'own_embeddings' — the router owns token + positional embedding tables
    #                      and learns them from the policy-gradient signal,
    #                      reading the raw token sequence directly (models/
    #                      router.py EmbeddingRouter). The LM is never touched.
    #
    # Distinct from hierarchical_representation='embedder', which chunk-pools
    # the *LM's* embedding tables under torch.no_grad() -- frozen, and shaped
    # by the LM's own objective. Both pool identically (cfg.n_chunks segments,
    # mean-pooled, concatenated), so the pair is a controlled comparison of
    # learned-by-the-router vs. borrowed-from-the-LM embeddings.
    router_feature_source: str = "features"  # options: features, own_embeddings
    # Width of the router's own embedding tables. Only read when
    # router_feature_source='own_embeddings'. The router head then sees
    # n_chunks * router_embed_dim inputs.
    router_embed_dim: int = 256

    # Training algorithm
    training_algorithm: str = "reinforce"  # options: reinforce, grpo, ppo

    # Router freeze: once training progress (global_step / total_steps)
    # reaches this fraction, stop updating the router (no more REINFORCE/
    # GRPO/PPO policy-gradient steps) but keep using its current, now-frozen
    # weights to score/select samples for the rest of training -- an
    # ablation for whether continued router training helps past some point,
    # vs. an early-converged router already being "good enough". The LM
    # itself keeps training normally throughout; only the router's own
    # parameter updates stop. None = never freeze (current behavior).
    router_freeze_progress: float | None = None

    # Router update cadence: only compute the reward signal's extra
    # loss_after forward pass and perform the router's policy-gradient
    # update (REINFORCE/GRPO/PPO) once every router_update_every LM training
    # steps. On the other steps the router still scores/selects the pool
    # with its current weights each step (selection logic unchanged) --
    # this only throttles how often it *learns* from a reward, trading
    # update frequency for the compute of that extra forward pass. Composes
    # with router_freeze_progress above: once frozen, the router never
    # updates regardless of this value. 1 = every step (current behavior,
    # default).
    router_update_every: int = 1

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
    # Second-order (Hessian-approximated-as-identity) redundancy term, added on top of the
    # first-order greats_score reward: penalizes gradient redundancy WITHIN the already-selected
    # batch. Unlike GREATS's own greedy candidate selection (examples/greats/sft/gram_scorer.py),
    # this doesn't need a candidate-candidate Gram matrix -- the router already fixed the batch, so
    # the penalty sum_{i<j in S} <g_i,g_j> is evaluated directly via
    # (||sum_i g_i||^2 - sum_i ||g_i||^2) / 2, both cheap for a fixed, already-known S. Requires
    # greats_log_grad_norms=True (for sum_i ||g_i||^2) and reward_signal='greats_score'.
    greats_diversity_term: bool = False

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

    # Curriculum-ratio schedule: instead of a fixed cfg.per_rank_batch_size,
    # the number of samples selected into the training batch each step is
    # round(ratio * pool_size), with `ratio` annealed from
    # curriculum_ratio_initial down to curriculum_ratio_min over training
    # progress. Starts weakly selective (rate/accept most of the pool) and
    # tightens into a strongly selective curriculum (only the router's
    # top few percent) by the end of training. cfg.global_batch_size is
    # unused while this is on -- see train_router_experiments() in rl_training.py.
    use_curriculum_ratio_schedule: bool = False
    curriculum_ratio_schedule: str = "linear_decay"  # same vocabulary as temp_schedule
    curriculum_ratio_initial: float = 0.9  # fraction of pool selected at progress=0
    curriculum_ratio_min: float = 0.1  # fraction of pool selected at progress=1

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

    # Non-learned control baselines (training.train_baseline: uniform random
    # selection, no router/aux_net at all). run_random_batch_baseline draws
    # cfg.global_batch_size random samples per step (same shape as the
    # router's selected batch, split across ranks like any other run);
    # run_random_pool_baseline draws cfg.pool (the router's full candidate
    # pool, unfiltered, also split across ranks) -- see utils/experiment_worker.py.
    run_random_batch_baseline: bool = False
    run_random_pool_baseline: bool = False

    # Checkpoint the trained LM (and router/aux_net, when one was trained) at
    # the end of run_single_experiment() to <scratch_dir>/checkpoints/<name>.pt.
    # Off by default: a sweep runs many experiments, and hf_pretrained models
    # like GPT2-XL are multi-GB each -- opt in per run (parallel_experiments.py
    # --save-model, or this field directly in a YAML) rather than paying that
    # disk cost for every experiment in every sweep.
    save_model_at_end: bool = False


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

    # experiment_name defaults to the YAML file's own name (without
    # extension) so a run's name/wandb_project track the config file used to
    # launch it. An explicit experiment_name: key in the YAML still takes
    # precedence over this default.
    if "experiment_name" not in overrides:
        overrides["experiment_name"] = os.path.splitext(os.path.basename(path))[0]

    # wandb_project is lazily derived from experiment_name in __post_init__,
    # but only when still None; by this point cfg already has it resolved to
    # a concrete string (from the ExperimentConfig() default above, or from
    # the caller-supplied cfg). Since experiment_name is now always being set
    # (explicitly or via the filename default above), force it back to None
    # so __post_init__ re-derives it from the new name instead of keeping the
    # stale resolved value from the old one.
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

