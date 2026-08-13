"""
Router factory and feature extraction for curriculum learning experiments.

This module bridges the language model and the RL training loop: it defines
how samples are featurized and which router architecture scores them.

Router architectures (all nn.Module, produce [B] scalar scores):
  LinearRouter   — single linear layer, fewest parameters, fastest
  MLPRouter      — two-layer MLP with GELU, more expressive
  AuxNetRouter   — supervised alternative trained with MSE to predict
                   loss improvement (not policy gradient)

The primary router architectures (AttentionRouter, MultiHeadAttentionRouter)
are defined in model.py. build_router() here is the factory for all of them.

Feature extraction:
  extract_router_features() — concatenates up to three feature groups into
                              the vector fed to the router
  get_router_feature_dim()  — computes the expected input dimension so the
                              router can be instantiated before training starts
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from config import ExperimentConfig

from models.model import (
    TinyGPT,
    AttentionRouter,
    MultiHeadAttentionRouter,
    compute_text_statistics,
    extract_hierarchical_hidden,
)


class LinearRouter(nn.Module):
    def __init__(self, d_input: int):
        super().__init__()
        self.fc = nn.Linear(d_input, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(feats).squeeze(-1)


class MLPRouter(nn.Module):
    def __init__(self, d_input: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


class AuxNetRouter(nn.Module):
    """
    Supervised alternative to the RL router — used as an ablation baseline.

    Trained with MSE loss to directly regress the observed per-sample
    loss-improvement signal, rather than via policy gradient. At inference
    time it scores samples identically to the attention router (top-k by
    predicted score), making it a controlled comparison:
      - Same features (output of extract_router_features)
      - Same selection logic (top-k in rl_training.train_aux_baseline)
      - Different training objective: MSE regression vs. REINFORCE

    Instantiate via build_router(arch='auxnet').
    Training loop: rl_training.train_aux_baseline().
    """
    def __init__(self, d_input: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)  # [B]


def extract_router_features(
    model: TinyGPT,
    X: torch.Tensor,
    cfg: ExperimentConfig,
    pad_token_id: int,
    vocab_size: int,
    external_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build the final feature vector that the router scores each candidate sample
    with. Everything the router sees is concatenated in here -- callers pass
    in raw ingredients (X, external_embedding) and get back the finished [B, F]
    tensor; no further concatenation should happen on the outside.

    Concatenates up to three optional feature groups in this order:
      1. Hierarchical hidden states  [B, n_chunks * d_model]
         Enabled by cfg.enable_text_hierarchical. Runs a transformer forward
         pass (or uses only embeddings if cfg.hierarchical_representation='embedder').
      2. Text statistics  [B, 4]
         Enabled by cfg.enable_text_stat. Cheap surface features: fill ratio,
         lexical diversity, normalised mean/std token id.
      3. External embedding  [B, D]
         Passed in by the caller as `external_embedding` -- e.g. the current
         pool's rows of TokenizedCorpus.embeddings, populated via
         cfg.use_external_embeddings or cfg.sentence_embedder_model. Required
         (must be non-None) whenever either of those flags is set.

    If no group is enabled, returns random features as a fallback
    (router learns nothing — intended only for sanity-check baselines).

    Returns [B, F] where F == get_router_feature_dim(cfg).
    """
    features = []
    if cfg.use_original_sequence:
        return X.float()
    if cfg.enable_text_hierarchical:
        hidden_feat = extract_hierarchical_hidden(model, X, cfg)  # [B, n_chunks*D]
        features.append(hidden_feat)

    if cfg.enable_text_stat:
        stats = compute_text_statistics(
            X,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            block=cfg.block,
        )
        features.append(stats)

    needs_external = getattr(cfg, "use_external_embeddings", False) or bool(
        getattr(cfg, "sentence_embedder_model", "")
    )
    if needs_external and external_embedding is None:
        raise ValueError(
            "cfg.use_external_embeddings or cfg.sentence_embedder_model is set, "
            "so extract_router_features() requires external_embedding (e.g. the "
            "current pool's TokenizedCorpus.embeddings rows)."
        )
    if external_embedding is not None:
        features.append(external_embedding)  # [B, D]

    if not features:
        print("Warning: No features enabled for router; returning random features.")
        full_dim = cfg.n_chunks * cfg.d_model + 4
        return torch.randn(X.size(0), full_dim, device=X.device)

    # .float() is load-bearing, not defensive. The router is always fp32 (it
    # is replicated, never sharded or autocast -- see
    # utils/distributed_utils.wrap_replica), but two of the three feature
    # groups arrive in reduced precision: the sentence-embedder cache is
    # stored fp16 on purpose (utils/sentence_embedder.py), and under
    # distributed='FSDP' the LM's hidden states come back in the
    # MixedPrecisionPolicy param_dtype (bf16). torch.cat type-promotes, so a
    # config with several groups enabled silently lands on fp32 and works --
    # but a single-group config ('sentence embedder alone', or hierarchical
    # alone under FSDP) hands the router a Half/BFloat16 tensor against its
    # fp32 weights, and F.linear raises "expected mat1 and mat2 to have the
    # same dtype". Pinning the contract here rather than at each call site
    # makes that independent of which flags happen to be on.
    return torch.cat(features, dim=1).float()  # [B, F]


def get_router_feature_dim(cfg: ExperimentConfig, sequence_size: int) -> int:
    """
    Compute the router's expected input dimensionality from config flags.

    Mirrors the concatenation order in extract_router_features():
      n_chunks * d_model   if enable_text_hierarchical  (hierarchical hidden)
      + 4                  if enable_text_stat           (text statistics)
      + external_dim       if use_external_embeddings    (external embeddings)

    When both hierarchical and stat flags are False, returns the full fallback
    dimension n_chunks * d_model + 4 to match the random-feature path in
    extract_router_features().
    """
    if cfg.use_original_sequence:
        return sequence_size
    has_precomputed_emb = getattr(cfg, "use_external_embeddings", False) or bool(
        getattr(cfg, "sentence_embedder_model", "")
    )
    if not cfg.enable_text_hierarchical and not cfg.enable_text_stat and not has_precomputed_emb:
        # Matches the random-feature fallback in extract_router_features().
        return cfg.n_chunks * cfg.d_model + 4
    dim = 0
    if cfg.enable_text_hierarchical:
        dim += cfg.n_chunks * cfg.d_model
    if cfg.enable_text_stat:
        dim += 4
    if getattr(cfg, "use_external_embeddings", False):
        dim += getattr(cfg, "external_embedding_dim", 768)
    if getattr(cfg, "sentence_embedder_model", ""):
        dim += getattr(cfg, "sentence_embedder_dim", 768)
    return dim


def build_router(
    d_input: int,
    arch: str = "attention",
    d_k: int = 128,
    d_hidden: int = 256,
    n_heads: int = 1,
) -> nn.Module | None:
    """
    Factory for all router architectures.

    Args:
        d_input:  Input feature dimensionality. Pass get_router_feature_dim(cfg).
        arch:     Architecture name:
                    'attention' — AttentionRouter (n_heads=1) or
                                  MultiHeadAttentionRouter (n_heads > 1)
                    'linear'   — single linear projection
                    'mlp'      — two-hidden-layer MLP with GELU
                    'auxnet'   — supervised AuxNetRouter (MSE training)
                    'random'   — returns None; training falls back to random scores
        d_k:      Key/query dimension for attention routers.
        d_hidden: Hidden dimension for MLP/auxnet routers.
        n_heads:  Attention heads (attention arch only; >1 enables multi-head).

    Returns an nn.Module or None (for arch='random').
    """
    if arch == "attention":
        if n_heads == 1:
            return AttentionRouter(d_input=d_input, d_k=d_k)
        return MultiHeadAttentionRouter(d_input=d_input, d_k=d_k, n_heads=n_heads)
    if arch == "linear":
        return LinearRouter(d_input=d_input)
    if arch == "mlp":
        return MLPRouter(d_input=d_input, d_hidden=d_hidden)
    if arch == "auxnet":
        return AuxNetRouter(d_input=d_input, d_hidden=d_hidden)
    if arch == "random":
        return None
    raise ValueError(f"Unknown router arch: {arch}")
