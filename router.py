from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from config import ExperimentConfig

from model import (
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
    Supervised alternative to the RL router.

    Regresses directly on the observed per-sample loss-improvement signal
    using MSE. At inference time its predicted improvement is used as a
    selection score (top-k), exactly like the attention router — making it
    a direct, fair comparison: same features, same selection logic, different
    training objective (supervised regression vs. policy gradient).
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
    external_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build the feature vector fed to the router.

    Concatenates any enabled feature groups:
      - Hierarchical hidden states (cfg.enable_text_hierarchical)
      - Text statistics: length, uniqueness, avg/std token id (cfg.enable_text_stat)
      - Pre-computed external embeddings (cfg.use_external_embeddings, external_emb)
    """
    features = []

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

    if getattr(cfg, "use_external_embeddings", False) and external_emb is not None:
        features.append(external_emb)  # [B, external_embedding_dim]

    if not features:
        print("Warning: No features enabled for router; returning random features.")
        full_dim = cfg.n_chunks * cfg.d_model + 4
        return torch.randn(X.size(0), full_dim, device=X.device)

    return torch.cat(features, dim=1)  # [B, F]


def get_router_feature_dim(cfg: ExperimentConfig) -> int:
    """Compute feature dimensionality based on enabled feature flags."""
    full_dim = cfg.n_chunks * cfg.d_model + 4
    if not cfg.enable_text_hierarchical and not cfg.enable_text_stat:
        return full_dim
    dim = 0
    if cfg.enable_text_hierarchical:
        dim += cfg.n_chunks * cfg.d_model
    if cfg.enable_text_stat:
        dim += 4
    if getattr(cfg, "use_external_embeddings", False):
        dim += getattr(cfg, "external_embedding_dim", 768)
    return dim


def build_router(
    d_input: int,
    arch: str = "attention",
    d_k: int = 128,
    d_hidden: int = 256,
    n_heads: int = 1,
) -> nn.Module | None:
    """
    Factory for router architectures.

    arch options:
      "attention"  — single-head AttentionRouter (n_heads=1) or
                     MultiHeadAttentionRouter (n_heads > 1)
      "linear"     — single linear projection to scalar
      "mlp"        — two-hidden-layer MLP
      "auxnet"     — supervised aux-net that regresses on loss improvement
      "random"     — returns None (scores will be random in training code)
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
