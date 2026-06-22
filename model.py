# model.py
"""
Model and router architecture definitions.

TinyGPT:
  Small causal language model built on PyTorch's TransformerEncoder with a
  causal attention mask (upper-triangular -inf). Shares weights between the
  token embedding and the LM head (weight tying). Used as the student LM
  in all experiments.

Router architectures (all produce a scalar score [B] per sample in the pool):
  AttentionRouter          — single (projection, query) pair; the baseline router
  MultiHeadAttentionRouter — n independent heads, scores averaged across heads

Feature extraction utilities:
  compute_text_statistics()     — 4 cheap surface-level features: sequence fill
                                  ratio, lexical diversity, mean/std token id
  extract_hierarchical_hidden() — transformer hidden states, chunked & pooled
  extract_hierarchical_features() — combines the above two (legacy helper used
                                    by training.py's reference router loop)
"""

from __future__ import annotations

import torch
from torch import nn

from config import Config


class TinyGPT(nn.Module):
    """
    Small causal GPT-style language model (decoder-only transformer).

    Uses nn.TransformerEncoderLayer with an upper-triangular causal mask to
    simulate autoregressive decoding. Weight tying: lm_head.weight == tok_embed.weight,
    halving the effective parameter count and stabilising training.

    forward_to_hidden(x) exposes the transformer hidden states without computing
    logits — used by extract_hierarchical_hidden() for feature extraction without
    a second full forward pass.
    """
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.vocab_size = vocab_size
        self.block = cfg.block
        self.d_model = cfg.d_model

        self.tok_embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.block, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

    def _causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((L, L), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        b, L = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(b, L)
        h = self.tok_embed(x) + self.pos_embed(pos)
        mask = self._causal_mask(L, x.device)
        h = self.tr(h, mask=mask)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_to_hidden(x)
        return self.lm_head(h)


class AttentionRouter(nn.Module):
    """
    Single-head attention-based sample scorer.

    Learns a linear projection W ∈ R^{d_input × d_k} and a query vector
    q ∈ R^{d_k}. For a batch of feature vectors F ∈ R^{B × d_input}:
        scores = (F @ W^T) @ q  ∈ R^B

    Equivalent to a single-head cross-attention where F are the keys and q
    is the query. This is the default/baseline router architecture.
    """
    def __init__(self, d_input: int, d_k: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_input, d_k, bias=False)
        self.q = nn.Parameter(torch.randn(d_k))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        keys = self.proj(feats)  # [B, d_k]
        return keys @ self.q     # [B]


class MultiHeadAttentionRouter(nn.Module):
    """
    n_heads independent (projection, query) pairs whose scores are averaged.
    Each head attends to a different linear subspace of the feature vector,
    letting the router vote on sample quality from multiple perspectives.
    """
    def __init__(self, d_input: int, d_k: int = 128, n_heads: int = 4):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(d_input, d_k, bias=False) for _ in range(n_heads)
        ])
        self.queries = nn.ParameterList([
            nn.Parameter(torch.randn(d_k)) for _ in range(n_heads)
        ])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        scores = torch.stack(
            [(h(feats) @ q) for h, q in zip(self.heads, self.queries)], dim=1
        )  # [B, n_heads]
        return scores.mean(dim=1)  # [B]


def compute_text_statistics(
    X: torch.Tensor,
    pad_token_id: int,
    vocab_size: int,
    block: int,
) -> torch.Tensor:
    """
    Compute 4 cheap surface-level text features per sample.

    Returns a [B, 4] tensor. Column semantics:
      [0] relative_length — non-pad tokens / block  (sequence fill ratio)
      [1] unique_ratio    — unique tokens / sequence length  (lexical diversity)
      [2] avg_token       — mean token id / vocab_size  (normalized)
      [3] std_token       — std of token ids / vocab_size  (normalized)

    All values are in [0, 1]. These four statistics are fast to compute
    (no transformer forward pass) and capture coarse difficulty signals:
    longer, more diverse sequences with unusual token distributions tend to
    be harder for the model to predict.
    """
    mask = X != pad_token_id
    lengths = mask.sum(dim=1).clamp(min=1)
    rel_length = lengths.float() / float(block)

    stats = []
    B = X.size(0)
    for i in range(B):
        tokens = X[i][mask[i]]
        uniq = tokens.unique().numel()
        unique_ratio = uniq / lengths[i].float()
        avg_token = tokens.float().mean() / float(vocab_size)
        std_token = tokens.float().std(unbiased=False) / float(vocab_size)
        stats.append(
            torch.stack(
                [
                    rel_length[i],
                    unique_ratio,
                    avg_token,
                    std_token,
                ]
            )
        )
    return torch.stack(stats, dim=0).to(X.device)


def extract_hierarchical_hidden(
    model: TinyGPT,
    X: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    """
    Extract chunked, mean-pooled hidden states from TinyGPT.

    The sequence of length L is divided into cfg.n_chunks equal segments.
    Each segment's hidden states are mean-pooled to a single d_model vector.
    The n_chunks vectors are concatenated to produce [B, n_chunks * d_model].

    Chunking captures positional structure: early chunks encode document
    start (typically more predictable), later chunks encode content density.
    This is richer than a single mean-pool over the whole sequence.

    Two modes (cfg.hierarchical_representation):
      'full'     — uses full transformer hidden states (one LM forward pass)
      'embedder' — uses only token + positional embeddings, no transformer
                   (~10× faster but loses contextual information)

    Always runs under torch.no_grad() — never affects LM gradients.
    """
    with torch.no_grad():
        repr_mode = getattr(cfg, "hierarchical_representation", "full")
        if repr_mode == "full":
            h = model.forward_to_hidden(X)  # [B, L, D]
        elif repr_mode == "embedder":
            b, L = X.size()
            pos = torch.arange(L, device=X.device).unsqueeze(0).expand(b, L)
            h = model.tok_embed(X) + model.pos_embed(pos)
        else:
            raise ValueError(f"Unknown hierarchical_representation: {repr_mode}")
    B, L, D = h.shape
    assert L % cfg.n_chunks == 0, "Sequence length must be divisible by n_chunks"
    chunk_len = L // cfg.n_chunks
    h_reshaped = h.view(B, cfg.n_chunks, chunk_len, D)
    pooled = h_reshaped.mean(dim=2)          # [B, n_chunks, D]
    return pooled.reshape(B, cfg.n_chunks * D)


def extract_hierarchical_features(
    model: TinyGPT,
    X: torch.Tensor,
    cfg: Config,
    pad_token_id: int,
    vocab_size: int,
) -> torch.Tensor:
    # Original "full" features: hierarchical hidden + stats
    pooled = extract_hierarchical_hidden(model, X, cfg)  # [B, n_chunks*D]
    stats = compute_text_statistics(
        X,
        pad_token_id=pad_token_id,
        vocab_size=vocab_size,
        block=cfg.block,
    )
    return torch.cat([pooled, stats], dim=1)



