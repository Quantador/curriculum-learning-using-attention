# model.py
from __future__ import annotations

import torch
from torch import nn

from config import Config


class TinyGPT(nn.Module):
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
    def __init__(self, d_input: int, d_k: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_input, d_k, bias=False)
        self.q = nn.Parameter(torch.randn(d_k))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        keys = self.proj(feats)  # [B, d_k]
        return keys @ self.q     # [B]


def compute_text_statistics(
    X: torch.Tensor,
    pad_token_id: int,
    vocab_size: int,
    block: int,
) -> torch.Tensor:
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
    with torch.no_grad():
        h = model.forward_to_hidden(X)  # [B, L, D]
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



