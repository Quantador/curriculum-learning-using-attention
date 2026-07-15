# model.py
"""
Model and router architecture definitions.

TinyGPT:
  Small causal language model built on PyTorch's TransformerEncoder with a
  causal attention mask (upper-triangular -inf). Shares weights between the
  token embedding and the LM head (weight tying). The default student LM.

HFCausalLM:
  Wraps a HuggingFace causal LM architecture (e.g. Qwen3-1.7B) behind the
  same interface as TinyGPT, randomly initialized (not fine-tuned from a
  checkpoint). Selected via Config.model_type == 'hf_pretrained'.

build_model(vocab_size, cfg):
  Factory that returns TinyGPT or HFCausalLM based on cfg.model_type.

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
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM

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
    supports_embedder_mode = True

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


class HFCausalLM(nn.Module):
    """
    Wraps a HuggingFace causal LM architecture (e.g. Qwen3-1.7B) behind the
    same interface TinyGPT exposes (forward, forward_to_hidden, d_model,
    block, vocab_size), so it can be dropped into the existing curriculum
    loop wherever a TinyGPT is expected.

    Weights are randomly initialized from the architecture's config
    (AutoModelForCausalLM.from_config) — this trains the architecture from
    scratch, it does not load pretrained checkpoint weights.

    'embedder' hierarchical_representation mode needs a learned absolute
    positional embedding table. Architectures with one (e.g. GPT-2/GPT2-XL,
    exposed as transformer.wpe) support it just like TinyGPT. RoPE-based
    architectures (e.g. Qwen3) compute position inline in attention and have
    no such table, so supports_embedder_mode is set dynamically per instance
    based on whether one was actually found — extract_hierarchical_hidden()
    checks this flag and raises a clear error rather than an AttributeError.
    """
    # Known (backbone_attr, pos_embed_attr) paths, in priority order.
    _POS_EMBED_PATHS = [
        ("transformer", "wpe"),  # GPT-2 family (gpt2, gpt2-xl, distilgpt2, ...)
    ]

    def __init__(self, model_name: str, vocab_size: int, block: int):
        super().__init__()
        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.vocab_size = vocab_size
        if hasattr(hf_config, "max_position_embeddings"):
            hf_config.max_position_embeddings = max(
                block, getattr(hf_config, "max_position_embeddings", block)
            )
        self.hf = AutoModelForCausalLM.from_config(hf_config)
        self.vocab_size = vocab_size
        self.block = block
        self.d_model = hf_config.hidden_size

        self._pos_embed_module = self._find_positional_embedding()
        self.supports_embedder_mode = self._pos_embed_module is not None

    def _find_positional_embedding(self) -> nn.Embedding | None:
        for backbone_attr, pos_attr in self._POS_EMBED_PATHS:
            backbone = getattr(self.hf, backbone_attr, None)
            if backbone is not None and hasattr(backbone, pos_attr):
                return getattr(backbone, pos_attr)
        return None

    @property
    def tok_embed(self) -> nn.Embedding:
        return self.hf.get_input_embeddings()

    @property
    def pos_embed(self) -> nn.Embedding:
        if self._pos_embed_module is None:
            raise AttributeError(
                f"{type(self.hf).__name__} has no learned positional embedding table."
            )
        return self._pos_embed_module

    def _expanded_position_ids(self, x: torch.Tensor) -> torch.Tensor | None:
        # HF's default GPT-2 forward looks up wpe with a batch dim of 1
        # (position_ids = cache_position.unsqueeze(0)) and broadcasts the
        # result across the batch when adding it to the token embeddings.
        # That's fine for plain training, but it means autograd sums the
        # position-embedding gradient across every sample before any
        # per-sample hook (e.g. GhostSuite's ghost gradient dot product)
        # sees it, which then chokes on a batch-size-1 activation. Passing
        # an explicitly batch-expanded position_ids leaves the forward
        # output unchanged but makes the embedding lookup genuinely
        # per-sample, like TinyGPT's pos_embed(pos) call already does.
        if self._pos_embed_module is None:
            return None
        b, L = x.shape
        return torch.arange(L, device=x.device).unsqueeze(0).expand(b, L)

    def forward_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        # output_hidden_states works across HF causal LM architectures
        # regardless of the backbone attribute name (Qwen3 uses `.model`,
        # GPT-2 uses `.transformer`, etc.) — avoids hardcoding either.
        position_ids = self._expanded_position_ids(x)
        return self.hf(
            input_ids=x, position_ids=position_ids, output_hidden_states=True
        ).hidden_states[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_ids = self._expanded_position_ids(x)
        return self.hf(input_ids=x, position_ids=position_ids).logits


def build_model(vocab_size: int, cfg: Config) -> nn.Module:
    """
    Construct the student LM named by cfg.model_type:
      'tiny_gpt'      — TinyGPT(vocab_size, cfg)
      'hf_pretrained' — HFCausalLM(cfg.hf_model_name, vocab_size, cfg.block)
    """
    model_type = getattr(cfg, "model_type", "tiny_gpt")
    if model_type == "tiny_gpt":
        return TinyGPT(vocab_size=vocab_size, cfg=cfg)
    elif model_type == "hf_pretrained":
        return HFCausalLM(model_name=cfg.hf_model_name, vocab_size=vocab_size, block=cfg.block)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Expected 'tiny_gpt' or 'hf_pretrained'.")


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

    Accepts model wrapped in DistributedDataParallel: forward_to_hidden/
    tok_embed/pos_embed are accessed via model.module in that case, since
    DDP only proxies its own registered submodules (the wrapped module as
    a whole) through __getattr__, not the wrapped module's own attributes.
    """
    m = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        repr_mode = getattr(cfg, "hierarchical_representation", "full")
        if repr_mode == "full":
            h = m.forward_to_hidden(X)  # [B, L, D]
        elif repr_mode == "embedder":
            if not getattr(m, "supports_embedder_mode", True):
                raise ValueError(
                    f"{type(m).__name__} does not support "
                    "hierarchical_representation='embedder' (no separate "
                    "learned positional embedding table). Use 'full' instead."
                )
            b, L = X.size()
            pos = torch.arange(L, device=X.device).unsqueeze(0).expand(b, L)
            h = m.tok_embed(X) + m.pos_embed(pos)
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
    print(f"{pooled.shape=}")
    print(f"{stats.shape=}")
    return torch.cat([pooled, stats], dim=1)



