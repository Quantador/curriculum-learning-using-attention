import torch
import torch.nn.functional as F
from .router import gumbel_noise_like, softmax_with_temp

def topk_straight_through(probs, k: int):
    """
    Hard top-k in forward, soft gradient in backward (ST).
    Returns: hard_mask [M], soft_probs [M]
    """
    with torch.no_grad():
        topk_idx = torch.topk(probs, k=k, dim=-1).indices
        hard = torch.zeros_like(probs)
        hard[topk_idx] = 1.0
    # ST trick: forward uses hard, backward flows as if probs
    st = (hard - probs).detach() + probs
    return hard, st

def gumbel_topk_st(probs, k: int, tau: float):
    """
    Add Gumbel noise to logits (via log probs), then do ST top-k.
    """
    logits = probs.clamp_min(1e-12).log()  # treat as logits
    noisy = logits + gumbel_noise_like(logits)
    soft = F.softmax(noisy / tau, dim=-1)
    hard, st = topk_straight_through(soft, k)
    return hard, st, soft

def soft_dense_weights(probs):
    """
    Dense relaxation: use probs for all items (compute-heavy).
    """
    return probs  # [M], sums to 1
