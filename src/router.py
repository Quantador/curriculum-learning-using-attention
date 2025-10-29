import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterMLP(nn.Module):
    """
    Tiny router over per-sample features φ(x) -> score s_j.
    """
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # scalar score per sample
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [M, in_dim]
        returns logits: [M]
        """
        return self.net(feats).squeeze(-1)

def gumbel_noise_like(t):
    # Gumbel(0,1) noise
    u = torch.rand_like(t).clamp_(1e-6, 1 - 1e-6)
    return -torch.log(-torch.log(u))

def softmax_with_temp(logits, tau):
    return F.softmax(logits / tau, dim=-1)

def entropy_from_probs(p, eps: float = 1e-12):
    return -(p * (p.add_(eps).log())).sum()
