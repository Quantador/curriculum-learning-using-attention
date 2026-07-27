import torch 
import math 

# =============================================================================
# Entropy formulations
# =============================================================================

def compute_shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Standard Shannon entropy: H = -sum(p * log(p))

    Args:
        probs: Probability distribution [M]

    Returns:
        Scalar entropy value
    """
    return (probs * probs.clamp_min(1e-12).log()).sum()


def compute_renyi_entropy(probs: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Rényi entropy: H_α = (1/(1-α)) * log(sum(p^α))

    Special cases:
    - α → 1: Shannon entropy
    - α = 0: Max entropy (log of support size)
    - α = 2: Collision entropy (related to collision probability)
    - α → ∞: Min-entropy

    Args:
        probs: Probability distribution [M]
        alpha: Rényi parameter (> 0, != 1)

    Returns:
        Scalar Rényi entropy value (negated for use as loss)
    """
    if abs(alpha - 1.0) < 1e-6:
        # Limit case: Shannon entropy
        return compute_shannon_entropy(probs)

    # Rényi entropy: (1/(1-α)) * log(sum(p^α))
    p_alpha = probs.clamp_min(1e-12).pow(alpha)
    renyi = (1.0 / (1.0 - alpha)) * p_alpha.sum().clamp_min(1e-12).log()

    # Return negative for consistency (minimizing = maximizing entropy)
    return -renyi


def compute_tsallis_entropy(probs: torch.Tensor, q: float = 2.0) -> torch.Tensor:
    """
    Tsallis entropy: S_q = (1/(q-1)) * (1 - sum(p^q))

    Non-extensive entropy that generalizes Boltzmann-Gibbs.
    - q → 1: Shannon entropy
    - q < 1: Favors rare events
    - q > 1: Favors common events

    Args:
        probs: Probability distribution [M]
        q: Tsallis parameter (> 0)

    Returns:
        Scalar Tsallis entropy value (negated for use as loss)
    """
    if abs(q - 1.0) < 1e-6:
        # Limit case: Shannon entropy
        return compute_shannon_entropy(probs)

    p_q = probs.clamp_min(1e-12).pow(q)
    tsallis = (1.0 / (q - 1.0)) * (1.0 - p_q.sum())

    # Return negative for consistency
    return -tsallis

def compute_kl_from_uniform(probs: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from uniform distribution: KL(p || u)

    Measures how far the distribution is from uniform (maximum entropy).
    KL(p || u) = sum(p * log(p)) - log(1/n) = -H(p) + log(n)

    Args:
        probs: Probability distribution [M]

    Returns:
        KL divergence (0 = uniform, higher = more peaked)
    """
    n = probs.shape[0]
    log_n = math.log(n)
    shannon = -compute_shannon_entropy(probs)  # H(p)
    return log_n - shannon  # KL(p || u)


def compute_entropy(
    probs: torch.Tensor,
    entropy_type: str = "shannon",
    alpha: float = 2.0,
    q: float = 2.0,
) -> torch.Tensor:
    """
    Compute entropy using specified formulation.

    Args:
        probs: Probability distribution [M]
        entropy_type: 'shannon', 'renyi', 'tsallis', or 'kl_uniform'
        alpha: Parameter for Rényi entropy
        q: Parameter for Tsallis entropy

    Returns:
        Entropy value (to be used in loss with positive lambda_ent)
    """
    if entropy_type == "shannon":
        return compute_shannon_entropy(probs)

    elif entropy_type == "renyi":
        return compute_renyi_entropy(probs, alpha)

    elif entropy_type == "tsallis":
        return compute_tsallis_entropy(probs, q)

    elif entropy_type == "kl_uniform":
        # Return negative KL so that minimizing increases uniformity
        return -compute_kl_from_uniform(probs)

    else:
        return compute_shannon_entropy(probs)