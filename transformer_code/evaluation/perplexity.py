import math

from .cross_entropy import cross_entropy_loss


def perplexity(logits, targets, pad_idx=0):
    """PPL = exp(average token negative log-likelihood)."""
    return math.exp(cross_entropy_loss(logits, targets, pad_idx))
