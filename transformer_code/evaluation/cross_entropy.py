import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets, pad_idx=0):
    """Token-averaged cross-entropy. logits: (B,T,V), targets: (B,T)."""
    b, t, v = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * t, v),
        targets.reshape(b * t),
        ignore_index=pad_idx,
        reduction="sum",
    )
    n = (targets != pad_idx).sum().clamp_min(1)
    return (loss / n).item()
