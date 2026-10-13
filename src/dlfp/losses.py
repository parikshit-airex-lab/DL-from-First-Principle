"""Loss functions, including mask-aware cross-entropy for padded sequences."""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Target value that marks positions excluded from the loss (e.g. padded LM targets).
IGNORE_INDEX = -100


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over positions where ``mask`` is 1.

    ``(values * mask).sum() / mask.sum()``, with the denominator clamped to at
    least 1 so an all-pad batch yields 0 instead of NaN.

    Args:
        values: Tensor of per-element values.
        mask: Same-shape 0/1 (float or bool) mask.

    Returns:
        Scalar tensor.
    """
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1)


def sequence_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-level cross-entropy averaged over real (non-pad) positions only.

    Logits ``(B, T, V)`` and targets ``(B, T)`` are flattened to ``(B*T, V)`` and
    ``(B*T,)``; the per-token loss is computed with ``reduction="none"`` and then
    averaged with :func:`masked_mean`.

    Why not a plain mean: in a padded batch, pad positions are not real
    predictions. Averaging over all ``B*T`` positions divides by too many
    tokens, so the reported loss is understated by the pad fraction (and changes
    when the same data is padded differently), and any loss computed *at* pad
    positions would push gradient into parameters for predictions that do not
    exist. Masking zeroes both the contribution and the gradient of pad
    positions and divides by the true token count.

    Args:
        logits: ``(B, T, V)`` unnormalised scores.
        targets: ``(B, T)`` integer targets; ``ignore_index`` marks positions to skip.
        mask: Optional ``(B, T)`` 1 = real, 0 = pad. Combined with ``ignore_index``.
        ignore_index: Target value excluded from the loss.

    Returns:
        Scalar mean loss over real tokens.
    """
    v = logits.size(-1)
    flat_logits = logits.reshape(-1, v)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if mask is not None:
        valid = valid & mask.reshape(-1).to(torch.bool)
    per_token = F.cross_entropy(flat_logits, flat_targets.masked_fill(~valid, 0), reduction="none")
    return masked_mean(per_token, valid)


def classification_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy for ``(B, C)`` logits and ``(B,)`` class targets.

    Args:
        logits: ``(B, C)`` unnormalised scores.
        targets: ``(B,)`` class indices.

    Returns:
        Scalar mean loss.
    """
    return F.cross_entropy(logits, targets)
