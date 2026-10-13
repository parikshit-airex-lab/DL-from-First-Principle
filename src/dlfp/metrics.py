"""Evaluation metrics and a weighted running-mean tracker."""

from __future__ import annotations

import math

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Fraction of argmax predictions equal to ``targets``.

    Args:
        logits: ``(..., C)`` scores.
        targets: ``(...)`` class indices.
        mask: Optional ``(...)`` 0/1 mask restricting which positions count.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    correct = (logits.argmax(dim=-1) == targets).float()
    if mask is None:
        return correct.mean().item()
    mask = mask.float()
    return ((correct * mask).sum() / mask.sum().clamp(min=1)).item()


def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean of per-class F1 scores.

    Classes with no predictions and no targets are excluded from the mean.

    Args:
        logits: ``(N, C)`` scores.
        targets: ``(N,)`` class indices.
        num_classes: Number of classes ``C``.

    Returns:
        Macro F1 in ``[0, 1]``.
    """
    preds = logits.argmax(dim=-1)
    scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def perplexity(loss: float) -> float:
    """``exp(loss)`` for a mean token-level cross-entropy, capped to avoid overflow.

    Args:
        loss: Mean cross-entropy in nats.

    Returns:
        Perplexity.
    """
    return math.exp(min(loss, 50.0))


class MetricTracker:
    """Accumulates weighted running means of scalar metrics.

    ``update({"loss": 0.5}, n=32)`` weights the value by the number of samples
    (or tokens) it was computed on, so the epoch result is a true average even
    when the last batch is smaller.
    """

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, float] = {}

    def update(self, metrics: dict[str, float], n: int | float = 1) -> None:
        """Add one batch of metrics.

        Args:
            metrics: Metric name -> value for the batch.
            n: Weight of this batch.
        """
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v) * n
            self._counts[k] = self._counts.get(k, 0.0) + n

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def result(self) -> dict[str, float]:
        """Return the weighted mean of every metric seen since the last reset."""
        return {k: self._sums[k] / max(self._counts[k], 1e-12) for k in self._sums}

    def __str__(self) -> str:
        """Compact ``name=value`` summary."""
        return " ".join(f"{k}={v:.4f}" for k, v in self.result().items())
