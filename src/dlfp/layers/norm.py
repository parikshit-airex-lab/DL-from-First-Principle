"""Batch and layer normalisation implemented directly from tensor operations."""

from __future__ import annotations

import torch
from torch import nn


class BatchNorm2d(nn.Module):
    """Batch normalisation over ``(N, H, W)`` for each channel.

    ``y = gamma * (x - mean) / sqrt(var + eps) + beta``

    *Training mode*: ``mean`` and ``var`` are the statistics of the current
    batch over ``(N, H, W)``. Normalisation uses the **biased** variance
    (divide by ``n``), while the running estimate is updated with the
    **unbiased** variance (divide by ``n - 1``), because the running value is an
    estimate of the population variance used later at inference::

        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var  = (1 - momentum) * running_var  + momentum * batch_var_unbiased

    *Eval mode*: ``running_mean``/``running_var`` are used instead, so the output
    for a sample no longer depends on the rest of the batch.

    Args:
        num_features: Number of channels ``C``.
        eps: Added to the variance for numerical stability.
        momentum: Running-stat update factor; ``None`` uses a cumulative average.
        affine: Whether to learn ``gamma`` and ``beta``.
        track_running_stats: Whether to keep running statistics. If ``False``,
            batch statistics are used in both modes.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` per channel.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            Normalised tensor of the same shape.
        """
        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            dims = (0, 2, 3)
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            if self.training and self.track_running_stats:
                n = x.numel() / x.shape[1]
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum
                    unbiased = var * n / max(n - 1, 1)
                    self.running_mean.mul_(1 - factor).add_(factor * mean)
                    self.running_var.mul_(1 - factor).add_(factor * unbiased)
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        if self.affine:
            x_hat = x_hat * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return x_hat

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}"


class LayerNorm(nn.Module):
    """Layer normalisation over the trailing ``normalized_shape`` dimensions.

    ``y = gamma * (x - mean) / sqrt(var + eps) + beta`` where ``mean`` and the
    biased ``var`` are computed per sample over the last ``len(normalized_shape)``
    dimensions. Unlike batch norm, there are no running statistics: train and
    eval behave identically.

    Args:
        normalized_shape: Trailing shape to normalise over (int or tuple).
        eps: Added to the variance for numerical stability.
        elementwise_affine: Whether to learn ``gamma`` and ``beta``.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` over its trailing dimensions.

        Args:
            x: Tensor whose trailing dims equal ``normalized_shape``.

        Returns:
            Normalised tensor of the same shape.
        """
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_hat = x_hat * self.gamma + self.beta
        return x_hat

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
