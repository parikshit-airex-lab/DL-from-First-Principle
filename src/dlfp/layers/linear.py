"""Fully connected (affine) layer implemented directly from tensor operations."""

from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """Affine map ``y = x W^T + b``.

    ``W`` has shape ``(out_features, in_features)`` and ``b`` shape
    ``(out_features,)``. The input may have any number of leading dimensions:
    ``(*, in_features) -> (*, out_features)``.

    Initialisation matches PyTorch's default: Kaiming-uniform with
    ``a = sqrt(5)``, which works out to ``W ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))``,
    and ``b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))``.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: Whether to learn an additive bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform (a=sqrt(5)) weights and fan-in-uniform bias."""
        fan_in = self.in_features
        # kaiming_uniform bound = gain * sqrt(3 / fan_in), gain = sqrt(2 / (1 + a^2)).
        a = math.sqrt(5)
        gain = math.sqrt(2.0 / (1 + a * a))
        w_bound = gain * math.sqrt(3.0 / fan_in) if fan_in > 0 else 0.0
        with torch.no_grad():
            self.weight.uniform_(-w_bound, w_bound)
            if self.bias is not None:
                b_bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                self.bias.uniform_(-b_bound, b_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``x @ W^T + b``.

        Args:
            x: Tensor of shape ``(*, in_features)``.

        Returns:
            Tensor of shape ``(*, out_features)``.
        """
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
