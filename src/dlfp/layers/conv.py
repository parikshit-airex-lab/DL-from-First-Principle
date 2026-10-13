"""2-D convolution and pooling implemented with ``F.unfold`` (im2col) and matmul."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise an int or 2-tuple to a 2-tuple."""
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected an int or a 2-tuple, got {value!r}")
    return (int(value[0]), int(value[1]))


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Spatial output size ``floor((size + 2p - k) / s) + 1``."""
    return (size + 2 * padding - kernel) // stride + 1


class Conv2d(nn.Module):
    """2-D cross-correlation via im2col.

    For input ``(N, C_in, H, W)``, ``F.unfold`` extracts every receptive field
    into a column, giving ``(N, C_in * kH * kW, L)`` with ``L = H_out * W_out``.
    The weight ``(C_out, C_in, kH, kW)`` is flattened to ``(C_out, C_in*kH*kW)``
    so the convolution becomes one batched matmul ``W_flat @ cols``, reshaped to
    ``(N, C_out, H_out, W_out)``, where::

        H_out = floor((H + 2 * pad_h - kH) / stride_h) + 1
        W_out = floor((W + 2 * pad_w - kW) / stride_w) + 1

    Initialisation matches PyTorch (Kaiming-uniform a=sqrt(5), fan-in bias).

    Args:
        in_channels: Input channels.
        out_channels: Output channels (number of filters).
        kernel_size: Kernel size, int or ``(kH, kW)``.
        stride: Stride, int or ``(sH, sW)``.
        padding: Zero padding, int or ``(pH, pW)``.
        bias: Whether to learn a per-channel bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming-uniform (a=sqrt(5)) weights, fan-in-uniform bias."""
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = 1.0 / math.sqrt(fan_in)  # == sqrt(1/3) * sqrt(3/fan_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve ``x``.

        Args:
            x: Input of shape ``(N, C_in, H, W)``.

        Returns:
            Output of shape ``(N, C_out, H_out, W_out)``.
        """
        n, _, h, w = x.shape
        h_out = _out_size(h, self.kernel_size[0], self.stride[0], self.padding[0])
        w_out = _out_size(w, self.kernel_size[1], self.stride[1], self.padding[1])
        cols = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)  # (N, C*kh*kw, L)
        out = self.weight.view(self.out_channels, -1) @ cols  # (N, C_out, L)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1)
        return out.view(n, self.out_channels, h_out, w_out)

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}")


class MaxPool2d(nn.Module):
    """2-D max pooling: the maximum over each ``kH x kW`` window.

    Implemented by unfolding each channel independently into window columns and
    taking ``max`` over the window dimension. Padding is filled with ``-inf`` (as
    in PyTorch) so padded cells never win the max. Output size follows the same
    formula as :class:`Conv2d`.

    Args:
        kernel_size: Window size, int or 2-tuple.
        stride: Stride (defaults to ``kernel_size``).
        padding: Implicit ``-inf`` padding, int or 2-tuple.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Max-pool ``x``.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            Output of shape ``(N, C, H_out, W_out)``.
        """
        n, c, h, w = x.shape
        ph, pw = self.padding
        if ph or pw:
            x = F.pad(x, (pw, pw, ph, ph), value=float("-inf"))
        h_out = _out_size(h, self.kernel_size[0], self.stride[0], ph)
        w_out = _out_size(w, self.kernel_size[1], self.stride[1], pw)
        cols = F.unfold(x.reshape(n * c, 1, x.shape[2], x.shape[3]), self.kernel_size, stride=self.stride)
        return cols.max(dim=1).values.view(n, c, h_out, w_out)

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class GlobalAvgPool2d(nn.Module):
    """Average over all spatial positions: ``(N, C, H, W) -> (N, C)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean over the H and W dimensions.

        Args:
            x: Input of shape ``(N, C, H, W)``.

        Returns:
            Tensor of shape ``(N, C)``.
        """
        return x.mean(dim=(2, 3))
