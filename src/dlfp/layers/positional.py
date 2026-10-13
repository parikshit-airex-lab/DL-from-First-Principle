"""Positional encodings added to token embeddings in the Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn

from .embedding import Embedding


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding (Vaswani et al., 2017)::

        PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

    The table is a non-persistent buffer: it moves with the module across
    devices but is not stored in checkpoints (it is a pure function of shape).

    Args:
        d_model: Model width.
        max_len: Longest supported sequence.
        dropout: Dropout applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe, persistent=False)
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings.

        Args:
            x: ``(B, T, d_model)`` with ``T <= max_len``.

        Returns:
            ``(B, T, d_model)``.
        """
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        return self.dropout(x + self.pe[:t].to(x.dtype))


class LearnedPositionalEmbedding(nn.Module):
    """Trainable absolute position embeddings: ``x + P[0:T]``.

    Args:
        max_len: Longest supported sequence.
        d_model: Model width.
        dropout: Dropout applied after adding the embedding.
    """

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_len = max_len
        self.embedding = Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned position embeddings.

        Args:
            x: ``(B, T, d_model)`` with ``T <= max_len``.

        Returns:
            ``(B, T, d_model)``.
        """
        t = x.size(1)
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        positions = torch.arange(t, device=x.device)
        return self.dropout(x + self.embedding(positions))
