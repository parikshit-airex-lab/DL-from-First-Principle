"""Scaled dot-product and multi-head attention written out in tensor operations.

Mask convention used throughout this module: a boolean (or 0/1) mask where
``True``/``1`` means *may attend* and ``False``/``0`` means *blocked*. (This is
the opposite of ``nn.MultiheadAttention``'s ``key_padding_mask``, where
``True`` marks padding; it matches the data pipeline's ``mask``, 1 = real.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .linear import Linear


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Lower-triangular boolean mask: position ``i`` may attend to ``j <= i``.

    Args:
        seq_len: Sequence length ``T``.
        device: Device for the mask.

    Returns:
        Bool tensor of shape ``(1, 1, T, T)``.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)).view(1, 1, seq_len, seq_len)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``softmax(q k^T / sqrt(d_k) + mask) v``.

    Blocked positions get a score of ``-inf`` so their softmax weight is exactly
    zero. A query row whose keys are *all* blocked would otherwise produce NaN
    (``softmax`` of all ``-inf``); such rows get all-zero weights instead.

    Args:
        q: Queries ``(..., T_q, d_k)``.
        k: Keys ``(..., T_k, d_k)``.
        v: Values ``(..., T_k, d_v)``.
        mask: Optional mask broadcastable to ``(..., T_q, T_k)``; True = attend.
        dropout_p: Dropout probability on the attention weights.
        training: Whether dropout is active.

    Returns:
        ``(output (..., T_q, d_v), attn_weights (..., T_q, T_k))``.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    all_blocked = None
    if mask is not None:
        allowed = mask.to(torch.bool)
        scores = scores.masked_fill(~allowed, float("-inf"))
        all_blocked = (~allowed).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_blocked, 0.0)
    attn = F.softmax(scores, dim=-1)
    if all_blocked is not None:
        attn = attn.masked_fill(all_blocked, 0.0)
    if dropout_p > 0.0 and training:
        attn = F.dropout(attn, p=dropout_p, training=True)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q, K, V and output projections.

    ``head_i = Attention(x W_q^i, x W_k^i, x W_v^i)``,
    ``MHA(x) = concat(head_1..head_h) W_o``. Each head has ``d_head = d_model / n_heads``.

    Args:
        d_model: Model width.
        n_heads: Number of heads; must divide ``d_model``.
        dropout: Dropout on attention weights.
        bias: Whether projections have a bias.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout
        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model) -> (B, n_heads, T, d_head)``."""
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend from ``query`` to ``key``/``value`` (self-attention by default).

        Args:
            query: ``(B, T_q, d_model)``.
            key: ``(B, T_k, d_model)``; defaults to ``query``.
            value: ``(B, T_k, d_model)``; defaults to ``key``.
            attn_mask: Optional mask broadcastable to ``(B, n_heads, T_q, T_k)``,
                e.g. :func:`causal_mask`; True = attend.
            key_padding_mask: Optional ``(B, T_k)``; 1/True = real key, 0 = pad.

        Returns:
            ``(output (B, T_q, d_model), attn_weights (B, n_heads, T_q, T_k))``.
        """
        key = query if key is None else key
        value = key if value is None else value
        b, t_q, _ = query.shape
        t_k = key.size(1)

        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        mask = None
        shape = (b, self.n_heads, t_q, t_k)
        if attn_mask is not None:
            mask = attn_mask.to(torch.bool).expand(shape)
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(torch.bool).view(b, 1, 1, t_k).expand(shape)
            mask = kpm if mask is None else (mask & kpm)

        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout, self.training)
        out = out.transpose(1, 2).reshape(b, t_q, self.d_model)
        return self.out_proj(out), weights
