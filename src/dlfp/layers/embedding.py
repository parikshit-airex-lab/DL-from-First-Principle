"""Token embedding lookup table implemented directly from tensor indexing."""

from __future__ import annotations

import torch
from torch import nn


class Embedding(nn.Module):
    """Lookup table mapping integer ids to dense vectors: ``y = W[ids]``.

    ``W`` has shape ``(num_embeddings, embedding_dim)`` and is initialised
    ``N(0, 1)`` (PyTorch's default). Indexing is differentiable: the gradient
    w.r.t. ``W`` is a scatter-add of the upstream gradient into the used rows.

    With ``padding_idx`` set, that row is zeroed at init and a gradient hook
    zeroes its gradient on every backward pass, so the pad vector stays exactly
    zero and padding contributes nothing downstream.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Vector size.
        padding_idx: Optional id whose vector is fixed at zero.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        if padding_idx is not None and padding_idx < 0:
            padding_idx += num_embeddings
        if padding_idx is not None and not 0 <= padding_idx < num_embeddings:
            raise ValueError(f"padding_idx {padding_idx} out of range for {num_embeddings} embeddings")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()
        if padding_idx is not None:
            self.weight.register_hook(self._zero_pad_grad)

    def reset_parameters(self) -> None:
        """``N(0, 1)`` init, with the padding row zeroed."""
        with torch.no_grad():
            self.weight.normal_(0.0, 1.0)
            if self.padding_idx is not None:
                self.weight[self.padding_idx].zero_()

    def _zero_pad_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Gradient hook: return a copy of ``grad`` with the pad row zeroed."""
        grad = grad.clone()
        grad[self.padding_idx] = 0
        return grad

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up ``input_ids``.

        Args:
            input_ids: Integer tensor of any shape ``(*)``.

        Returns:
            Tensor of shape ``(*, embedding_dim)``.
        """
        return self.weight[input_ids]

    def extra_repr(self) -> str:
        """Summary shown in ``repr``."""
        return f"{self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx}"
