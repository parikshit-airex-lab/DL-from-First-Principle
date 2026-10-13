"""Elman RNN model: :class:`SequenceModel` with :class:`RNNCell`."""

from __future__ import annotations

from typing import Any

from dlfp.layers import RNNCell
from dlfp.models.base import SequenceModel
from dlfp.registry import register_model


@register_model("rnn")
class RNNModel(SequenceModel):
    """Recurrent model with tanh RNN cells. See :class:`SequenceModel` for params."""

    def __init__(self, **params: Any) -> None:
        super().__init__(cell_factory=RNNCell, **params)
