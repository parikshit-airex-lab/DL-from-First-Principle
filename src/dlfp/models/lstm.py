"""LSTM model: :class:`SequenceModel` with :class:`LSTMCell`."""

from __future__ import annotations

from typing import Any

from dlfp.layers import LSTMCell
from dlfp.models.base import SequenceModel
from dlfp.registry import register_model


@register_model("lstm")
class LSTMModel(SequenceModel):
    """Recurrent model with LSTM cells. See :class:`SequenceModel` for params."""

    def __init__(self, **params: Any) -> None:
        super().__init__(cell_factory=LSTMCell, **params)
