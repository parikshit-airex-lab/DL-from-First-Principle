"""Model zoo. Importing this package registers every model.

Registration happens in the ``@register_model`` decorators, and decorators only
run when their module is imported, so this file is what makes the model
registry non-empty.
"""

from __future__ import annotations

from . import cnn, lstm, rnn, transformer  # noqa: F401  (triggers registration)
from .base import Batch, BaseModel, SequenceModel

__all__ = ["Batch", "BaseModel", "SequenceModel"]
