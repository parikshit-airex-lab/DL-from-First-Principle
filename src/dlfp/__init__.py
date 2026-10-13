"""dlfp: deep learning models from first principles.

A single training framework driving CNN, RNN, LSTM and Transformer models whose
layers are implemented directly from tensor operations rather than PyTorch's
built-in layer modules. Importing :mod:`dlfp.models` and :mod:`dlfp.data`
populates the model and dataset registries in :mod:`dlfp.registry`. The
optimizers are registered here, on any ``import dlfp``, because they are needed
by every training run and depend on nothing but torch.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .training import optim as _optim  # noqa: F401,E402  (registers sgd/adam/torch_sgd/torch_adam)
