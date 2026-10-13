"""Training: optimizers, schedulers, checkpointing, callbacks and the trainer.

Importing this package registers the optimizers.
"""

from __future__ import annotations

from .callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LRSchedulerCallback,
    ModelCheckpoint,
    build_callbacks,
)
from .checkpoint import load_checkpoint, restore_state, save_checkpoint
from .optim import SGD, Adam, build_optimizer_from_config
from .schedulers import ConstantLR, CosineWithWarmup, StepLR, build_scheduler
from .trainer import Trainer

__all__ = [
    "Adam",
    "Callback",
    "CallbackList",
    "ConstantLR",
    "CosineWithWarmup",
    "EarlyStopping",
    "LRSchedulerCallback",
    "ModelCheckpoint",
    "SGD",
    "StepLR",
    "Trainer",
    "build_callbacks",
    "build_optimizer_from_config",
    "build_scheduler",
    "load_checkpoint",
    "restore_state",
    "save_checkpoint",
]
