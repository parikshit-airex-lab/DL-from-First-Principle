"""Learning-rate schedules that set ``group["lr"]`` directly.

Each scheduler records the base learning rates at construction and sets the
lr for its current step on every ``step()`` call. ``interval`` says whether it
is meant to be stepped once per epoch or once per optimizer step; the
:class:`~dlfp.training.callbacks.LRSchedulerCallback` honours it.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch


class _Scheduler(ABC):
    """Base class: tracks ``last_step`` and applies ``self.lr_at(step, base_lr)``."""

    interval = "epoch"

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group.setdefault("initial_lr", group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._apply()

    @abstractmethod
    def lr_at(self, step: int, base_lr: float) -> float:
        """Learning rate for ``step`` given the group's base rate."""

    def _apply(self) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.lr_at(self.last_step, base)

    def step(self) -> None:
        """Advance one step (or epoch) and update every group's lr."""
        self.last_step += 1
        self._apply()

    def get_last_lr(self) -> list[float]:
        """Current lr of each parameter group."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """Serializable scheduler state."""
        return {"last_step": self.last_step, "base_lrs": list(self.base_lrs)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state and re-apply the corresponding learning rates.

        Args:
            state: Output of :meth:`state_dict`.
        """
        self.last_step = int(state["last_step"])
        self.base_lrs = list(state["base_lrs"])
        self._apply()


class ConstantLR(_Scheduler):
    """Keeps every group at its base learning rate."""

    def lr_at(self, step: int, base_lr: float) -> float:
        """Always ``base_lr``."""
        return base_lr


class StepLR(_Scheduler):
    """``lr = base_lr * gamma ** (epoch // step_size)``, stepped per epoch.

    Args:
        optimizer: Wrapped optimizer.
        step_size: Epochs between decays.
        gamma: Multiplicative decay.
    """

    interval = "epoch"

    def __init__(self, optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.1) -> None:
        self.step_size = max(1, int(step_size))
        self.gamma = gamma
        super().__init__(optimizer)

    def lr_at(self, step: int, base_lr: float) -> float:
        """Step-decayed rate."""
        return base_lr * self.gamma ** (step // self.step_size)


class CosineWithWarmup(_Scheduler):
    """Linear warmup then cosine decay to ``min_lr``, stepped per optimizer step.

    For ``s < warmup_steps``: ``lr = base_lr * (s + 1) / warmup_steps``.
    Afterwards, with ``p = (s - warmup) / (total - warmup)`` clamped to ``[0, 1]``:
    ``lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * p))``.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Linear warmup length.
        total_steps: Total optimizer steps in training.
        min_lr: Final learning rate.
    """

    interval = "step"

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
                 min_lr: float = 0.0) -> None:
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr = min_lr
        super().__init__(optimizer)

    def lr_at(self, step: int, base_lr: float) -> float:
        """Warmup-then-cosine rate."""
        if step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


Scheduler = _Scheduler


def build_scheduler(name: str | None, optimizer: torch.optim.Optimizer, cfg: Any,
                    total_steps: int) -> _Scheduler | None:
    """Build the scheduler named by ``cfg.train.scheduler``.

    * ``None`` -> no scheduler.
    * ``"constant"`` -> :class:`ConstantLR`.
    * ``"step"`` -> :class:`StepLR` decaying by 10x every ``max(1, epochs // 3)`` epochs.
    * ``"cosine"`` -> :class:`CosineWithWarmup` over ``total_steps`` with
      ``cfg.train.warmup_steps`` warmup, decaying to ``lr / 100``.

    Args:
        name: Scheduler name or ``None``.
        optimizer: Optimizer to schedule.
        cfg: Full :class:`~dlfp.config.Config`.
        total_steps: Total optimizer steps (``epochs * steps_per_epoch``).

    Returns:
        The scheduler, or ``None``.
    """
    if name is None:
        return None
    key = name.lower()
    if key == "constant":
        return ConstantLR(optimizer)
    if key == "step":
        return StepLR(optimizer, step_size=max(1, cfg.train.epochs // 3), gamma=0.1)
    if key == "cosine":
        return CosineWithWarmup(optimizer, cfg.train.warmup_steps, total_steps, min_lr=cfg.train.lr / 100)
    raise KeyError(f"Unknown scheduler '{name}'. Available schedulers: [constant, cosine, step]")
