"""Training callbacks: LR scheduling, early stopping and checkpointing.

A callback receives the :class:`~dlfp.training.trainer.Trainer` at each hook
and may read its public state (``current_epoch``, ``global_step``,
``last_metrics``, ``model``, ``optimizer``, ``cfg``, ``logger``) or set
``trainer.should_stop``. Callbacks with state implement ``state_dict`` /
``load_state_dict`` so that state survives a resume.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from dlfp.training.trainer import Trainer


class Callback:
    """Base class with no-op hooks."""

    def on_train_start(self, trainer: "Trainer") -> None:
        """Called once before the first epoch."""

    def on_epoch_start(self, trainer: "Trainer") -> None:
        """Called at the start of each epoch."""

    def on_batch_end(self, trainer: "Trainer") -> None:
        """Called after each optimizer step."""

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Called after training and validation of each epoch."""

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called once after the last epoch."""

    def state_dict(self) -> dict[str, Any]:
        """State to store in checkpoints (default: none)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict`."""


class CallbackList(Callback):
    """Fans every hook out to a list of callbacks, in order.

    Args:
        callbacks: Callbacks to call.
    """

    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def __iter__(self):
        """Iterate over the wrapped callbacks."""
        return iter(self.callbacks)

    def on_train_start(self, trainer: "Trainer") -> None:
        """Fan out ``on_train_start``."""
        for cb in self.callbacks:
            cb.on_train_start(trainer)

    def on_epoch_start(self, trainer: "Trainer") -> None:
        """Fan out ``on_epoch_start``."""
        for cb in self.callbacks:
            cb.on_epoch_start(trainer)

    def on_batch_end(self, trainer: "Trainer") -> None:
        """Fan out ``on_batch_end``."""
        for cb in self.callbacks:
            cb.on_batch_end(trainer)

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Fan out ``on_epoch_end``."""
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        """Fan out ``on_train_end``."""
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def state_dict(self) -> dict[str, Any]:
        """Per-callback state, keyed by position and class name."""
        return {f"{i}:{type(cb).__name__}": cb.state_dict() for i, cb in enumerate(self.callbacks)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore per-callback state for callbacks whose key matches."""
        for i, cb in enumerate(self.callbacks):
            key = f"{i}:{type(cb).__name__}"
            if key in state:
                cb.load_state_dict(state[key])


def _improved(value: float, best: float | None, mode: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    return value < best - min_delta if mode == "min" else value > best + min_delta


class _Monitoring:
    """Shared lookup of the monitored metric with a one-time warning if absent."""

    monitor: str
    _warned = False

    def _value(self, trainer: "Trainer") -> float | None:
        value = trainer.last_metrics.get(self.monitor)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            if not self._warned:
                trainer.logger.info(f"[{type(self).__name__}] metric '{self.monitor}' not available "
                                    f"(have: {sorted(trainer.last_metrics)}); skipping.")
                self._warned = True
            return None
        return float(value)


class EarlyStopping(Callback, _Monitoring):
    """Stop when ``monitor`` has not improved for ``patience`` epochs.

    Args:
        monitor: Metric key in ``trainer.last_metrics`` (e.g. ``"val_loss"``).
        mode: ``"min"`` or ``"max"``.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum change counted as an improvement.
    """

    def __init__(self, monitor: str, mode: str, patience: int, min_delta: float = 0.0) -> None:
        self.monitor, self.mode, self.patience, self.min_delta = monitor, mode, patience, min_delta
        self.best: float | None = None
        self.wait = 0

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Update the patience counter and request a stop if exhausted."""
        value = self._value(trainer)
        if value is None:
            return
        if _improved(value, self.best, self.mode, self.min_delta):
            self.best, self.wait = value, 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            trainer.should_stop = True
            trainer.logger.info(
                f"[EarlyStopping] stopping after epoch {trainer.current_epoch}: {self.monitor} has not "
                f"improved on {self.best:.6g} ({self.mode}) for {self.wait} epoch(s) (patience={self.patience})."
            )

    def state_dict(self) -> dict[str, Any]:
        """Best value and patience counter."""
        return {"best": self.best, "wait": self.wait}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore best value and patience counter."""
        self.best, self.wait = state.get("best"), int(state.get("wait", 0))


class ModelCheckpoint(Callback, _Monitoring):
    """Write ``ckpt/best.pt`` when ``monitor`` improves and ``ckpt/last.pt`` every epoch.

    The best value is kept in ``trainer.best_metric`` so it is saved in, and
    restored from, every checkpoint. If the monitored metric is unavailable
    (e.g. no validation split), only ``last.pt`` is written.

    Args:
        monitor: Metric key in ``trainer.last_metrics``.
        mode: ``"min"`` or ``"max"``.
        save_last: Also refresh ``last.pt`` every epoch.
    """

    def __init__(self, monitor: str, mode: str, save_last: bool = True) -> None:
        self.monitor, self.mode, self.save_last = monitor, mode, save_last

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Save ``best.pt`` on improvement, then ``last.pt``."""
        value = self._value(trainer)
        if value is not None and _improved(value, trainer.best_metric, self.mode):
            trainer.best_metric = value
            trainer.save_checkpoint(trainer.logger.ckpt_dir / "best.pt")
            trainer.logger.info(f"[ModelCheckpoint] new best {self.monitor}={value:.6g} -> best.pt")
        if self.save_last:
            trainer.save_checkpoint(trainer.logger.ckpt_dir / "last.pt")


class LRSchedulerCallback(Callback):
    """Step a scheduler per epoch (``on_epoch_end``) or per batch (``on_batch_end``).

    Args:
        scheduler: Scheduler with a ``step()`` method.
        interval: ``"epoch"`` or ``"step"``; defaults to the scheduler's own
            ``interval`` attribute, else ``"epoch"``.
    """

    def __init__(self, scheduler: Any, interval: str | None = None) -> None:
        self.scheduler = scheduler
        self.interval = interval or getattr(scheduler, "interval", "epoch")
        if self.interval not in ("epoch", "step"):
            raise ValueError(f"interval must be 'epoch' or 'step', got {self.interval!r}")

    def on_batch_end(self, trainer: "Trainer") -> None:
        """Step per optimizer step if ``interval == "step"``."""
        if self.interval == "step":
            self.scheduler.step()

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """Step per epoch if ``interval == "epoch"``."""
        if self.interval == "epoch":
            self.scheduler.step()


def build_callbacks(cfg: Any, scheduler: Any | None, logger: Any) -> list[Callback]:
    """Standard callback list for a config.

    Order matters: the scheduler steps and early stopping updates *before*
    ``ModelCheckpoint`` writes, so ``last.pt`` captures the state the next epoch
    will start from.

    Args:
        cfg: Full :class:`~dlfp.config.Config`.
        scheduler: Scheduler or ``None``.
        logger: Run logger (unused here; callbacks use ``trainer.logger``).

    Returns:
        List of callbacks.
    """
    callbacks: list[Callback] = []
    if scheduler is not None:
        callbacks.append(LRSchedulerCallback(scheduler))
    if cfg.train.early_stopping_patience:
        callbacks.append(EarlyStopping(cfg.train.monitor, cfg.train.monitor_mode, cfg.train.early_stopping_patience))
    callbacks.append(ModelCheckpoint(cfg.train.monitor, cfg.train.monitor_mode, save_last=True))
    return callbacks
