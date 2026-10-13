"""Model-agnostic training loop.

RULE: this file contains no model-type branching: no isinstance checks on the
model and no conditionals on the model's name. Everything model-specific goes
through the :class:`~dlfp.models.base.BaseModel` interface (``compute_loss``,
``batch_size``, ``finalize_metrics``, ``describe``). If a new model seems to
need a special case here, the interface is wrong: fix the interface instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dlfp.metrics import MetricTracker
from dlfp.training.callbacks import Callback, CallbackList, LRSchedulerCallback
from dlfp.training.checkpoint import load_checkpoint, restore_state, save_checkpoint
from dlfp.utils.device import move_to_device, resolve_device


class Trainer:
    """Runs epochs of optimisation and evaluation, driving callbacks.

    Public state for callbacks: ``should_stop``, ``current_epoch``,
    ``global_step``, ``last_metrics`` (``train_*`` and ``val_*`` keys),
    ``best_metric``, ``logger``, ``model``, ``optimizer``, ``cfg``.

    Args:
        model: A :class:`~dlfp.models.base.BaseModel`.
        train_loader: Training batches (dicts).
        val_loader: Validation batches, or ``None``.
        optimizer: Optimizer over the model's parameters.
        cfg: Full :class:`~dlfp.config.Config`.
        logger: :class:`~dlfp.utils.logging.RunLogger`.
        callbacks: Callbacks to run.
        device: Device (defaults to ``cfg.device``).
    """

    def __init__(self, model: Any, train_loader: DataLoader, val_loader: DataLoader | None,
                 optimizer: torch.optim.Optimizer, cfg: Any, logger: Any,
                 callbacks: list[Callback] | None = None, device: torch.device | str | None = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(device if device is not None else cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.logger = logger
        self.callbacks = CallbackList(callbacks)
        self.should_stop = False
        self.current_epoch = 0
        self.start_epoch = 1
        self.global_step = 0
        self.best_metric: float | None = None
        self.last_metrics: dict[str, float] = {}

    @property
    def scheduler(self) -> Any | None:
        """The scheduler driven by an :class:`LRSchedulerCallback`, if any."""
        for cb in self.callbacks:
            if isinstance(cb, LRSchedulerCallback):
                return cb.scheduler
        return None

    def fit(self) -> dict:
        """Train from ``start_epoch`` to ``cfg.train.epochs``.

        Returns:
            The metrics of the last completed epoch.
        """
        self.logger.info(self.model.describe())
        self.logger.info(f"device={self.device} | epochs {self.start_epoch}..{self.cfg.train.epochs} | "
                         f"{len(self.train_loader)} train batches/epoch")
        self.callbacks.on_train_start(self)
        for epoch in range(self.start_epoch, self.cfg.train.epochs + 1):
            self.current_epoch = epoch
            self.callbacks.on_epoch_start(self)
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, "val") if self.val_loader is not None else {}
            self.logger.log_metrics(epoch, self.global_step, train_metrics, "train")
            if val_metrics:
                self.logger.log_metrics(epoch, self.global_step, val_metrics, "val")
            self.last_metrics = {**{f"train_{k}": v for k, v in train_metrics.items()},
                                 **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.callbacks.on_epoch_end(self)
            if self.should_stop:
                break
        self.callbacks.on_train_end(self)
        return self.last_metrics

    def train_epoch(self, epoch: int) -> dict:
        """Run one epoch of optimisation.

        The shuffle order is re-seeded from ``(seed, epoch)``, so a resumed run
        sees exactly the batches an uninterrupted run would have.

        Args:
            epoch: 1-based epoch number.

        Returns:
            Weighted epoch means of the model's logged metrics, plus ``lr``.
        """
        self.model.train()
        generator = getattr(self.train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(self.cfg.seed + epoch)
        tracker = MetricTracker()
        n_batches = len(self.train_loader)
        # Log at least ~5 times per epoch, so short (smoke) epochs still show progress.
        interval = max(1, min(self.cfg.train.log_interval, n_batches // 5 or 1))
        bar = tqdm(self.train_loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
        for i, batch in enumerate(bar, start=1):
            batch = move_to_device(batch, self.device)
            loss, logs = self.model.compute_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()

            self.global_step += 1
            tracker.update(logs, n=self.model.batch_size(batch))
            bar.set_postfix({k: f"{v:.4f}" for k, v in logs.items()})
            if i == 1 or i % interval == 0 or i == n_batches:
                running = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                self.logger.info(f"epoch {epoch} step {i}/{n_batches} (global {self.global_step}) "
                                 f"{running} lr={self.optimizer.param_groups[0]['lr']:.3g}")
            self.callbacks.on_batch_end(self)
        bar.close()
        metrics = self.model.finalize_metrics(tracker.result())
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> dict:
        """Average the model's logged metrics over ``loader`` in eval mode.

        Args:
            loader: Batches to evaluate.
            split: Name used in the progress bar.

        Returns:
            Weighted means of the model's logged metrics (unprefixed).
        """
        was_training = self.model.training
        self.model.eval()
        tracker = MetricTracker()
        for batch in tqdm(loader, desc=split, leave=False, dynamic_ncols=True):
            batch = move_to_device(batch, self.device)
            _, logs = self.model.compute_loss(batch)
            tracker.update(logs, n=self.model.batch_size(batch))
        self.model.train(was_training)
        return self.model.finalize_metrics(tracker.result())

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimizer, scheduler, callback and RNG state to ``path``.

        Args:
            path: Destination file.
        """
        save_checkpoint(path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                        epoch=self.current_epoch, global_step=self.global_step, best_metric=self.best_metric,
                        config=self.cfg, extra={"callbacks": self.callbacks.state_dict()})

    def resume_from(self, path: str | Path) -> None:
        """Restore full training state; :meth:`fit` then continues at the next epoch.

        Args:
            path: Checkpoint written by :meth:`save_checkpoint` (e.g. ``ckpt/last.pt``).
        """
        ckpt = load_checkpoint(path, map_location=self.device)
        info = restore_state(ckpt, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler)
        self.callbacks.load_state_dict(ckpt.get("extra", {}).get("callbacks", {}))
        self.current_epoch = info["epoch"]
        self.start_epoch = info["epoch"] + 1
        self.global_step = info["global_step"]
        self.best_metric = info["best_metric"]
        self.logger.info(f"Resumed from {path}: completed epoch {info['epoch']}, global step "
                         f"{self.global_step}; continuing at epoch {self.start_epoch}.")
