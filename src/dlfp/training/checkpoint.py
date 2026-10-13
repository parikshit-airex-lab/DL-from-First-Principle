"""Saving and restoring full training state.

A checkpoint holds model, optimizer and scheduler state, progress counters,
the resolved config, and the state of every random number generator (Python,
NumPy, torch CPU, all CUDA devices). Restoring the RNG state is what makes a
resumed run match an uninterrupted one: dropout masks, data shuffling and
sampling continue from exactly where they left off instead of replaying (or
diverging from) the original sequence.

Everything is stored as tensors and plain Python containers, so checkpoints
load with ``torch.load(weights_only=True)`` and never execute pickled code.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dlfp.config import Config, to_dict


def _rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (name, keys.tolist(), int(pos), int(has_gauss), float(cached)),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda") or []
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    config: Config | dict | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a complete training checkpoint (atomically) to ``path``.

    Args:
        path: Destination ``.pt`` file.
        model: Model to save.
        optimizer: Optimizer to save (or ``None``).
        scheduler: Scheduler with ``state_dict()`` (or ``None``).
        epoch: Last *completed* epoch.
        global_step: Optimizer steps taken so far.
        best_metric: Best monitored value so far.
        config: Resolved config (dataclass or dict).
        extra: Additional plain-data state (e.g. callback state).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": None if best_metric is None else float(best_metric),
        "config": to_dict(config) if isinstance(config, Config) else config,
        "rng": _rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: Device to map tensors onto.

    Returns:
        The checkpoint dict.
    """
    return torch.load(path, map_location=map_location, weights_only=True)


def restore_state(
    ckpt: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load checkpoint state into live objects.

    Args:
        ckpt: Output of :func:`load_checkpoint`.
        model: Model to load weights into.
        optimizer: Optimizer to restore (skipped if ``None``).
        scheduler: Scheduler to restore (skipped if ``None``).
        restore_rng: Also restore every RNG state.

    Returns:
        ``{"epoch", "global_step", "best_metric"}``.
    """
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng and ckpt.get("rng") is not None:
        _set_rng_state(ckpt["rng"])
    return {"epoch": ckpt["epoch"], "global_step": ckpt["global_step"], "best_metric": ckpt["best_metric"]}
