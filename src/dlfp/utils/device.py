"""Device selection and batch transfer."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(spec: str | torch.device | None = "auto") -> torch.device:
    """Turn a device spec into a ``torch.device``.

    ``"auto"`` (or ``None``) picks CUDA when available, otherwise CPU. Any other
    string (``"cpu"``, ``"cuda"``, ``"cuda:1"``) is passed through, so the GPU
    index is always the user's choice, never hardcoded here.

    Args:
        spec: Device spec.

    Returns:
        The resolved device.
    """
    if isinstance(spec, torch.device):
        return spec
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move every tensor value of ``batch`` to ``device``; leave other values alone.

    Args:
        batch: Mapping of names to tensors (or arbitrary values).
        device: Target device.

    Returns:
        A new dict with tensors on ``device``.
    """
    non_blocking = device.type == "cuda"
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
