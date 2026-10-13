"""Reproducibility helpers: global seeding and seeded DataLoader workers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device).

    With ``deterministic=True`` cuDNN is restricted to deterministic kernels,
    autotuning is disabled, and PyTorch warns (rather than errors) whenever an
    op has no deterministic implementation.

    Args:
        seed: Seed value.
        deterministic: Whether to request deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader``: derive per-worker seeds.

    PyTorch seeds each worker's torch RNG from the loader's generator; this
    propagates that seed to ``random`` and ``numpy`` inside the worker.

    Args:
        worker_id: Worker index (unused; the torch seed already differs per worker).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with ``seed``.

    Args:
        seed: Seed value.

    Returns:
        The seeded generator.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
