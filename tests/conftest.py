"""Shared pytest fixtures: force CPU, fixed seeds, float32 default dtype."""

from __future__ import annotations # --> Important to not execute/evaluate type hints when the code runs.

import os
import random

# Never touch a GPU from the test suite on the shared server. This must run
# before torch initialises CUDA, so it lives at import time of conftest.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


@pytest.fixture(autouse=True)
def _seed_and_dtype() -> None:
    """Seed every RNG and pin the default dtype before each test."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def config_dir() -> str:
    """Absolute path of the repository's ``configs/`` directory."""
    return CONFIG_DIR
