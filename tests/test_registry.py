"""Tests for the name -> constructor registries."""

from __future__ import annotations

import pytest

from dlfp import registry


@pytest.fixture
def clean_models(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Swap the model table for an empty one for the duration of a test."""
    table: dict = {}
    monkeypatch.setattr(registry, "MODELS", table)
    return table


def test_register_and_build(clean_models: dict) -> None:
    @registry.register_model("Toy")
    class Toy:
        def __init__(self, width: int) -> None:
            self.width = width

    assert "toy" in clean_models
    obj = registry.build_model("TOY", width=3)
    assert isinstance(obj, Toy) and obj.width == 3
    assert registry.available_models() == ["toy"]


def test_duplicate_registration_raises(clean_models: dict) -> None:
    @registry.register_model("dup")
    class A:
        pass

    with pytest.raises(ValueError, match="dup"):

        @registry.register_model("DUP")
        class B:
            pass


def test_unknown_name_lists_all_available(clean_models: dict) -> None:
    for name in ("zeta", "alpha", "mid"):
        registry.register_model(name)(type(name, (), {}))
    with pytest.raises(KeyError) as excinfo:
        registry.build_model("nope")
    message = str(excinfo.value)
    assert "nope" in message
    assert "alpha, mid, zeta" in message  # every name, sorted


def test_all_models_registered_after_import() -> None:
    import dlfp.models  # noqa: F401

    assert {"cnn", "rnn", "lstm", "transformer"} <= set(registry.available_models())


def test_datasets_and_optimizers_registered_after_import() -> None:
    import dlfp.data  # noqa: F401
    import dlfp.training  # noqa: F401

    assert {"mnist", "cifar10", "charlm", "imdb"} <= set(registry.available_datasets())
    assert {"sgd", "adam", "torch_sgd", "torch_adam"} <= set(registry.available_optimizers())
