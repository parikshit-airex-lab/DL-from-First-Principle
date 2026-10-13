"""Name -> constructor registries for models, datasets and optimizers.

Components register themselves with a decorator at import time::

    @register_model("cnn")
    class CNN(BaseModel): ...

and are later built by name from a config (``build_model("cnn", **params)``).
Names are matched case-insensitively and stored lowercase. Looking up an
unknown name raises a ``KeyError`` listing every registered name, which makes a
typo in a config immediately diagnosable.

This module deliberately imports nothing from ``dlfp.models`` or ``dlfp.data``
so that those packages can import it without creating a cycle.
"""

from __future__ import annotations

from typing import Any, Callable

MODELS: dict[str, type] = {}
DATASETS: dict[str, Callable] = {}
OPTIMIZERS: dict[str, type] = {}


def _register(table: dict[str, Any], kind: str, name: str) -> Callable[[Any], Any]:
    """Return a decorator that inserts its argument into ``table`` under ``name``.

    Args:
        table: Registry dict to insert into.
        kind: Human-readable registry kind used in error messages.
        name: Registration name (case-insensitive).

    Returns:
        A decorator that registers and returns the decorated object unchanged.
    """
    key = name.lower()

    def decorator(obj: Any) -> Any:
        if key in table:
            raise ValueError(
                f"Duplicate {kind} registration: '{key}' is already registered to "
                f"{table[key]!r}; cannot register {obj!r}."
            )
        table[key] = obj
        return obj

    return decorator


def _lookup(table: dict[str, Any], kind: str, name: str) -> Any:
    """Fetch ``name`` from ``table`` or raise a ``KeyError`` listing all names.

    Args:
        table: Registry dict to search.
        kind: Human-readable registry kind used in error messages.
        name: Name to look up (case-insensitive).

    Returns:
        The registered object.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    key = name.lower()
    if key not in table:
        available = ", ".join(sorted(table)) or "<none>"
        raise KeyError(f"Unknown {kind} '{name}'. Available {kind}s: [{available}]")
    return table[key]


def register_model(name: str) -> Callable[[type], type]:
    """Class decorator registering a model under ``name``.

    Args:
        name: Registry key (case-insensitive).

    Returns:
        Decorator returning the class unchanged.
    """
    return _register(MODELS, "model", name)


def register_dataset(name: str) -> Callable[[Callable], Callable]:
    """Decorator registering a dataset builder under ``name``.

    Args:
        name: Registry key (case-insensitive).

    Returns:
        Decorator returning the builder unchanged.
    """
    return _register(DATASETS, "dataset", name)


def register_optimizer(name: str) -> Callable[[type], type]:
    """Class decorator registering an optimizer under ``name``.

    Args:
        name: Registry key (case-insensitive).

    Returns:
        Decorator returning the class unchanged.
    """
    return _register(OPTIMIZERS, "optimizer", name)


def build_model(name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate the model registered under ``name``.

    Args:
        name: Registered model name.
        *args: Positional constructor arguments.
        **kwargs: Keyword constructor arguments.

    Returns:
        The constructed model.
    """
    return _lookup(MODELS, "model", name)(*args, **kwargs)


def build_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call the dataset builder registered under ``name``.

    Args:
        name: Registered dataset name.
        *args: Positional builder arguments.
        **kwargs: Keyword builder arguments.

    Returns:
        Whatever the builder returns (a ``DatasetBundle`` for built-in datasets).
    """
    return _lookup(DATASETS, "dataset", name)(*args, **kwargs)


def build_optimizer(name: str, params: Any, **kwargs: Any) -> Any:
    """Instantiate the optimizer registered under ``name``.

    Args:
        name: Registered optimizer name.
        params: Iterable of parameters or parameter groups.
        **kwargs: Optimizer hyperparameters.

    Returns:
        The constructed optimizer.
    """
    return _lookup(OPTIMIZERS, "optimizer", name)(params, **kwargs)


def available_models() -> list[str]:
    """Return the sorted list of registered model names."""
    return sorted(MODELS)


def available_datasets() -> list[str]:
    """Return the sorted list of registered dataset names."""
    return sorted(DATASETS)


def available_optimizers() -> list[str]:
    """Return the sorted list of registered optimizer names."""
    return sorted(OPTIMIZERS)
