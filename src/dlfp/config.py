"""Typed configuration with a three-layer merge.

A run's configuration is assembled as::

    configs/base.yaml  <  configs/<experiment>.yaml  <  CLI overrides

each layer overriding the previous one. Nested mappings are merged key by key
(not replaced wholesale), so an experiment config that sets only ``train.lr``
keeps every other ``train`` default from ``base.yaml``.

CLI overrides use dotted paths, in either ``--train.lr 3e-4`` or
``--train.lr=3e-4`` form. Values are parsed with ``yaml.safe_load`` so that
``true``, ``null``, ``[32, 64]`` and numbers get their natural types.

Nothing in this module ever writes to ``configs/``; :func:`save_config` is only
used for the resolved copy stored inside a run directory.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# configs/ lives at the repository root: <root>/src/dlfp/config.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = _REPO_ROOT / "configs" / "base.yaml"


@dataclass
class ModelConfig:
    """Which model to build and its constructor parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset selection and data-loading options."""

    name: str
    root: str = "./data"
    batch_size: int = 64
    eval_batch_size: int | None = None
    num_workers: int = 4
    val_split: float = 0.1
    subset_size: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Optimisation and training-loop options."""

    epochs: int = 10
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    grad_clip: float | None = 1.0
    scheduler: str | None = None  # None | "cosine" | "step"
    warmup_steps: int = 0
    log_interval: int = 50
    early_stopping_patience: int | None = None
    monitor: str = "val_loss"
    monitor_mode: str = "min"


@dataclass
class Config:
    """Complete, resolved configuration for one run."""

    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    seed: int = 42
    device: str = "auto"
    run_name: str | None = None
    output_dir: str = "runs"
    deterministic: bool = True


_TOP_LEVEL_KEYS = {f.name for f in fields(Config)}
_FLOAT_FIELDS = {
    "lr", "weight_decay", "momentum", "grad_clip", "val_split",
}


def _parse_scalar(text: str) -> Any:
    """Parse a CLI value string into a Python value.

    ``yaml.safe_load`` handles booleans, null, lists and most numbers, but
    PyYAML follows YAML 1.1 and reads exponent floats without a dot (``3e-4``)
    as strings, so string results get a numeric retry.

    Args:
        text: Raw value string.

    Returns:
        The parsed value.
    """
    value = yaml.safe_load(text)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (empty file -> empty dict).

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dicts are merged key by key; any other value in ``override`` replaces
    the one in ``base``. Neither input is mutated.

    Args:
        base: Lower-priority mapping.
        override: Higher-priority mapping.

    Returns:
        A new merged dict.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg_dict: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted-path CLI overrides to a config dict.

    Accepts ``["--train.lr", "3e-4"]`` and ``["--train.lr=3e-4"]`` (mixed freely).
    Intermediate dicts are created as needed.

    Args:
        cfg_dict: Config mapping to start from (not mutated).
        overrides: Flat list of CLI tokens.

    Returns:
        A new dict with overrides applied.

    Raises:
        ValueError: On a malformed override list.
    """
    out = copy.deepcopy(cfg_dict)
    tokens = list(overrides or [])
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Expected an override starting with '--', got {token!r}")
        key = token[2:]
        if "=" in key:
            key, raw = key.split("=", 1)
            i += 1
        else:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"Override {token!r} is missing a value")
            raw = tokens[i + 1]
            i += 2
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _parse_scalar(raw)
    return out


def _coerce_floats(section: dict[str, Any]) -> dict[str, Any]:
    """Cast float-typed fields given as strings or ints (e.g. ``1e-3`` in YAML)."""
    out = dict(section)
    for key in _FLOAT_FIELDS & out.keys():
        if out[key] is not None:
            out[key] = float(out[key])
    return out


def _build_section(cls: type, section: Any, name: str) -> Any:
    """Construct a dataclass section, turning bad keys into ``ValueError``."""
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping, got {type(section).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) in '{name}': {sorted(unknown)}. Allowed: {sorted(allowed)}")
    try:
        return cls(**_coerce_floats(section))
    except TypeError as exc:
        raise ValueError(f"Invalid '{name}' section: {exc}") from exc


def from_dict(cfg_dict: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a merged dict.

    Args:
        cfg_dict: Fully merged configuration mapping.

    Returns:
        The validated config.

    Raises:
        ValueError: On unknown keys, missing sections or out-of-range values.
    """
    unknown = set(cfg_dict) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}. Allowed: {sorted(_TOP_LEVEL_KEYS)}")
    for required in ("model", "data"):
        if required not in cfg_dict:
            raise ValueError(f"Config is missing the required '{required}' section")

    model = _build_section(ModelConfig, cfg_dict["model"], "model")
    data = _build_section(DataConfig, cfg_dict["data"], "data")
    train = _build_section(TrainConfig, cfg_dict.get("train", {}) or {}, "train")
    train.betas = tuple(float(b) for b in train.betas)  # YAML yields a list

    top = {k: v for k, v in cfg_dict.items() if k not in ("model", "data", "train")}
    cfg = Config(model=model, data=data, train=train, **top)

    if cfg.train.monitor_mode not in ("min", "max"):
        raise ValueError(f"train.monitor_mode must be 'min' or 'max', got {cfg.train.monitor_mode!r}")
    if not 0.0 <= cfg.data.val_split < 1.0:
        raise ValueError(f"data.val_split must be in [0, 1), got {cfg.data.val_split}")
    if len(cfg.train.betas) != 2:
        raise ValueError(f"train.betas must have two values, got {cfg.train.betas}")
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    """Convert a :class:`Config` to a plain YAML-serialisable dict.

    Args:
        cfg: Config to convert.

    Returns:
        Nested dict with tuples turned into lists.
    """
    d = asdict(cfg)
    d["train"]["betas"] = list(d["train"]["betas"])
    return d


def save_config(cfg: Config, path: str | Path) -> None:
    """Write a resolved config to ``path`` as YAML.

    Args:
        cfg: Config to save.
        path: Destination file (parent directories are created).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load ``base.yaml`` -> ``config_path`` -> CLI overrides into a :class:`Config`.

    ``base.yaml`` is located relative to the installed package (the repo's
    ``configs/`` directory), so this works from any working directory. If it is
    missing there, a ``base.yaml`` beside ``config_path`` is used instead, if any.

    Args:
        config_path: Experiment (or saved run) config file.
        overrides: Optional CLI override tokens.

    Returns:
        The validated, merged config.
    """
    config_path = Path(config_path).expanduser().resolve()
    base_path = BASE_CONFIG_PATH
    if not base_path.is_file():
        base_path = config_path.parent / "base.yaml"
    merged = load_yaml(base_path) if base_path.is_file() else {}
    if config_path != base_path.resolve():
        merged = deep_merge(merged, load_yaml(config_path))
    merged = apply_overrides(merged, overrides)
    return from_dict(merged)
