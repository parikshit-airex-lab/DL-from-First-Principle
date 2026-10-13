"""Tests for config loading, merging and CLI overrides."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dlfp.config import (
    Config,
    apply_overrides,
    deep_merge,
    load_config,
    load_yaml,
    save_config,
)


def test_deep_merge_is_recursive() -> None:
    base = {"a": {"x": 1, "y": {"p": 1, "q": 2}}, "b": 1}
    override = {"a": {"y": {"q": 20}}, "c": 3}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": {"p": 1, "q": 20}}, "b": 1, "c": 3}
    assert base["a"]["y"]["q"] == 2  # inputs untouched


def test_experiment_config_beats_base(config_dir: str) -> None:
    base = load_yaml(os.path.join(config_dir, "base.yaml"))
    cfg = load_config(os.path.join(config_dir, "transformer_charlm.yaml"))
    assert base["train"]["lr"] != cfg.train.lr
    assert cfg.train.lr == pytest.approx(3e-4)
    # Keys the experiment does not set fall through from base.yaml.
    assert cfg.data.val_split == base["data"]["val_split"]
    assert cfg.train.monitor == base["train"]["monitor"]


def test_cli_override_beats_both(config_dir: str) -> None:
    path = os.path.join(config_dir, "transformer_charlm.yaml")
    cfg = load_config(path, ["--train.lr", "5e-5", "--data.batch_size=7", "--seed", "3"])
    assert cfg.train.lr == pytest.approx(5e-5)
    assert cfg.data.batch_size == 7
    assert cfg.seed == 3


def test_override_parses_float_and_types() -> None:
    out = apply_overrides(
        {}, ["--train.lr", "3e-4", "--model.params.channels=[8, 16]", "--train.scheduler", "null",
             "--deterministic", "false", "--a.b.c", "hello"]
    )
    assert isinstance(out["train"]["lr"], float) and out["train"]["lr"] == pytest.approx(3e-4)
    assert out["model"]["params"]["channels"] == [8, 16]
    assert out["train"]["scheduler"] is None
    assert out["deterministic"] is False
    assert out["a"]["b"]["c"] == "hello"  # intermediate keys created


def test_save_load_round_trip(config_dir: str, tmp_path: Path) -> None:
    cfg = load_config(os.path.join(config_dir, "lstm_imdb.yaml"), ["--train.lr", "2e-3"])
    out = tmp_path / "config.yaml"
    save_config(cfg, out)
    again = load_config(out)
    assert isinstance(again, Config)
    assert again == cfg


def _digest_dir(path: str) -> dict[str, str]:
    return {
        name: hashlib.sha256(Path(path, name).read_bytes()).hexdigest()
        for name in sorted(os.listdir(path))
    }


def test_configs_untouched_by_loading(config_dir: str) -> None:
    before = _digest_dir(config_dir)
    for name in before:
        if name != "base.yaml":
            load_config(os.path.join(config_dir, name), ["--train.lr", "1e-2", "--data.batch_size=3"])
    assert _digest_dir(config_dir) == before


def test_validation_errors(config_dir: str) -> None:
    path = os.path.join(config_dir, "cnn_mnist.yaml")
    with pytest.raises(ValueError, match="bogus"):
        load_config(path, ["--bogus", "1"])
    with pytest.raises(ValueError, match="monitor_mode"):
        load_config(path, ["--train.monitor_mode", "sideways"])
    with pytest.raises(ValueError, match="val_split"):
        load_config(path, ["--data.val_split", "1.0"])
