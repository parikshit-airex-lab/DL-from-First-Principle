"""Trainer, checkpointing, callbacks and optimizer tests on synthetic in-memory data."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import dlfp.models  # noqa: F401
import dlfp.training  # noqa: F401  (registers optimizers)
from dlfp.config import from_dict
from dlfp.data.base import image_collate, pad_collate
from dlfp.registry import build_model, build_optimizer
from dlfp.training import (
    EarlyStopping,
    Trainer,
    build_callbacks,
    build_optimizer_from_config,
    build_scheduler,
    load_checkpoint,
    restore_state,
    save_checkpoint,
)
from dlfp.utils.logging import RunLogger
from dlfp.utils.seed import get_generator, set_seed


def _cfg(epochs: int = 2, **train: object):
    return from_dict({
        "model": {"name": "cnn", "params": {}},
        "data": {"name": "synthetic", "batch_size": 16, "num_workers": 0},
        "train": {"epochs": epochs, "lr": 1e-2, "log_interval": 2, **train},
        "device": "cpu",
        "seed": 0,
    })


def _image_loaders(seed: int = 0) -> tuple[DataLoader, DataLoader]:
    g = torch.Generator().manual_seed(123)
    x = torch.randn(64, 1, 8, 8, generator=g)
    y = (x.mean(dim=(1, 2, 3)) > 0).long()
    ds = TensorDataset(x, y)
    train = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=image_collate, generator=get_generator(seed))
    val = DataLoader(ds, batch_size=32, collate_fn=image_collate)
    return train, val


def _tiny_cnn() -> torch.nn.Module:
    return build_model("cnn", in_channels=1, num_classes=2, channels=[4, 8], fc_hidden=8, dropout=0.1)


def _make_trainer(tmp_path: Path, name: str, cfg=None, model=None) -> Trainer:
    cfg = cfg or _cfg()
    set_seed(cfg.seed)
    model = model or _tiny_cnn()
    train, val = _image_loaders(cfg.seed)
    optimizer = build_optimizer_from_config(model, cfg)
    scheduler = build_scheduler(cfg.train.scheduler, optimizer, cfg, cfg.train.epochs * len(train))
    logger = RunLogger(tmp_path, name, cfg)
    return Trainer(model, train, val, optimizer, cfg, logger, build_callbacks(cfg, scheduler, logger), "cpu")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_two_epoch_run_writes_metrics(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, "run", _cfg(scheduler="cosine", warmup_steps=2))
    metrics = trainer.fit()
    run_dir = trainer.logger.run_dir
    rows = _read_csv(run_dir / "metrics.csv")
    assert [(r["epoch"], r["split"]) for r in rows] == [("1", "train"), ("1", "val"), ("2", "train"), ("2", "val")]
    assert {"train_loss", "train_acc", "val_loss", "val_acc"} <= set(metrics)
    assert (run_dir / "config.yaml").is_file() and (run_dir / "log.txt").is_file()
    assert (run_dir / "ckpt" / "last.pt").is_file() and (run_dir / "ckpt" / "best.pt").is_file()
    assert trainer.global_step == 2 * len(trainer.train_loader)
    trainer.logger.close()


def test_checkpoint_round_trip_reproduces_loss(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, "ck", _cfg(epochs=1))
    trainer.fit()
    batch = next(iter(trainer.val_loader))
    trainer.model.eval()
    with torch.no_grad():
        expected, _ = trainer.model.compute_loss(batch)
    path = tmp_path / "manual.pt"
    save_checkpoint(path, model=trainer.model, optimizer=trainer.optimizer, scheduler=None, epoch=1,
                    global_step=trainer.global_step, best_metric=0.5, config=trainer.cfg)
    after_save = torch.rand(3)

    fresh = _tiny_cnn()
    fresh_opt = build_optimizer("adam", fresh.parameters(), lr=1e-2)
    info = restore_state(load_checkpoint(path), model=fresh, optimizer=fresh_opt)
    fresh.eval()
    with torch.no_grad():
        got, _ = fresh.compute_loss(batch)
    assert torch.equal(got, expected)
    assert info == {"epoch": 1, "global_step": trainer.global_step, "best_metric": 0.5}
    assert torch.equal(torch.rand(3), after_save)  # RNG state restored too
    assert fresh_opt.state_dict()["state"].keys() == trainer.optimizer.state_dict()["state"].keys()
    trainer.logger.close()


def test_resume_continues_from_right_epoch_and_matches_uninterrupted(tmp_path: Path) -> None:
    full = _make_trainer(tmp_path / "full", "full", _cfg(epochs=2))
    full.fit()

    first = _make_trainer(tmp_path / "a", "a", _cfg(epochs=1))
    first.fit()
    last = first.logger.ckpt_dir / "last.pt"
    assert load_checkpoint(last)["epoch"] == 1

    resumed = _make_trainer(tmp_path / "b", "b", _cfg(epochs=2))
    resumed.resume_from(last)
    assert resumed.start_epoch == 2
    resumed.fit()
    rows = _read_csv(resumed.logger.run_dir / "metrics.csv")
    assert {r["epoch"] for r in rows} == {"2"}
    assert resumed.global_step == full.global_step
    for (name, p_full), (_, p_res) in zip(full.model.state_dict().items(), resumed.model.state_dict().items()):
        assert torch.allclose(p_full.float(), p_res.float(), atol=1e-6), name
    for t in (full, first, resumed):
        t.logger.close()


def test_early_stopping_triggers_on_flat_metric(tmp_path: Path) -> None:
    # lr = 0 -> weights never change -> val_loss never improves after epoch 1.
    trainer = _make_trainer(tmp_path, "es", _cfg(epochs=10, lr=0.0, early_stopping_patience=2))
    trainer.fit()
    assert trainer.should_stop
    assert trainer.current_epoch == 3
    assert "EarlyStopping" in (trainer.logger.run_dir / "log.txt").read_text()
    trainer.logger.close()


def test_early_stopping_unit() -> None:
    messages: list[str] = []
    fake = SimpleNamespace(should_stop=False, current_epoch=0, last_metrics={},
                           logger=SimpleNamespace(info=messages.append))
    cb = EarlyStopping("val_acc", "max", patience=2)
    for epoch, acc in enumerate([0.5, 0.6, 0.6, 0.59], start=1):
        fake.current_epoch, fake.last_metrics = epoch, {"val_acc": acc}
        cb.on_epoch_end(fake)
    assert fake.should_stop and messages


def _optimizer_equivalence(ours_name: str, ref_name: str, steps: int = 50, **kwargs: object) -> None:
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    opt_a = build_optimizer(ours_name, [a], **kwargs)
    opt_b = build_optimizer(ref_name, [b], **kwargs)
    target = torch.randn(5, 3)
    for _ in range(steps):
        for p, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            ((p - target) ** 2 * torch.arange(1.0, 4.0)).sum().backward()
            opt.step()
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_adam_matches_torch_adam() -> None:
    _optimizer_equivalence("adam", "torch_adam", lr=1e-2)
    _optimizer_equivalence("adam", "torch_adam", lr=3e-3, betas=(0.8, 0.99), weight_decay=0.1)


@pytest.mark.parametrize("kwargs", [dict(lr=1e-2), dict(lr=1e-2, momentum=0.9, weight_decay=1e-3),
                                    dict(lr=1e-2, momentum=0.9, nesterov=True)])
def test_sgd_matches_torch_sgd(kwargs: dict) -> None:
    _optimizer_equivalence("sgd", "torch_sgd", **kwargs)


def test_trainer_runs_padded_classification(tmp_path: Path) -> None:
    """The classification path with real padding, end to end through the trainer."""
    torch.manual_seed(0)
    items = []
    for i in range(48):
        n = int(torch.randint(2, 12, ()))
        label = i % 2
        ids = torch.randint(2, 6, (n,)) + 4 * label  # label-dependent token range
        items.append({"inputs": ids, "targets": torch.tensor(label)})
    loader = DataLoader(items, batch_size=8, shuffle=True, collate_fn=pad_collate(0), generator=get_generator(0))
    cfg = _cfg(epochs=3)
    model = build_model("lstm", vocab_size=12, embed_dim=8, hidden_size=16, mode="classification")
    logger = RunLogger(tmp_path, "cls", cfg)
    trainer = Trainer(model, loader, loader, build_optimizer_from_config(model, cfg), cfg, logger,
                      build_callbacks(cfg, None, logger), "cpu")
    metrics = trainer.fit()
    assert metrics["val_acc"] > 0.9
    logger.close()
