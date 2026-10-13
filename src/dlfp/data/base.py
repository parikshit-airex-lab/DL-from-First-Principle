"""Dataset bundle, collate functions and DataLoader construction.

Every dataset builder registered with ``@register_dataset`` takes the
:class:`~dlfp.config.DataConfig` and returns a :class:`DatasetBundle`. The
loaders always yield *batch dicts*:

* images: ``{"inputs": (B, C, H, W), "targets": (B,)}``
* text:   ``{"inputs": (B, T), "targets": (B, T) or (B,), "lengths": (B,), "mask": (B, T)}``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from dlfp.config import DataConfig
from dlfp.losses import IGNORE_INDEX
from dlfp.registry import build_dataset
from dlfp.utils.seed import get_generator, seed_worker

Batch = dict[str, torch.Tensor]


@dataclass
class DatasetBundle:
    """Train/val/test splits plus metadata the model and scripts need.

    ``meta`` may carry ``vocab_size``, ``num_classes``, ``pad_idx``,
    ``in_channels``, ``tokenizer``, and ``collate`` (``"image"`` or ``"pad"``).
    """

    train: Dataset
    val: Dataset | None
    test: Dataset | None
    meta: dict[str, Any] = field(default_factory=dict)


def pad_collate(pad_idx: int) -> Callable[[list[dict[str, torch.Tensor]]], Batch]:
    """Build a collate function that right-pads variable-length sequences.

    Each item is ``{"inputs": (T_i,), "targets": (T_i,) or ()}``. Inputs are
    padded with ``pad_idx`` to the longest sequence in the batch. Sequence
    targets (language modelling) are padded with ``IGNORE_INDEX`` so padded
    positions never enter the loss; scalar targets (classification) are stacked.

    Args:
        pad_idx: Padding token id.

    Returns:
        ``collate(items) -> {"inputs", "targets", "lengths", "mask"}``.
    """

    def collate(items: list[dict[str, torch.Tensor]]) -> Batch:
        lengths = torch.tensor([len(it["inputs"]) for it in items], dtype=torch.long)
        t_max = int(lengths.max())
        b = len(items)
        inputs = torch.full((b, t_max), pad_idx, dtype=torch.long)
        sequence_targets = items[0]["targets"].dim() == 1
        targets = torch.full((b, t_max), IGNORE_INDEX, dtype=torch.long) if sequence_targets else None
        for i, it in enumerate(items):
            n = len(it["inputs"])
            inputs[i, :n] = it["inputs"]
            if sequence_targets:
                targets[i, :n] = it["targets"]
        if not sequence_targets:
            targets = torch.stack([it["targets"] for it in items]).long()
        mask = (torch.arange(t_max).unsqueeze(0) < lengths.unsqueeze(1)).float()
        return {"inputs": inputs, "targets": targets, "lengths": lengths, "mask": mask}

    return collate


def image_collate(items: list[Any]) -> Batch:
    """Stack ``(image, label)`` pairs (or dicts) into ``{"inputs", "targets"}``.

    Args:
        items: List of ``(C, H, W)`` image tensors with integer labels.

    Returns:
        Batch dict.
    """
    if isinstance(items[0], dict):
        images = [it["inputs"] for it in items]
        labels = [it["targets"] for it in items]
    else:
        images, labels = zip(*items)
    return {"inputs": torch.stack(list(images)), "targets": torch.as_tensor(list(labels), dtype=torch.long)}


def split_dataset(dataset: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset | None]:
    """Randomly split ``dataset`` into train/val with a seeded generator.

    Args:
        dataset: Dataset to split.
        val_split: Fraction for validation in ``[0, 1)``; 0 means no val split.
        seed: Seed making the split reproducible.

    Returns:
        ``(train, val)``; ``val`` is ``None`` when ``val_split`` is 0.
    """
    if val_split <= 0:
        return dataset, None
    n_val = max(1, int(round(len(dataset) * val_split)))
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=get_generator(seed))
    return train, val


def _subset(dataset: Dataset | None, n: int | None) -> Dataset | None:
    """First ``n`` items of ``dataset`` (unchanged if ``n`` is None or too large)."""
    if dataset is None or n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, range(n))


def build_dataloaders(
    cfg: DataConfig, seed: int
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict[str, Any]]:
    """Build the dataset named in ``cfg`` and wrap each split in a DataLoader.

    ``cfg.subset_size`` keeps only the first N items of each split (this is what
    makes smoke runs fast). Only the training loader shuffles; it uses a seeded
    generator and ``seed_worker`` so shuffling and worker RNG are reproducible.

    Args:
        cfg: Data config.
        seed: Run seed.

    Returns:
        ``(train_loader, val_loader, test_loader, meta)``.
    """
    bundle: DatasetBundle = build_dataset(cfg.name, cfg)
    meta = dict(bundle.meta)
    if meta.get("collate", "image") == "pad":
        collate = pad_collate(meta.get("pad_idx", 0))
    else:
        collate = image_collate
    eval_bs = cfg.eval_batch_size or cfg.batch_size
    common = dict(num_workers=cfg.num_workers, collate_fn=collate, worker_init_fn=seed_worker,
                  pin_memory=torch.cuda.is_available(), persistent_workers=False)

    def loader(ds: Dataset | None, train: bool) -> DataLoader | None:
        ds = _subset(ds, cfg.subset_size)
        if ds is None or len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=cfg.batch_size if train else eval_bs, shuffle=train,
                          generator=get_generator(seed), drop_last=False, **common)

    train_loader = loader(bundle.train, True)
    if train_loader is None:
        raise ValueError(f"Dataset '{cfg.name}' produced an empty training split")
    return train_loader, loader(bundle.val, False), loader(bundle.test, False), meta
