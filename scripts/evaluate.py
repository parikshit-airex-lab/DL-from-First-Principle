"""Evaluate a checkpoint on a data split.

Usage::

    python scripts/evaluate.py --checkpoint runs/<run>/ckpt/best.pt --split test [--device cpu]

The model and data are rebuilt from the config stored *inside the checkpoint*
(not from ``configs/``), so the evaluation matches what was trained. Extra
``--section.key value`` arguments override that stored config (e.g.
``--data.num_workers 0``).
"""

from __future__ import annotations

import argparse

import dlfp.data  # noqa: F401  (registers datasets)
import dlfp.models  # noqa: F401  (registers models)
from dlfp.config import apply_overrides, from_dict
from dlfp.data import build_dataloaders
from dlfp.registry import build_model
from dlfp.training import Trainer, load_checkpoint, restore_state
from dlfp.utils import resolve_device, set_seed


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's flags; the rest are overrides of the stored config.

    Returns:
        ``(known args, remaining override tokens)``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file (best.pt or last.pt).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate.")
    parser.add_argument("--device", default=None, help="Device override (default: config's device).")
    return parser.parse_known_args()


def main() -> None:
    """Load the checkpoint, evaluate the split and print a metrics table."""
    args, overrides = parse_args()
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = from_dict(apply_overrides(ckpt["config"], overrides))
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(args.device or cfg.device)

    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg.data, cfg.seed)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    if loader is None:
        raise SystemExit(f"Dataset '{cfg.data.name}' has no '{args.split}' split.")

    model = build_model(cfg.model.name, **cfg.model.params)
    restore_state(ckpt, model=model, restore_rng=False)
    trainer = Trainer(model, train_loader, None, optimizer=None, cfg=cfg, logger=None, device=device)
    metrics = trainer.evaluate(loader, split=args.split)

    width = max(len(k) for k in metrics) + 2
    print(f"\nCheckpoint: {args.checkpoint} (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    print(f"Model: {cfg.model.name} | data: {cfg.data.name} | split: {args.split} | batches: {len(loader)}")
    print("-" * (width + 14))
    print(f"{'metric':<{width}}{'value':>14}")
    print("-" * (width + 14))
    for k, v in metrics.items():
        print(f"{k:<{width}}{v:>14.6f}")
    print("-" * (width + 14))


if __name__ == "__main__":
    main()
