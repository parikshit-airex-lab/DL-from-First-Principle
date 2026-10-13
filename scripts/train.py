"""Train any registered model from a YAML config.

Usage::

    python scripts/train.py --config configs/cnn_mnist.yaml [--train.lr 3e-4 ...]
                            [--resume runs/<run>/ckpt/last.pt] [--run-name NAME]

Any ``--section.key value`` (or ``--section.key=value``) argument overrides the
config; see :mod:`dlfp.config`. Works from any working directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dlfp.data  # noqa: F401  (registers datasets)
import dlfp.models  # noqa: F401  (registers models)
from dlfp.config import load_config, save_config
from dlfp.data import build_dataloaders
from dlfp.registry import build_model
from dlfp.training import Trainer, build_callbacks, build_optimizer_from_config, build_scheduler
from dlfp.utils import RunLogger, set_seed

# Dataset metadata copied into the model params unless the config sets them.
META_KEYS = ("vocab_size", "num_classes", "pad_idx", "in_channels")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the script's own flags; everything else is a config override.

    Returns:
        ``(known args, remaining override tokens)``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment YAML (merged over configs/base.yaml).")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from (e.g. ckpt/last.pt).")
    parser.add_argument("--run-name", default=None, help="Run-directory prefix (default: config file stem).")
    return parser.parse_known_args()


def main() -> None:
    """Build everything from the config and run training."""
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    if args.run_name:
        cfg.run_name = args.run_name
    cfg.run_name = cfg.run_name or Path(args.config).stem
    # Store an absolute data root so evaluate.py/generate.py work from any directory.
    cfg.data.root = str(Path(cfg.data.root).expanduser().resolve())

    set_seed(cfg.seed, cfg.deterministic)
    logger = RunLogger(cfg.output_dir, cfg.run_name, cfg)
    logger.info(f"Run directory: {logger.run_dir}")

    train_loader, val_loader, _, meta = build_dataloaders(cfg.data, cfg.seed)
    for key in META_KEYS:
        if key in meta and key not in cfg.model.params:
            cfg.model.params[key] = meta[key]
    save_config(cfg, logger.run_dir / "config.yaml")  # now includes the injected params
    if meta.get("tokenizer") is not None:
        meta["tokenizer"].save(logger.ckpt_dir / "tokenizer.json")

    model = build_model(cfg.model.name, **cfg.model.params)
    optimizer = build_optimizer_from_config(model, cfg)
    scheduler = build_scheduler(cfg.train.scheduler, optimizer, cfg, cfg.train.epochs * len(train_loader))
    callbacks = build_callbacks(cfg, scheduler, logger)
    trainer = Trainer(model, train_loader, val_loader, optimizer, cfg, logger, callbacks)
    if args.resume:
        trainer.resume_from(Path(args.resume).expanduser())
    metrics = trainer.fit()

    logger.info("Final metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    logger.close()
    print(f"Run directory: {logger.run_dir}")


if __name__ == "__main__":
    main()
