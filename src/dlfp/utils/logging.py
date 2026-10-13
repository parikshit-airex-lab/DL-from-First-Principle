"""Run-directory management and text/CSV logging."""

from __future__ import annotations

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dlfp.config import Config, save_config

# Fixed leading columns; metric columns are appended (sorted) after these and
# frozen on first write so every row of metrics.csv lines up.
_BASE_COLUMNS = ["epoch", "step", "split"]


class RunLogger:
    """Owns ``runs/<run_name>_<timestamp>/`` and everything written into it.

    Layout::

        <run_dir>/config.yaml   resolved config, written immediately
        <run_dir>/log.txt       copy of everything logged to stdout
        <run_dir>/metrics.csv   one row per (epoch, split)
        <run_dir>/ckpt/         checkpoints and the saved tokenizer

    Args:
        output_dir: Parent directory for runs.
        run_name: Run-name prefix.
        config: Resolved config to save into the run directory (optional).
        run_dir: Use this exact directory instead of creating a timestamped one.
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str,
        config: Config | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        if run_dir is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = Path(output_dir) / f"{run_name}_{stamp}"
        self._run_dir = Path(run_dir).resolve()
        self._ckpt_dir = self._run_dir / "ckpt"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            save_config(config, self._run_dir / "config.yaml")

        self._logger = logging.getLogger(f"dlfp.run.{self._run_dir}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout),
                        logging.FileHandler(self._run_dir / "log.txt", encoding="utf-8")):
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)

        self._metrics_path = self._run_dir / "metrics.csv"
        self._columns: list[str] | None = None
        if self._metrics_path.is_file():  # resuming into an existing run dir
            with open(self._metrics_path, newline="", encoding="utf-8") as f:
                self._columns = next(csv.reader(f), None)

    @property
    def run_dir(self) -> Path:
        """Directory holding this run's outputs."""
        return self._run_dir

    @property
    def ckpt_dir(self) -> Path:
        """Directory holding this run's checkpoints."""
        return self._ckpt_dir

    def info(self, message: str) -> None:
        """Log ``message`` to stdout and ``log.txt``.

        Args:
            message: Text to log.
        """
        self._logger.info(message)

    def log_metrics(self, epoch: int, step: int, metrics: dict[str, Any], split: str) -> None:
        """Append one row to ``metrics.csv`` and echo it to the log.

        The column set is fixed by the first write; later rows fill missing
        columns with blanks and ignore unseen keys (which are still logged).

        Args:
            epoch: Epoch number (1-based).
            step: Global optimizer step.
            metrics: Metric name -> value.
            split: ``"train"``, ``"val"`` or ``"test"``.
        """
        if self._columns is None:
            self._columns = _BASE_COLUMNS + sorted(metrics)
            with open(self._metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._columns)
        row = {"epoch": epoch, "step": step, "split": split, **metrics}
        with open(self._metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([_fmt(row.get(c, "")) for c in self._columns])
        pretty = " ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
        self.info(f"[{split}] epoch={epoch} step={step} {pretty}")

    def close(self) -> None:
        """Detach and close this logger's handlers."""
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)


def _fmt(value: Any) -> str:
    """Format floats compactly for logs and CSV."""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
