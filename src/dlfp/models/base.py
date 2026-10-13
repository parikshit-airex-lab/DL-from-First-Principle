"""The model interface the trainer depends on, plus the shared recurrent body.

Every model consumes a *batch dict* (``inputs``, ``targets`` and, for padded
sequences, ``lengths`` and ``mask``) and implements two methods:

* ``forward(batch) -> logits``
* ``compute_loss(batch) -> (loss, logs)``

Why ``compute_loss`` takes the whole batch rather than ``(pred, target)``:
sequence models need ``mask``/``lengths`` to exclude padding from the loss
average, a language model scores ``(B, T, V)`` logits per token while a
classifier scores ``(B, C)`` logits per example, and only the model knows how
its own outputs should be scored. Pushing that knowledge into the model is what
keeps the trainer free of model-type conditionals: the trainer calls
``compute_loss`` and never asks what kind of model it is holding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import nn

from dlfp.layers import Embedding, Linear, last_valid_output, unroll
from dlfp.losses import IGNORE_INDEX, classification_cross_entropy, sequence_cross_entropy
from dlfp.metrics import accuracy, perplexity

Batch = dict[str, torch.Tensor]


class BaseModel(nn.Module, ABC):
    """Abstract base class for every model driven by the trainer.

    Attributes:
        task: ``"classification"`` or ``"language_modeling"``.
        hparams: The constructor parameters, kept for ``describe()``.
    """

    task: str = "classification"

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.hparams: dict[str, Any] = dict(params)

    @abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        """Return raw logits."""

    @abstractmethod
    def compute_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (scalar loss with grad, dict of float metrics for logging)."""

    def predict(self, batch: Batch) -> torch.Tensor:
        """Return argmax predictions over the last (class/vocab) dimension.

        Args:
            batch: Batch dict.

        Returns:
            Integer predictions with the logits' shape minus the last dim.
        """
        return self.forward(batch).argmax(dim=-1)

    def batch_size(self, batch: Batch) -> int:
        """Number of examples in ``batch`` (weight for running metric means).

        Args:
            batch: Batch dict.

        Returns:
            The leading dimension of ``batch["inputs"]``.
        """
        return int(batch["inputs"].size(0))

    def finalize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Post-process epoch-averaged metrics (default: unchanged).

        Hook for metrics that are not linear in the per-batch values, e.g.
        perplexity must be ``exp(mean loss)``, not the mean of per-batch ``exp``.

        Args:
            metrics: Weighted epoch means of the per-batch logs.

        Returns:
            Final metrics.
        """
        return metrics

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters (shared/tied parameters counted once).

        Args:
            trainable_only: Count only parameters with ``requires_grad``.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def describe(self) -> str:
        """One-paragraph human-readable summary of the model."""
        hp = ", ".join(f"{k}={v}" for k, v in self.hparams.items())
        return (f"{type(self).__name__} (task={self.task}) | "
                f"{self.num_parameters():,} trainable parameters | {hp}")


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Sample one token id per row from ``(B, V)`` logits.

    Args:
        logits: ``(B, V)`` next-token logits.
        temperature: Softmax temperature (> 0); lower is greedier.
        top_k: If set, sample only among the ``top_k`` most likely tokens.

    Returns:
        ``(B, 1)`` sampled ids.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class SequenceModel(BaseModel):
    """Shared body for recurrent models: embed -> stacked recurrent layers -> head.

    ``RNNModel`` and ``LSTMModel`` differ only in the ``cell_factory`` they pass.

    * ``mode="lm"``: per-timestep ``Linear(hidden_size, vocab_size)``, logits
      ``(B, T, V)``, loss is mask-aware token cross-entropy.
    * ``mode="classification"``: take each row's output at its last real step
      (``lengths - 1``), then ``Linear(hidden_size, num_classes)``.

    Args:
        cell_factory: ``(input_size, hidden_size) -> cell``.
        vocab_size: Vocabulary size.
        embed_dim: Embedding size.
        hidden_size: Recurrent hidden size.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout after the embedding and between layers.
        mode: ``"lm"`` or ``"classification"``.
        num_classes: Classes for classification mode.
        pad_idx: Padding token id.
        tie_weights: Share the embedding matrix with the LM output layer
            (requires ``embed_dim == hidden_size``).
    """

    def __init__(
        self,
        cell_factory: Callable[[int, int], nn.Module],
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        mode: str = "lm",
        num_classes: int = 2,
        pad_idx: int = 0,
        tie_weights: bool = False,
    ) -> None:
        super().__init__(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size,
                         num_layers=num_layers, dropout=dropout, mode=mode, num_classes=num_classes,
                         pad_idx=pad_idx, tie_weights=tie_weights)
        if mode not in ("lm", "classification"):
            raise ValueError(f"mode must be 'lm' or 'classification', got {mode!r}")
        self.mode = mode
        self.task = "language_modeling" if mode == "lm" else "classification"
        self.pad_idx = pad_idx
        self.embedding = Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList(
            cell_factory(embed_dim if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        )
        if mode == "lm":
            self.head = Linear(hidden_size, vocab_size)
            if tie_weights:
                assert embed_dim == hidden_size, "tie_weights requires embed_dim == hidden_size"
                self.head.weight = self.embedding.weight
        else:
            self.head = Linear(hidden_size, num_classes)

    def _encode(self, inputs: torch.Tensor, mask: torch.Tensor | None = None,
                states: list | None = None) -> tuple[torch.Tensor, list]:
        """Embed and run every recurrent layer.

        Args:
            inputs: ``(B, T)`` token ids.
            mask: Optional ``(B, T)`` 1 = real.
            states: Optional per-layer initial states.

        Returns:
            ``(top-layer outputs (B, T, H), per-layer final states)``.
        """
        x = self.dropout(self.embedding(inputs))
        new_states = []
        for i, cell in enumerate(self.cells):
            x, state = unroll(cell, x, None if states is None else states[i], mask)
            new_states.append(state)
            x = self.dropout(x)
        return x, new_states

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return ``(B, T, V)`` (lm) or ``(B, num_classes)`` (classification) logits.

        Args:
            batch: Needs ``inputs``; uses ``mask`` if present; classification
                uses ``lengths`` (or derives them from ``mask``).

        Returns:
            Logits.
        """
        inputs = batch["inputs"]
        mask = batch.get("mask")
        outputs, _ = self._encode(inputs, mask)
        if self.mode == "lm":
            return self.head(outputs)
        lengths = batch.get("lengths")
        if lengths is None:
            lengths = mask.sum(1).long() if mask is not None else torch.full(
                (inputs.size(0),), inputs.size(1), device=inputs.device)
        return self.head(last_valid_output(outputs, lengths))

    def compute_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Masked token cross-entropy (lm) or example cross-entropy (classification).

        Args:
            batch: Batch dict with ``targets``.

        Returns:
            ``(loss, {"loss", "ppl"})`` or ``(loss, {"loss", "acc"})``.
        """
        logits = self.forward(batch)
        targets = batch["targets"]
        if self.mode == "lm":
            loss = sequence_cross_entropy(logits, targets, batch.get("mask"), ignore_index=IGNORE_INDEX)
            return loss, {"loss": loss.item(), "ppl": perplexity(loss.item())}
        loss = classification_cross_entropy(logits, targets)
        return loss, {"loss": loss.item(), "acc": accuracy(logits.detach(), targets)}

    def finalize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Recompute perplexity as ``exp(mean loss)`` for language modelling."""
        if self.mode == "lm" and "loss" in metrics:
            metrics = {**metrics, "ppl": perplexity(metrics["loss"])}
        return metrics

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Autoregressively sample continuations of ``prompt_ids``.

        The prompt is run once; each new token then advances the carried
        recurrent state by a single step.

        Args:
            prompt_ids: ``(T,)`` or ``(B, T)`` token ids (T >= 1).
            max_new_tokens: Number of tokens to sample.
            temperature: Sampling temperature.
            top_k: Optional top-k filter.

        Returns:
            ``(B, T + max_new_tokens)`` ids (prompt included).

        Raises:
            RuntimeError: If the model is not in ``"lm"`` mode.
        """
        if self.mode != "lm":
            raise RuntimeError(f"generate() requires mode='lm', but this model has mode={self.mode!r}")
        was_training = self.training
        self.eval()
        ids = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        outputs, states = self._encode(ids)
        generated = [ids]
        for _ in range(max_new_tokens):
            next_id = sample_next_token(self.head(outputs[:, -1]), temperature, top_k)
            generated.append(next_id)
            outputs, states = self._encode(next_id, states=states)
        self.train(was_training)
        return torch.cat(generated, dim=1)
