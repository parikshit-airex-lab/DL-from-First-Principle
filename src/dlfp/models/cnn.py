"""Convolutional image classifier."""

from __future__ import annotations

import torch
from torch import nn

from dlfp.layers import BatchNorm2d, Conv2d, GlobalAvgPool2d, Linear, MaxPool2d
from dlfp.losses import classification_cross_entropy
from dlfp.metrics import accuracy
from dlfp.models.base import Batch, BaseModel
from dlfp.registry import register_model


@register_model("cnn")
class CNN(BaseModel):
    """Stack of conv blocks, global average pooling and an MLP head.

    Each block is ``Conv2d(padding=k//2) -> [BatchNorm2d] -> ReLU -> MaxPool2d(2)``.
    The head is ``GlobalAvgPool2d -> Linear -> ReLU -> Dropout -> Linear``.

    Global average pooling collapses whatever spatial size remains to one value
    per channel, so the head always sees ``channels[-1]`` features and the model
    accepts any input resolution (28x28 MNIST, 32x32 CIFAR, ...) without
    flatten-size arithmetic.

    Args:
        in_channels: Image channels (1 for MNIST, 3 for CIFAR-10).
        num_classes: Output classes.
        channels: Output channels of each conv block.
        kernel_size: Conv kernel size (odd, so ``k//2`` padding preserves size).
        use_batchnorm: Insert ``BatchNorm2d`` after each conv.
        dropout: Dropout before the final layer.
        fc_hidden: Width of the hidden fully connected layer.
    """

    task = "classification"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: list[int] | tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        dropout: float = 0.25,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes, channels=list(channels),
                         kernel_size=kernel_size, use_batchnorm=use_batchnorm, dropout=dropout,
                         fc_hidden=fc_hidden)
        blocks: list[nn.Module] = []
        prev = in_channels
        for out in channels:
            # The conv bias is redundant before batch norm (its mean is subtracted).
            blocks.append(Conv2d(prev, out, kernel_size, padding=kernel_size // 2, bias=not use_batchnorm))
            if use_batchnorm:
                blocks.append(BatchNorm2d(out))
            blocks.append(nn.ReLU())
            blocks.append(MaxPool2d(2))
            prev = out
        self.features = nn.Sequential(*blocks)
        self.pool = GlobalAvgPool2d()
        self.head = nn.Sequential(
            Linear(prev, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(fc_hidden, num_classes),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return ``(N, num_classes)`` logits for ``batch["inputs"]`` ``(N, C, H, W)``."""
        return self.head(self.pool(self.features(batch["inputs"])))

    def compute_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross-entropy over classes (``mask``/``lengths`` are ignored).

        Args:
            batch: Batch dict with ``inputs`` and ``targets``.

        Returns:
            ``(loss, {"loss", "acc"})``.
        """
        logits = self.forward(batch)
        loss = classification_cross_entropy(logits, batch["targets"])
        return loss, {"loss": loss.item(), "acc": accuracy(logits.detach(), batch["targets"])}
