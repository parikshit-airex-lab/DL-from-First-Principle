"""MNIST and CIFAR-10 via ``torchvision.datasets``."""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms

from dlfp.config import DataConfig
from dlfp.data.base import DatasetBundle, split_dataset
from dlfp.registry import register_dataset

# Per-channel training-set statistics:
#   MNIST:    mean 0.1307, std 0.3081
#   CIFAR-10: mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _download_error(name: str, root: Path, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Could not download {name} into {root.resolve()} ({exc}). If this machine is behind a "
        f"proxy, set HTTPS_PROXY, or download the dataset elsewhere and copy torchvision's "
        f"'{name}' folder into {root.resolve()}."
    )


def _build_vision(cfg: DataConfig, dataset_cls: type, name: str, mean: tuple, std: tuple,
                  in_channels: int, image_size: int) -> DatasetBundle:
    root = Path(cfg.root).expanduser()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    try:
        full_train = dataset_cls(root=str(root), train=True, download=True, transform=tf)
        test = dataset_cls(root=str(root), train=False, download=True, transform=tf)
    except Exception as exc:  # network / proxy failures surface as many exception types
        raise _download_error(name, root, exc) from exc
    seed = int(cfg.params.get("split_seed", 0))
    train, val = split_dataset(full_train, cfg.val_split, seed)
    meta = {"num_classes": 10, "in_channels": in_channels, "image_size": image_size, "collate": "image"}
    return DatasetBundle(train=train, val=val, test=test, meta=meta)


@register_dataset("mnist")
def build_mnist(cfg: DataConfig) -> DatasetBundle:
    """MNIST: 60k train (split into train/val by ``cfg.val_split``) and 10k test.

    Args:
        cfg: Data config.

    Returns:
        The dataset bundle.
    """
    return _build_vision(cfg, datasets.MNIST, "MNIST", MNIST_MEAN, MNIST_STD, in_channels=1, image_size=28)


@register_dataset("cifar10")
def build_cifar10(cfg: DataConfig) -> DatasetBundle:
    """CIFAR-10: 50k train (split into train/val by ``cfg.val_split``) and 10k test.

    Args:
        cfg: Data config.

    Returns:
        The dataset bundle.
    """
    return _build_vision(cfg, datasets.CIFAR10, "cifar-10-batches-py", CIFAR10_MEAN, CIFAR10_STD,
                         in_channels=3, image_size=32)
