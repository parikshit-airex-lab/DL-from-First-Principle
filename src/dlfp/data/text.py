"""Text datasets: Tiny Shakespeare (character LM) and IMDB (sentiment classification).

Both download into ``cfg.root`` on first use and are cached there. On a lab
server behind a proxy the download may fail; the error then names the exact
path where the file can be placed manually.
"""

from __future__ import annotations

import random
import re
import tarfile
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset

from dlfp.config import DataConfig
from dlfp.data.base import DatasetBundle
from dlfp.data.tokenizer import CharTokenizer, WordTokenizer
from dlfp.registry import register_dataset

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def _download(url: str, dest: Path, manual_hint: str) -> None:
    """Stream ``url`` to ``dest`` atomically, or raise a clear manual-fallback error.

    Args:
        url: Source URL.
        dest: Destination file.
        manual_hint: Instructions appended to the error message.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download {url} ({type(exc).__name__}: {exc}).\n{manual_hint}\n"
            f"If this machine is behind a proxy, setting HTTPS_PROXY may also fix the download."
        ) from exc


# --------------------------------------------------------------------------- charlm


class CharWindowDataset(Dataset):
    """Non-overlapping ``block_size`` windows over a token-id stream.

    Item ``i`` is ``inputs = ids[s : s+block]``, ``targets = ids[s+1 : s+block+1]``
    with ``s = i * block_size`` (targets are inputs shifted one to the left).

    Args:
        ids: 1-D tensor of token ids.
        block_size: Window length.
    """

    def __init__(self, ids: torch.Tensor, block_size: int) -> None:
        self.ids = ids
        self.block_size = block_size
        self._n = max(0, (len(ids) - 1) // block_size)

    def __len__(self) -> int:
        """Number of windows."""
        return self._n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return window ``i`` as ``{"inputs", "targets"}``."""
        if not 0 <= i < self._n:
            raise IndexError(i)
        s = i * self.block_size
        chunk = self.ids[s: s + self.block_size + 1]
        return {"inputs": chunk[:-1], "targets": chunk[1:]}


def _charlm_corpus_path(cfg: DataConfig) -> Path:
    override = cfg.params.get("corpus_path")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"data.params.corpus_path points to a missing file: {path.resolve()}")
        return path
    path = Path(cfg.root).expanduser() / "tinyshakespeare" / "input.txt"
    if not path.is_file():
        _download(TINY_SHAKESPEARE_URL, path,
                  f"Download it manually and place it at exactly: {path.resolve()}\n"
                  f"(or set data.params.corpus_path to any plain-text file).")
    return path


@register_dataset("charlm")
def build_charlm(cfg: DataConfig) -> DatasetBundle:
    """Tiny Shakespeare character LM with a 90/5/5 split by character position.

    The tokenizer is built on the full text (the character set is tiny and
    fixed, so this leaks nothing meaningful).

    Args:
        cfg: Data config; ``params.block_size`` (default 128), ``params.corpus_path``.

    Returns:
        The dataset bundle.
    """
    text = _charlm_corpus_path(cfg).read_text(encoding="utf-8")
    block_size = int(cfg.params.get("block_size", 128))
    tokenizer = CharTokenizer.build_from_text(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = len(ids)
    a, b = int(0.9 * n), int(0.95 * n)
    meta = {"vocab_size": tokenizer.vocab_size, "pad_idx": tokenizer.pad_idx, "tokenizer": tokenizer,
            "block_size": block_size, "collate": "pad"}
    return DatasetBundle(
        train=CharWindowDataset(ids[:a], block_size),
        val=CharWindowDataset(ids[a:b], block_size),
        test=CharWindowDataset(ids[b:], block_size),
        meta=meta,
    )


# --------------------------------------------------------------------------- imdb


class TextClassificationDataset(Dataset):
    """Raw texts with integer labels, tokenized lazily on access.

    Args:
        texts: Documents.
        labels: Class index per document.
        tokenizer: Tokenizer applied in ``__getitem__`` (truncates to ``max_len``).
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer: WordTokenizer) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Number of documents."""
        return len(self.texts)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return ``{"inputs": (T,), "targets": ()}`` for document ``i``."""
        ids = self.tokenizer.encode(self.texts[i]) or [self.tokenizer.stoi["<unk>"]]
        return {"inputs": torch.tensor(ids, dtype=torch.long),
                "targets": torch.tensor(self.labels[i], dtype=torch.long)}


_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _imdb_dir(cfg: DataConfig) -> Path:
    root = Path(cfg.root).expanduser()
    extracted = root / "aclImdb"
    if (extracted / "train" / "pos").is_dir() and (extracted / "test" / "neg").is_dir():
        return extracted
    archive = root / "aclImdb_v1.tar.gz"
    if not archive.is_file():
        _download(IMDB_URL, archive,
                  f"Download it manually and place the archive at exactly: {archive.resolve()}\n"
                  f"(or extract it so that {extracted.resolve()}/train/pos exists).")
    with tarfile.open(archive, "r:gz") as tar:
        # The "data" filter rejects absolute paths, links out of root, etc. It
        # exists on Python >= 3.12 and on security-patched 3.10/3.11 releases.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, filter="data")
        else:
            tar.extractall(root)
    return extracted


def _read_split(directory: Path) -> tuple[list[str], list[int]]:
    """Read ``pos``/``neg`` reviews (label 1 = positive, 0 = negative), ``<br />`` stripped."""
    texts, labels = [], []
    for label_name, label in (("pos", 1), ("neg", 0)):
        for path in sorted((directory / label_name).glob("*.txt")):
            texts.append(_BR_RE.sub(" ", path.read_text(encoding="utf-8")))
            labels.append(label)
    return texts, labels


def _shuffled(texts: list[str], labels: list[int], seed: int) -> tuple[list[str], list[int]]:
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]


@register_dataset("imdb")
def build_imdb(cfg: DataConfig) -> DatasetBundle:
    """IMDB Large Movie Review sentiment dataset (25k train / 25k test, balanced).

    Files are read in sorted order and then shuffled with a fixed seed, so the
    ``subset_size`` prefix of each split contains both classes.

    Args:
        cfg: Data config; ``params.max_len`` (default 256), ``params.min_freq``
            (default 2), ``params.max_size`` (default 20000).

    Returns:
        The dataset bundle.
    """
    base = _imdb_dir(cfg)
    seed = int(cfg.params.get("split_seed", 0))
    train_texts, train_labels = _shuffled(*_read_split(base / "train"), seed)
    test_texts, test_labels = _shuffled(*_read_split(base / "test"), seed)

    n_val = int(round(len(train_texts) * cfg.val_split))
    val_texts, val_labels = train_texts[:n_val], train_labels[:n_val]
    train_texts, train_labels = train_texts[n_val:], train_labels[n_val:]

    # Build the vocabulary on the training split only (after removing validation),
    # so neither validation nor test words influence which tokens are known.
    tokenizer = WordTokenizer.build_from_texts(
        train_texts,
        min_freq=int(cfg.params.get("min_freq", 2)),
        max_size=int(cfg.params.get("max_size", 20000)),
        max_len=int(cfg.params.get("max_len", 256)),
    )
    meta = {"vocab_size": tokenizer.vocab_size, "num_classes": 2, "pad_idx": tokenizer.pad_idx,
            "tokenizer": tokenizer, "collate": "pad"}
    return DatasetBundle(
        train=TextClassificationDataset(train_texts, train_labels, tokenizer),
        val=TextClassificationDataset(val_texts, val_labels, tokenizer) if n_val else None,
        test=TextClassificationDataset(test_texts, test_labels, tokenizer),
        meta=meta,
    )
