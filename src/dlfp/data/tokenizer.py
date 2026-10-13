"""Character- and word-level tokenizers with JSON persistence.

Both reserve id 0 for ``<pad>`` and id 1 for ``<unk>``. A tokenizer is saved as
JSON beside the run's checkpoints so ``generate.py`` and ``evaluate.py`` can
decode without rebuilding the vocabulary from the corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Union

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


class _VocabTokenizer:
    """Shared vocabulary bookkeeping for the concrete tokenizers."""

    kind = "base"

    def __init__(self, itos: list[str] | None = None) -> None:
        self.itos: list[str] = list(itos) if itos else [PAD_TOKEN, UNK_TOKEN]
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        """Number of tokens including specials."""
        return len(self.itos)

    @property
    def pad_idx(self) -> int:
        """Id of ``<pad>``."""
        return PAD_ID

    def _to_dict(self) -> dict:
        return {"type": self.kind, "itos": self.itos}

    def save(self, path: str | Path) -> None:
        """Write the tokenizer to ``path`` as JSON.

        Args:
            path: Destination file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False)


class CharTokenizer(_VocabTokenizer):
    """Maps each distinct character to an id."""

    kind = "char"

    @classmethod
    def build_from_text(cls, text: str) -> "CharTokenizer":
        """Build a vocabulary from every distinct character of ``text`` (sorted).

        Args:
            text: Corpus.

        Returns:
            The tokenizer.
        """
        return cls([PAD_TOKEN, UNK_TOKEN] + sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        """Characters -> ids (unknown characters map to ``<unk>``).

        Args:
            text: Input string.

        Returns:
            List of ids.
        """
        return [self.stoi.get(ch, UNK_ID) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Ids -> string, dropping padding.

        Args:
            ids: Token ids.

        Returns:
            Decoded text.
        """
        return "".join(self.itos[i] for i in ids if i != PAD_ID)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load a tokenizer saved with :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["itos"])


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


class WordTokenizer(_VocabTokenizer):
    """Lowercasing regex word tokenizer with a frequency-capped vocabulary.

    Args:
        itos: Id -> token list (specials first).
        max_len: If set, :meth:`encode` truncates to this many tokens.
    """

    kind = "word"

    def __init__(self, itos: list[str] | None = None, max_len: int | None = None) -> None:
        super().__init__(itos)
        self.max_len = max_len

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase and split into words and punctuation marks.

        Args:
            text: Input string.

        Returns:
            Token strings.
        """
        return _WORD_RE.findall(text.lower())

    @classmethod
    def build_from_texts(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 20000,
                         max_len: int | None = None) -> "WordTokenizer":
        """Keep the ``max_size`` most frequent tokens occurring ``>= min_freq`` times.

        Args:
            texts: Corpus documents.
            min_freq: Minimum count to enter the vocabulary.
            max_size: Maximum vocabulary size, excluding specials.
            max_len: Truncation length for :meth:`encode`.

        Returns:
            The tokenizer.
        """
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(cls.tokenize(t))
        # Sort by (-count, token) so the vocabulary is deterministic under ties.
        words = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq]
        return cls([PAD_TOKEN, UNK_TOKEN] + words[:max_size], max_len=max_len)

    def encode(self, text: str) -> list[int]:
        """Text -> ids, truncated to ``max_len`` if set.

        Args:
            text: Input string.

        Returns:
            List of ids.
        """
        ids = [self.stoi.get(tok, UNK_ID) for tok in self.tokenize(text)]
        return ids[: self.max_len] if self.max_len else ids

    def decode(self, ids: Iterable[int]) -> str:
        """Ids -> space-joined tokens, dropping padding.

        Args:
            ids: Token ids.

        Returns:
            Decoded text.
        """
        return " ".join(self.itos[i] for i in ids if i != PAD_ID)

    def _to_dict(self) -> dict:
        return {**super()._to_dict(), "max_len": self.max_len}

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        """Load a tokenizer saved with :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["itos"], max_len=data.get("max_len"))


Tokenizer = Union[CharTokenizer, WordTokenizer]


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load whichever tokenizer type is stored at ``path``.

    Args:
        path: JSON file written by ``save``.

    Returns:
        A :class:`CharTokenizer` or :class:`WordTokenizer`.
    """
    with open(path, encoding="utf-8") as f:
        kind = json.load(f).get("type")
    if kind == "char":
        return CharTokenizer.load(path)
    if kind == "word":
        return WordTokenizer.load(path)
    raise ValueError(f"Unknown tokenizer type {kind!r} in {path}")
