"""Pre-norm Transformer for language modelling (causal) or sequence classification."""

from __future__ import annotations

import math

import torch
from torch import nn

from dlfp.layers import (
    Embedding,
    LayerNorm,
    LearnedPositionalEmbedding,
    Linear,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    causal_mask,
)
from dlfp.losses import IGNORE_INDEX, classification_cross_entropy, sequence_cross_entropy
from dlfp.metrics import accuracy, perplexity
from dlfp.models.base import Batch, BaseModel, sample_next_token
from dlfp.registry import register_model


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block::

        x = x + dropout(attn(ln1(x)))
        x = x + dropout(ff(ln2(x)))      ff = Linear -> GELU -> dropout -> Linear

    Pre-norm (normalising the *input* of each sub-layer, leaving the residual
    path an identity) keeps gradient magnitudes well-behaved through deep
    stacks, so it trains stably without the learning-rate warmup that the
    original post-norm layout needs.

    Args:
        d_model: Model width.
        n_heads: Attention heads.
        d_ff: Feed-forward hidden width.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = LayerNorm(d_model)
        self.ff = nn.Sequential(Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the block.

        Args:
            x: ``(B, T, d_model)``.
            attn_mask: Optional mask broadcastable to ``(B, H, T, T)``; True = attend.
            key_padding_mask: Optional ``(B, T)``; 1 = real token.

        Returns:
            ``(B, T, d_model)``.
        """
        attn_out, _ = self.attn(self.ln1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        return x + self.dropout(self.ff(self.ln2(x)))


@register_model("transformer")
class TransformerLM(BaseModel):
    """Stack of pre-norm Transformer blocks with an LM or classification head.

    * ``mode="lm"``: causal self-attention (combined with the key-padding mask),
      per-position ``Linear(d_model, vocab_size)`` head, optionally tied to the
      token embedding.
    * ``mode="classification"``: bidirectional attention, mask-aware mean
      pooling ``(x * mask).sum(1) / mask.sum(1)``, ``Linear(d_model, num_classes)``.

    Token embeddings are scaled by ``sqrt(d_model)`` before the positional
    encoding is added.

    Args:
        vocab_size: Vocabulary size.
        d_model: Model width.
        n_heads: Attention heads.
        n_layers: Number of blocks.
        d_ff: Feed-forward hidden width.
        dropout: Dropout rate.
        max_seq_len: Longest supported sequence.
        pos_encoding: ``"sinusoidal"`` or ``"learned"``.
        mode: ``"lm"`` or ``"classification"``.
        num_classes: Classes for classification mode.
        pad_idx: Padding token id.
        tie_weights: Share embedding and LM output weights.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pos_encoding: str = "sinusoidal",
        mode: str = "lm",
        num_classes: int = 2,
        pad_idx: int = 0,
        tie_weights: bool = True,
    ) -> None:
        super().__init__(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         d_ff=d_ff, dropout=dropout, max_seq_len=max_seq_len, pos_encoding=pos_encoding,
                         mode=mode, num_classes=num_classes, pad_idx=pad_idx, tie_weights=tie_weights)
        if mode not in ("lm", "classification"):
            raise ValueError(f"mode must be 'lm' or 'classification', got {mode!r}")
        self.mode = mode
        self.task = "language_modeling" if mode == "lm" else "classification"
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embedding = Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # Re-initialise with std 1/sqrt(d_model): after the sqrt(d_model) scaling
        # the embeddings have unit variance, and when tied to the output layer the
        # initial logits stay O(1) instead of O(sqrt(d_model)).
        with torch.no_grad():
            self.embedding.weight.normal_(0.0, d_model ** -0.5)
            self.embedding.weight[self.embedding.padding_idx].zero_()
        if pos_encoding == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        elif pos_encoding == "learned":
            self.pos = LearnedPositionalEmbedding(max_seq_len, d_model, dropout=dropout)
        else:
            raise ValueError(f"pos_encoding must be 'sinusoidal' or 'learned', got {pos_encoding!r}")
        self.blocks = nn.ModuleList(TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_f = LayerNorm(d_model)
        # With a tied head, the residual stream is dominated by the input token's
        # own embedding, so its logit is e . LN(e * sqrt(d)) ~ sqrt(d) while all
        # others are O(1): the untrained model confidently predicts "next token =
        # current token" (initial loss ~12 instead of ln V ~ 4 for d=256, V=67).
        # Scaling the final hidden state by 1/sqrt(d) before the tied projection
        # (as in T5) makes every initial logit O(1). The scale is a learnable
        # scalar so the model can still sharpen its predictions quickly; a fixed
        # 1/sqrt(d) caps logit growth at the rate the embedding norms grow.
        self.logit_scale: nn.Parameter | None = None
        if mode == "lm":
            self.head = Linear(d_model, vocab_size)
            if tie_weights:
                self.head.weight = self.embedding.weight
                self.logit_scale = nn.Parameter(torch.tensor(d_model ** -0.5))
        else:
            self.head = Linear(d_model, num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return ``(B, T, V)`` (lm) or ``(B, num_classes)`` (classification) logits.

        Args:
            batch: Needs ``inputs`` ``(B, T)``; uses ``mask`` ``(B, T)`` if present.

        Returns:
            Logits.
        """
        inputs = batch["inputs"]
        mask = batch.get("mask")
        x = self.pos(self.embedding(inputs) * math.sqrt(self.d_model))
        attn_mask = causal_mask(inputs.size(1), inputs.device) if self.mode == "lm" else None
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=mask)
        x = self.ln_f(x)
        if self.mode == "lm":
            return self.head(x if self.logit_scale is None else x * self.logit_scale)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            m = mask.to(x.dtype).unsqueeze(-1)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled)

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
        """Autoregressively sample, cropping the context to ``max_seq_len``.

        Args:
            prompt_ids: ``(T,)`` or ``(B, T)`` token ids.
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
        for _ in range(max_new_tokens):
            logits = self.forward({"inputs": ids[:, -self.max_seq_len:]})
            ids = torch.cat([ids, sample_next_token(logits[:, -1], temperature, top_k)], dim=1)
        self.train(was_training)
        return ids
