"""Behavioural tests for every model behind the common BaseModel interface."""

from __future__ import annotations

import math

import pytest
import torch

import dlfp.models  # noqa: F401  (registers models)
from dlfp.losses import IGNORE_INDEX
from dlfp.models.base import BaseModel
from dlfp.registry import build_model

VOCAB, PAD = 13, 0
B, T = 4, 7


def _lm_batch(lengths: list[int] | None = None) -> dict[str, torch.Tensor]:
    """Random LM batch, right-padded to ``T`` according to ``lengths``."""
    lengths = lengths or [T] * B
    inputs = torch.randint(1, VOCAB, (B, T))
    targets = torch.randint(1, VOCAB, (B, T))
    mask = (torch.arange(T).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)).float()
    inputs = inputs.masked_fill(mask == 0, PAD)
    targets = targets.masked_fill(mask == 0, IGNORE_INDEX)
    return {"inputs": inputs, "targets": targets, "mask": mask, "lengths": torch.tensor(lengths)}


def _cls_batch(lengths: list[int]) -> dict[str, torch.Tensor]:
    batch = _lm_batch(lengths)
    batch["targets"] = torch.tensor([0, 1, 1, 0])
    return batch


def _image_batch() -> dict[str, torch.Tensor]:
    return {"inputs": torch.randn(B, 1, 12, 10), "targets": torch.tensor([0, 1, 2, 1])}


SEQ_COMMON = {"vocab_size": VOCAB, "pad_idx": PAD, "dropout": 0.0}
CASES = {
    "cnn": (dict(in_channels=1, num_classes=3, channels=[8, 16], fc_hidden=16, dropout=0.0), _image_batch, (B, 3)),
    "rnn_lm": (dict(name="rnn", embed_dim=16, hidden_size=32, num_layers=2, mode="lm", **SEQ_COMMON),
               lambda: _lm_batch([7, 5, 3, 7]), (B, T, VOCAB)),
    "lstm_lm": (dict(name="lstm", embed_dim=16, hidden_size=32, num_layers=2, mode="lm", **SEQ_COMMON),
                lambda: _lm_batch([7, 5, 3, 7]), (B, T, VOCAB)),
    "lstm_cls": (dict(name="lstm", embed_dim=16, hidden_size=32, mode="classification", num_classes=2,
                      **SEQ_COMMON), lambda: _cls_batch([7, 2, 5, 4]), (B, 2)),
    "rnn_cls": (dict(name="rnn", embed_dim=16, hidden_size=32, mode="classification", num_classes=2,
                     **SEQ_COMMON), lambda: _cls_batch([7, 2, 5, 4]), (B, 2)),
    "transformer_lm": (dict(name="transformer", d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=32,
                            mode="lm", **SEQ_COMMON), lambda: _lm_batch([7, 5, 3, 7]), (B, T, VOCAB)),
    "transformer_lm_learned": (dict(name="transformer", d_model=32, n_heads=4, n_layers=1, d_ff=64,
                                    max_seq_len=32, pos_encoding="learned", tie_weights=False, mode="lm",
                                    **SEQ_COMMON), lambda: _lm_batch(), (B, T, VOCAB)),
    "transformer_cls": (dict(name="transformer", d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=32,
                             mode="classification", num_classes=2, **SEQ_COMMON),
                        lambda: _cls_batch([7, 2, 5, 4]), (B, 2)),
}


def _build(case: str) -> tuple[BaseModel, dict[str, torch.Tensor], tuple[int, ...]]:
    params, make_batch, shape = CASES[case]
    params = dict(params)
    name = params.pop("name", case)
    torch.manual_seed(0)
    return build_model(name, **params), make_batch(), shape


@pytest.mark.parametrize("case", list(CASES))
def test_forward_shape(case: str) -> None:
    model, batch, shape = _build(case)
    assert model(batch).shape == shape


@pytest.mark.parametrize("case", list(CASES))
def test_compute_loss_contract_and_gradients(case: str) -> None:
    model, batch, _ = _build(case)
    loss, logs = model.compute_loss(batch)
    assert loss.dim() == 0 and loss.requires_grad
    assert logs and all(isinstance(v, float) for v in logs.values())
    assert "loss" in logs
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
        if "running" not in name:
            assert p.grad.abs().sum() > 0, f"all-zero gradient for {name}"


@pytest.mark.parametrize("case", list(CASES))
def test_overfits_single_batch(case: str) -> None:
    model, batch, _ = _build(case)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(200):
        loss, _ = model.compute_loss(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    final, _ = model.compute_loss(batch)
    assert final.item() < 0.1, f"{case}: loss after 200 steps = {final.item():.4f}"


@pytest.mark.parametrize("case", [c for c in CASES if "_lm" in c])
def test_initial_lm_loss_near_uniform(case: str) -> None:
    """An untrained LM must start near ln(V), not confidently wrong.

    Regression test: a tied Transformer head without output rescaling started at
    loss ~11 with V=67 because each token's logit for itself was ~sqrt(d_model).
    """
    params = dict(CASES[case][0])
    name = params.pop("name")
    vocab = 67
    params["vocab_size"] = vocab
    if name == "transformer":  # full width, where the effect is largest
        params.update(d_model=256, n_heads=4)
    torch.manual_seed(0)
    model = build_model(name, **params).eval()
    inputs = torch.randint(1, vocab, (4, 16))
    with torch.no_grad():
        loss, _ = model.compute_loss({"inputs": inputs, "targets": torch.randint(1, vocab, (4, 16))})
    assert abs(loss.item() - math.log(vocab)) < 1.0, f"{case}: initial loss {loss.item():.2f}"


def _append_padding(batch: dict[str, torch.Tensor], extra: int) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["inputs"] = torch.cat([batch["inputs"], torch.full((B, extra), PAD)], dim=1)
    out["mask"] = torch.cat([batch["mask"], torch.zeros(B, extra)], dim=1)
    if batch["targets"].dim() == 2:
        out["targets"] = torch.cat([batch["targets"], torch.full((B, extra), IGNORE_INDEX)], dim=1)
    return out


@pytest.mark.parametrize("case", [c for c in CASES if c != "cnn"])
def test_padding_does_not_change_loss(case: str) -> None:
    model, batch, _ = _build(case)
    model.eval()
    with torch.no_grad():
        loss, _ = model.compute_loss(batch)
        padded_loss, _ = model.compute_loss(_append_padding(batch, 5))
    assert torch.allclose(loss, padded_loss, atol=1e-4)


def test_cnn_accepts_any_resolution() -> None:
    model, _, _ = _build("cnn")
    for hw in [(28, 28), (32, 32), (17, 23)]:
        assert model({"inputs": torch.randn(2, 1, *hw)}).shape == (2, 3)


@pytest.mark.parametrize("case", ["rnn_lm", "lstm_lm", "transformer_lm"])
def test_generate(case: str) -> None:
    model, _, _ = _build(case)
    out = model.generate(torch.tensor([1, 2, 3]), max_new_tokens=5, temperature=0.8, top_k=4)
    assert out.shape == (1, 8)
    assert torch.equal(out[0, :3], torch.tensor([1, 2, 3]))
    assert int(out.max()) < VOCAB


def test_recurrent_generate_matches_full_forward() -> None:
    """Stepping the carried state must agree with re-running the whole sequence."""
    model, _, _ = _build("lstm_lm")
    model.eval()
    torch.manual_seed(1)
    out = model.generate(torch.tensor([1, 2, 3]), max_new_tokens=4, temperature=1e-4)
    logits = model({"inputs": out[:, :-1]})
    assert torch.equal(logits.argmax(-1)[0, 2:], out[0, 3:])


@pytest.mark.parametrize("case", ["lstm_cls", "transformer_cls"])
def test_generate_rejects_classification_mode(case: str) -> None:
    model, _, _ = _build(case)
    with pytest.raises(RuntimeError, match="mode"):
        model.generate(torch.tensor([1, 2]), max_new_tokens=2)


def test_tied_weights_share_storage() -> None:
    model = build_model("transformer", vocab_size=VOCAB, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    assert model.head.weight is model.embedding.weight
    lstm = build_model("lstm", vocab_size=VOCAB, embed_dim=16, hidden_size=16, tie_weights=True)
    assert lstm.head.weight is lstm.embedding.weight
