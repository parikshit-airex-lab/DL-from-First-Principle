"""Numerical-equivalence tests: each layer against its ``torch.nn`` counterpart.

This is the one file allowed to use PyTorch's built-in layer modules, purely as
reference implementations.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlfp.layers import (
    BatchNorm2d,
    Conv2d,
    Embedding,
    LayerNorm,
    Linear,
    LSTMCell,
    MaxPool2d,
    MultiHeadAttention,
    RNNCell,
    causal_mask,
    last_valid_output,
    scaled_dot_product_attention,
    unroll,
)

ATOL = 1e-5


def test_linear_matches_torch() -> None:
    ours = Linear(7, 5)
    ref = nn.Linear(7, 5)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    x = torch.randn(2, 3, 7)  # extra leading dims
    assert torch.allclose(ours(x), ref(x), atol=ATOL)
    no_bias = Linear(7, 5, bias=False)
    assert no_bias.bias is None
    assert torch.allclose(no_bias(x), x @ no_bias.weight.T, atol=ATOL)


def test_linear_init_bounds() -> None:
    layer = Linear(100, 50)
    bound = 1.0 / 100 ** 0.5
    assert layer.weight.abs().max() <= bound + 1e-6
    assert layer.bias.abs().max() <= bound + 1e-6


@pytest.mark.parametrize(
    "in_c,out_c,kernel,stride,padding,hw",
    [
        (3, 4, 3, 1, 0, (8, 8)),
        (3, 4, 3, 1, 1, (8, 8)),
        (2, 5, 3, 2, 1, (9, 7)),       # non-square input
        (1, 3, (3, 5), (2, 1), (1, 2), (10, 6)),  # non-square kernel/stride/pad
        (4, 2, 1, 1, 0, (5, 11)),
        (2, 3, 5, 3, 2, (13, 8)),
    ],
)
def test_conv2d_matches_torch(in_c: int, out_c: int, kernel, stride, padding, hw) -> None:
    ours = Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)
    ref = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    x = torch.randn(2, in_c, *hw)
    out, expected = ours(x), ref(x)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=ATOL)


def test_conv2d_gradients_match_torch() -> None:
    ours = Conv2d(3, 4, 3, stride=2, padding=1)
    ref = nn.Conv2d(3, 4, 3, stride=2, padding=1)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
        ref.bias.copy_(ours.bias)
    x = torch.randn(2, 3, 7, 9)
    ours(x).pow(2).sum().backward()
    ref(x).pow(2).sum().backward()
    assert torch.allclose(ours.weight.grad, ref.weight.grad, atol=1e-4)
    assert torch.allclose(ours.bias.grad, ref.bias.grad, atol=1e-4)


@pytest.mark.parametrize(
    "kernel,stride,padding,hw",
    [(2, None, 0, (8, 8)), (3, 2, 1, (9, 7)), ((2, 3), (1, 2), (1, 1), (6, 10)), (3, 1, 0, (5, 5))],
)
def test_maxpool_matches_torch(kernel, stride, padding, hw) -> None:
    ours = MaxPool2d(kernel, stride=stride, padding=padding)
    ref = nn.MaxPool2d(kernel, stride=stride, padding=padding)
    x = torch.randn(2, 3, *hw) - 5.0  # all-negative region: padding must not win the max
    out, expected = ours(x), ref(x)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=ATOL)


def _copy_bn(ours: BatchNorm2d, ref: nn.BatchNorm2d) -> None:
    with torch.no_grad():
        ours.gamma.copy_(torch.rand(ours.num_features) + 0.5)
        ours.beta.copy_(torch.randn(ours.num_features))
        ref.weight.copy_(ours.gamma)
        ref.bias.copy_(ours.beta)


def test_batchnorm_train_and_eval_match_torch() -> None:
    ours = BatchNorm2d(4)
    ref = nn.BatchNorm2d(4)
    _copy_bn(ours, ref)
    for _ in range(3):  # several steps so running stats accumulate
        x = torch.randn(5, 4, 6, 3) * 3 + 1
        assert torch.allclose(ours(x), ref(x), atol=ATOL)
    assert torch.allclose(ours.running_mean, ref.running_mean, atol=ATOL)
    assert torch.allclose(ours.running_var, ref.running_var, atol=ATOL)  # unbiased update
    assert int(ours.num_batches_tracked) == int(ref.num_batches_tracked) == 3

    ours.eval()
    ref.eval()
    x = torch.randn(2, 4, 5, 5)
    assert torch.allclose(ours(x), ref(x), atol=ATOL)


def test_batchnorm_cumulative_momentum_matches_torch() -> None:
    ours = BatchNorm2d(3, momentum=None)
    ref = nn.BatchNorm2d(3, momentum=None)
    _copy_bn(ours, ref)
    for _ in range(3):
        x = torch.randn(4, 3, 4, 4)
        assert torch.allclose(ours(x), ref(x), atol=ATOL)
    assert torch.allclose(ours.running_var, ref.running_var, atol=ATOL)


@pytest.mark.parametrize("shape,x_shape", [(8, (2, 3, 8)), ((4, 5), (3, 4, 5))])
def test_layernorm_matches_torch(shape, x_shape) -> None:
    ours = LayerNorm(shape)
    ref = nn.LayerNorm(shape)
    with torch.no_grad():
        ours.gamma.copy_(torch.randn_like(ours.gamma))
        ours.beta.copy_(torch.randn_like(ours.beta))
        ref.weight.copy_(ours.gamma)
        ref.bias.copy_(ours.beta)
    x = torch.randn(*x_shape) * 2 + 3
    assert torch.allclose(ours(x), ref(x), atol=ATOL)


def test_embedding_matches_torch_and_pad_stays_zero() -> None:
    ours = Embedding(10, 4, padding_idx=0)
    ref = nn.Embedding(10, 4, padding_idx=0)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
    ids = torch.tensor([[0, 3, 9], [2, 0, 0]])
    assert torch.allclose(ours(ids), ref(ids), atol=ATOL)
    assert torch.equal(ours.weight[0], torch.zeros(4))
    ours(ids).pow(2).sum().backward()
    ref(ids).pow(2).sum().backward()
    assert torch.equal(ours.weight.grad[0], torch.zeros(4))
    assert torch.allclose(ours.weight.grad, ref.weight.grad, atol=ATOL)


def test_rnn_cell_matches_torch() -> None:
    ours = RNNCell(5, 6)
    ref = nn.RNNCell(5, 6)
    with torch.no_grad():
        ref.weight_ih.copy_(ours.ih.weight)
        ref.weight_hh.copy_(ours.hh.weight)
        ref.bias_ih.copy_(ours.ih.bias)
        ref.bias_hh.copy_(ours.hh.bias)
    x, h = torch.randn(3, 5), torch.randn(3, 6)
    assert torch.allclose(ours(x, h), ref(x, h), atol=ATOL)


def test_lstm_cell_matches_torch() -> None:
    ours = LSTMCell(5, 6)
    ref = nn.LSTMCell(5, 6)
    with torch.no_grad():
        # Both use gate order (i, f, g, o) along the 4H dimension.
        ref.weight_ih.copy_(ours.ih.weight)
        ref.weight_hh.copy_(ours.hh.weight)
        ref.bias_ih.copy_(ours.ih.bias)
        ref.bias_hh.copy_(ours.hh.bias)
    x, h, c = torch.randn(3, 5), torch.randn(3, 6), torch.randn(3, 6)
    h_ours, c_ours = ours(x, (h, c))
    h_ref, c_ref = ref(x, (h, c))
    assert torch.allclose(h_ours, h_ref, atol=ATOL)
    assert torch.allclose(c_ours, c_ref, atol=ATOL)


def test_lstm_forget_bias_initialised_to_one() -> None:
    cell = LSTMCell(4, 3)
    total = cell.ih.bias + cell.hh.bias
    assert torch.allclose(total[3:6], torch.ones(3))


@pytest.mark.parametrize("cell_cls", [RNNCell, LSTMCell])
def test_unroll_mask_carries_state_at_padded_steps(cell_cls) -> None:
    cell = cell_cls(4, 5)
    x = torch.randn(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    outputs, final = unroll(cell, x, mask=mask)
    # Row 0: every padded step repeats the output at its last real step (t=2).
    for t in (3, 4, 5):
        assert torch.equal(outputs[0, t], outputs[0, 2])
    unmasked, _ = unroll(cell, x)
    assert torch.allclose(outputs[1], unmasked[1], atol=ATOL)  # full-length row unaffected
    final_h = final[0] if isinstance(final, tuple) else final
    assert torch.equal(final_h[0], outputs[0, 2])


@pytest.mark.parametrize("cell_cls", [RNNCell, LSTMCell])
def test_masked_unroll_matches_individual_sequences(cell_cls) -> None:
    cell = cell_cls(3, 4)
    lengths = torch.tensor([5, 2, 4])
    x = torch.randn(3, 5, 3)
    mask = (torch.arange(5).unsqueeze(0) < lengths.unsqueeze(1)).float()
    outputs, final = unroll(cell, x, mask=mask)
    last = last_valid_output(outputs, lengths)
    for i, n in enumerate(lengths.tolist()):
        single_out, single_final = unroll(cell, x[i:i + 1, :n])
        assert torch.allclose(outputs[i, :n], single_out[0], atol=ATOL)
        assert torch.allclose(last[i], single_out[0, -1], atol=ATOL)
        if isinstance(final, tuple):
            for comp, single_comp in zip(final, single_final):
                assert torch.allclose(comp[i], single_comp[0], atol=ATOL)
        else:
            assert torch.allclose(final[i], single_final[0], atol=ATOL)


def test_sdpa_causal_mask_blocks_future() -> None:
    q, k, v = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    out, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
    future = ~torch.tril(torch.ones(6, 6, dtype=torch.bool))
    assert torch.all(weights[..., future] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 3, 6), atol=ATOL)
    # First query attends only to the first key, so its output is v[..., 0, :].
    assert torch.allclose(out[..., 0, :], v[..., 0, :], atol=ATOL)


def test_sdpa_matches_reference_formula() -> None:
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 3)
    out, _ = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-2, -1) / 8 ** 0.5, dim=-1) @ v
    assert torch.allclose(out, expected, atol=ATOL)


def test_mha_matches_torch() -> None:
    d, h = 8, 2
    ours = MultiHeadAttention(d, h)
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias]))
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(ours.out_proj.bias)
    x = torch.randn(3, 5, d)
    pad = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    out, _ = ours(x, attn_mask=causal_mask(5), key_padding_mask=pad)
    expected, _ = ref(x, x, x, attn_mask=~causal_mask(5)[0, 0], key_padding_mask=pad == 0)
    assert torch.allclose(out, expected, atol=ATOL)


def test_mha_key_padding_mask_ignores_padded_keys() -> None:
    mha = MultiHeadAttention(8, 2).eval()
    x = torch.randn(2, 5, 8)
    pad = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out, weights = mha(x, key_padding_mask=pad)
    assert torch.all(weights[0, :, :, 3:] == 0)
    # Changing the padded positions' content must not change outputs at real positions.
    x2 = x.clone()
    x2[0, 3:] = torch.randn(2, 8) * 100
    out2, _ = mha(x2, key_padding_mask=pad)
    assert torch.allclose(out[0, :3], out2[0, :3], atol=ATOL)
