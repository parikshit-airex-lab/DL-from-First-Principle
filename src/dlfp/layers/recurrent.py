"""Recurrent cells and a mask-aware unroll loop, built on :class:`Linear`.

A *cell* computes one time step, ``state_t = cell(x_t, state_{t-1})``. The
:func:`unroll` loop drives any cell over a ``(B, T, input_size)`` sequence. The
state may be a single tensor (RNN) or a tuple of tensors (LSTM ``(h, c)``); by
convention the first element of a tuple state is the output ``h``.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import nn

from .linear import Linear

State = Union[torch.Tensor, tuple[torch.Tensor, ...]]


def _init_recurrent(module: nn.Module, hidden_size: int) -> None:
    """PyTorch's recurrent init: every weight and bias ``~ U(-1/sqrt(H), 1/sqrt(H))``."""
    bound = 1.0 / math.sqrt(hidden_size)
    with torch.no_grad():
        for p in module.parameters():
            p.uniform_(-bound, bound)


class RNNCell(nn.Module):
    """Elman RNN cell: ``h' = act(W_ih x + b_ih + W_hh h + b_hh)``.

    Args:
        input_size: Features per input step.
        hidden_size: Hidden-state size.
        nonlinearity: ``"tanh"`` or ``"relu"``.
    """

    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh") -> None:
        super().__init__()
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError(f"nonlinearity must be 'tanh' or 'relu', got {nonlinearity!r}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ih = Linear(input_size, hidden_size)
        self.hh = Linear(hidden_size, hidden_size)
        _init_recurrent(self, hidden_size)

    def init_state(self, batch_size: int, device: torch.device | None = None,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
        """Zero hidden state of shape ``(batch_size, hidden_size)``.

        Args:
            batch_size: Batch size.
            device: Device for the state.
            dtype: Dtype for the state.

        Returns:
            The zero state ``h``.
        """
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """One step.

        Args:
            x: Input ``(B, input_size)``.
            h: Previous hidden ``(B, hidden_size)``.

        Returns:
            New hidden state ``(B, hidden_size)``.
        """
        pre = self.ih(x) + self.hh(h)
        return torch.tanh(pre) if self.nonlinearity == "tanh" else torch.relu(pre)


class LSTMCell(nn.Module):
    """Long short-term memory cell.

    The two affine maps produce ``4H`` pre-activations, split in PyTorch's gate
    order ``i, f, g, o``::

        i = sigmoid(.)   input gate        f = sigmoid(.)   forget gate
        g = tanh(.)      candidate cell    o = sigmoid(.)   output gate
        c' = f * c + i * g
        h' = o * tanh(c')

    Args:
        input_size: Features per input step.
        hidden_size: Hidden/cell-state size.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = Linear(input_size, 4 * hidden_size)
        self.hh = Linear(hidden_size, 4 * hidden_size)
        _init_recurrent(self, hidden_size)
        # Forget-gate bias = 1.0 (Gers et al. 2000; Jozefowicz et al. 2015). With a
        # ~0 bias, f starts near sigmoid(0) = 0.5 and the cell state halves every
        # step, so gradients vanish over long ranges before the model can learn to
        # remember. Starting at f ~ sigmoid(1) ~ 0.73 lets memory persist from the
        # outset. The ih and hh biases are summed, so set one to 1 and the other to 0.
        with torch.no_grad():
            h = hidden_size
            self.ih.bias[h:2 * h].fill_(1.0)
            self.hh.bias[h:2 * h].fill_(0.0)

    def init_state(self, batch_size: int, device: torch.device | None = None,
                   dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero ``(h, c)``, each of shape ``(batch_size, hidden_size)``.

        Args:
            batch_size: Batch size.
            device: Device for the state.
            dtype: Dtype for the state.

        Returns:
            The zero state ``(h, c)``.
        """
        z = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return (z, z.clone())

    def forward(self, x: torch.Tensor,
                state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """One step.

        Args:
            x: Input ``(B, input_size)``.
            state: Previous ``(h, c)``.

        Returns:
            New ``(h, c)``.
        """
        h, c = state
        gates = self.ih(x) + self.hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return (h_new, c_new)


def _output_of(state: State) -> torch.Tensor:
    """The output ``h`` of a state (the first element of a tuple state)."""
    return state[0] if isinstance(state, tuple) else state


def unroll(
    cell: nn.Module,
    x: torch.Tensor,
    state: State | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, State]:
    """Run ``cell`` over every time step of ``x``.

    With ``mask`` (``1`` = real token, ``0`` = padding), padded steps carry the
    previous state forward unchanged, for every state component::

        s_t = m_t * cell(x_t, s_{t-1}) + (1 - m_t) * s_{t-1}

    so the final state of a right-padded sequence equals its state at its true
    last token.

    Args:
        cell: A cell exposing ``init_state`` and ``forward(x_t, state)``.
        x: Inputs ``(B, T, input_size)``.
        state: Initial state, or ``None`` for zeros.
        mask: Optional ``(B, T)`` float/bool mask.

    Returns:
        ``(outputs, final_state)`` with outputs ``(B, T, hidden_size)``.
    """
    b, t, _ = x.shape
    if state is None:
        state = cell.init_state(b, device=x.device, dtype=x.dtype)
    if mask is not None:
        mask = mask.to(dtype=x.dtype)
    outputs = []
    for step in range(t):
        new_state = cell(x[:, step], state)
        if mask is not None:
            m = mask[:, step].unsqueeze(-1)
            if isinstance(new_state, tuple):
                new_state = tuple(m * n + (1 - m) * p for n, p in zip(new_state, state))
            else:
                new_state = m * new_state + (1 - m) * state
        state = new_state
        outputs.append(_output_of(state))
    return torch.stack(outputs, dim=1), state


def last_valid_output(outputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Select each row's output at its last real time step, ``lengths - 1``.

    Args:
        outputs: ``(B, T, H)``.
        lengths: ``(B,)`` true sequence lengths (>= 1).

    Returns:
        ``(B, H)``.
    """
    idx = (lengths.to(outputs.device).long() - 1).clamp(min=0)
    idx = idx.view(-1, 1, 1).expand(-1, 1, outputs.size(-1))
    return outputs.gather(1, idx).squeeze(1)
