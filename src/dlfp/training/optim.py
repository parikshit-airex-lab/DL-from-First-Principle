"""Optimizers with explicit update rules, built on ``torch.optim.Optimizer``.

``torch.optim.Optimizer`` is used only as a base class (parameter groups,
``zero_grad``, ``state_dict``); the update rules themselves are written out.
PyTorch's own SGD and Adam are also registered (``torch_sgd``/``torch_adam``) as
references for the equivalence test.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Iterable

import torch
from torch import nn

from dlfp.registry import build_optimizer, register_optimizer


@register_optimizer("sgd")
class SGD(torch.optim.Optimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum.

    With ``g = grad + weight_decay * p``::

        b_1 = g;   b_t = momentum * b_{t-1} + g          (momentum buffer)
        d   = g + momentum * b_t   if nesterov else  b_t  (or g without momentum)
        p   = p - lr * d

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: L2 penalty added to the gradient.
        nesterov: Use Nesterov momentum.
    """

    def __init__(self, params: Iterable[Any], lr: float, momentum: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Apply one update to every parameter that has a gradient.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The closure's loss, if given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd, nesterov = group["lr"], group["momentum"], group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if wd != 0:
                    d = d.add(p, alpha=wd)
                if mom != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = d.clone().detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(mom).add_(d)
                    d = d.add(buf, alpha=mom) if nesterov else buf
                p.add_(d, alpha=-lr)
        return loss


@register_optimizer("adam")
class Adam(torch.optim.Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction.

    With ``g = grad + weight_decay * p`` (L2 penalty, as in ``torch.optim.Adam``)::

        m_t = beta1 * m_{t-1} + (1 - beta1) * g
        v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)

    The last line is evaluated as
    ``p -= (lr / (1 - beta1^t)) * m_t / (sqrt(v_t) / sqrt(1 - beta2^t) + eps)``,
    which is algebraically identical and matches PyTorch's operation order.

    Args:
        params: Parameters or parameter groups.
        lr: Learning rate.
        betas: ``(beta1, beta2)`` decay rates for the moment estimates.
        eps: Denominator stabiliser.
        weight_decay: L2 penalty added to the gradient.
    """

    def __init__(self, params: Iterable[Any], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Apply one Adam update to every parameter that has a gradient.

        Args:
            closure: Optional closure re-evaluating the loss.

        Returns:
            The closure's loss, if given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias_correction1)
        return loss


register_optimizer("torch_sgd")(torch.optim.SGD)
register_optimizer("torch_adam")(torch.optim.Adam)


def build_optimizer_from_config(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Build ``cfg.train.optimizer`` over the model's trainable parameters.

    Passes whichever of ``lr``, ``weight_decay``, ``betas`` and ``momentum`` the
    optimizer's constructor accepts, so adding an optimizer needs no change here.

    Args:
        model: Model whose parameters are optimised.
        cfg: Full :class:`~dlfp.config.Config`.

    Returns:
        The optimizer.
    """
    from dlfp.registry import OPTIMIZERS

    tc = cfg.train
    candidates = {"lr": tc.lr, "weight_decay": tc.weight_decay, "betas": tuple(tc.betas), "momentum": tc.momentum}
    cls = OPTIMIZERS.get(tc.optimizer.lower())
    accepted = set(inspect.signature(cls.__init__).parameters) if cls is not None else set(candidates)
    kwargs = {k: v for k, v in candidates.items() if k in accepted}
    params = [p for p in model.parameters() if p.requires_grad]
    return build_optimizer(tc.optimizer, params, **kwargs)
