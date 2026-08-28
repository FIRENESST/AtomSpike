"""Surrogate-gradient LIF used by the spiking policy (no SpikingJelly required)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: float, alpha: float) -> Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.alpha = alpha
        return (x >= threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        th = ctx.threshold
        sg = (1.0 / (alpha * torch.abs(x - th) + 1.0) ** 2)
        return grad_output * sg, None, None


class LIFCell(nn.Module):
    """Hard-reset LIF. Membrane persistence matches key press/hold/release."""

    def __init__(self, tau: float = 0.5, v_th: float = 1.0, surrogate_alpha: float = 2.0):
        super().__init__()
        self.tau = tau
        self.v_th = v_th
        self.surrogate_alpha = surrogate_alpha

    def forward(self, x: Tensor, mem: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.tau * mem + x
        spike = SurrogateSpike.apply(mem, self.v_th, self.surrogate_alpha)
        mem = mem * (1.0 - spike)
        return spike, mem
