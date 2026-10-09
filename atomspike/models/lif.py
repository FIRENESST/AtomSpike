"""Surrogate-gradient LIF used by the spiking policy (no SpikingJelly required)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _sum_to(grad: Tensor, target: Tensor) -> Tensor:
    """Reduce broadcasted grad back to `target` shape."""
    if target.ndim == 0:
        return grad.sum()
    while grad.ndim > target.ndim:
        grad = grad.sum(dim=0)
    for i in range(grad.ndim):
        if int(target.shape[i]) == 1 and int(grad.shape[i]) > 1:
            grad = grad.sum(dim=i, keepdim=True)
    return grad.reshape(target.shape)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, alpha: float) -> Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.alpha = float(alpha)
        return (x >= threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, th = ctx.saved_tensors
        sg = (1.0 / (ctx.alpha * torch.abs(x - th) + 1.0) ** 2)
        gx = grad_output * sg
        gth = _sum_to(-(grad_output * sg), th)
        return gx, gth, None


class LIFCell(nn.Module):
    """Hard-reset LIF. Each unit has a learnable threshold (init = v_th).

    Membrane persistence matches key press/hold/release. Missing checkpoint
    keys keep the init threshold, so old `bc.pt` still fires at the same 1.0.
    """

    v_th_min = 0.15
    v_th_max = 4.0

    def __init__(
        self,
        tau: float = 0.5,
        v_th: float = 1.0,
        surrogate_alpha: float = 2.0,
        thresh_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.tau = float(tau)
        self.surrogate_alpha = float(surrogate_alpha)
        init = float(v_th)
        if thresh_shape is None:
            self.v_th = nn.Parameter(torch.tensor(init))
        else:
            self.v_th = nn.Parameter(torch.full(tuple(int(s) for s in thresh_shape), init))

    def threshold(self) -> Tensor:
        return self.v_th.clamp(self.v_th_min, self.v_th_max)

    def forward(self, x: Tensor, mem: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if mem is None:
            mem = torch.zeros_like(x)
        mem = self.tau * mem + x
        th = self.threshold()
        spike = SurrogateSpike.apply(mem, th, self.surrogate_alpha)
        mem = mem * (1.0 - spike)
        return spike, mem
