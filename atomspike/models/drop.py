"""Stochastic depth. Training-only; eval is a no-op so 30Hz cost stays put."""

from __future__ import annotations

from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(max(0.0, p))

    def forward(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p <= 0.0 or x.size(0) == 0:
            return x
        keep = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep
