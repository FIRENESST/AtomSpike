"""Activations that can switch into PMSM spikes without changing the module tree.

T6 used to replace nn.ReLU with a different class, then save state_dict and
reload into a fresh ANN — the conversion evaporated. Keeping one Act module
and storing spike_mode/threshold as buffers makes conversion survive load.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Act(nn.Module):
    def __init__(self, kind: str = "relu"):
        super().__init__()
        if kind not in {"relu", "gelu"}:
            raise ValueError(f"unsupported act kind: {kind}")
        self.kind = kind
        self.register_buffer("threshold", torch.zeros(()))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("spike_mode", torch.zeros((), dtype=torch.uint8))

    def set_spike(self, enabled: bool, threshold: float | None = None, scale: float | None = None) -> None:
        self.spike_mode.fill_(1 if enabled else 0)
        if threshold is not None:
            self.threshold.fill_(float(threshold))
        if scale is not None:
            self.scale.fill_(float(scale))

    @property
    def is_spike(self) -> bool:
        return int(self.spike_mode.item()) != 0

    def extra_repr(self) -> str:
        return f"kind={self.kind}, spike={self.is_spike}, th={float(self.threshold):.4f}"

    def forward(self, x: Tensor) -> Tensor:
        if self.is_spike:
            return (x > self.threshold).to(dtype=x.dtype) * self.scale
        if self.kind == "gelu":
            return F.gelu(x)
        return F.relu(x)


def iter_acts(module: nn.Module):
    for m in module.modules():
        if isinstance(m, Act):
            yield m
