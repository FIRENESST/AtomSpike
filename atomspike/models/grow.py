"""Capped residual adapters under the fused decision vector.

Layers stay identity until born (last linear is zero). At most a handful
are allowed so 30Hz policy ticks do not grow a new MLP stack at runtime.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualGrowLayer(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class GrowBank(nn.Module):
    def __init__(self, d_model: int, state_dim: int, max_layers: int = 2, hidden: int = 16):
        super().__init__()
        self.max_layers = max(0, int(max_layers))
        hidden = max(4, int(hidden))
        self.layers = nn.ModuleList(
            ResidualGrowLayer(d_model, hidden) for _ in range(self.max_layers)
        )
        # ``state_dim`` is retained in the constructor for checkpoint/API
        # compatibility.  The former unused probe added trainable parameters
        # but had no loss or forward path, so it has been removed.
        _ = state_dim
        self.register_buffer("alive", torch.zeros(self.max_layers, dtype=torch.float32))
        self._alive_i: list[int] = []
        self.register_load_state_dict_post_hook(GrowBank._sync_alive)

    def _sync_alive(self, *_args) -> None:
        a = self.alive.detach().cpu()
        self._alive_i = [i for i in range(int(a.numel())) if float(a[i]) > 0.5]

    def n_alive(self) -> int:
        return len(self._alive_i)

    def birth(self) -> int | None:
        for i in range(self.max_layers):
            if i in self._alive_i:
                continue
            self.alive[i] = 1.0
            self._alive_i.append(i)
            return i
        return None

    def forward(self, x: Tensor) -> Tensor:
        for i in self._alive_i:
            x = x + self.layers[i](x)
        return x
