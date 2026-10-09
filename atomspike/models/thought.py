"""10Hz spike deliberation: a few internal LIF steps before the action.

The policy already fires once per 30Hz tick. Thinking is a separate, smaller
loop. A handful of hypothesis cells take the current decision as current,
then recur through spike overlap (coincidence, not softmax). Each step
subtracts the mean spike so one hypothesis can win. The membrane is kept
across perception frames. The readout starts at zero, so an old checkpoint
acts exactly as before until this loop learns.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import ThoughtConfig
from atomspike.models.lif import LIFCell
from atomspike.models.sea import coincidence_weights


class SpikeThought(nn.Module):
    """K hypothesis LIFs, T internal steps, zero-init residual."""

    def __init__(self, cfg: ThoughtConfig, d_model: int):
        super().__init__()
        d = int(d_model)
        self.n_cells = max(2, int(getattr(cfg, "n_cells", 8) or 8))
        self.steps = max(1, int(getattr(cfg, "steps", 3) or 3))
        self.hidden = max(4, int(getattr(cfg, "hidden", 16) or 16))
        tau = float(getattr(cfg, "lif_tau", 0.5) or 0.5)
        v_th = float(getattr(cfg, "lif_v_th", 1.0) or 1.0)
        self.drive = nn.Linear(d, self.n_cells * self.hidden, bias=False)
        self.recur = nn.Linear(self.hidden, self.hidden, bias=False)
        self.lif = LIFCell(tau=tau, v_th=v_th, thresh_shape=(self.hidden,))
        self.read = nn.Linear(self.hidden, d, bias=False)
        nn.init.zeros_(self.read.weight)

    def observe(self, decision: Tensor, mem: Tensor | None = None) -> dict[str, Tensor]:
        b = int(decision.size(0))
        h = self.hidden
        k = self.n_cells
        current = self.drive(decision).view(b, k, h)
        if mem is None or mem.shape != current.shape:
            mem = current.new_zeros(b, k, h)
        settled = current.new_zeros(b, k, h)
        for _ in range(self.steps):
            spike, mem = self.lif(current, mem)
            # Coincidence mix, then subtract the mean so one cell can win.
            weights = coincidence_weights(spike, spike)
            mixed = torch.matmul(weights, spike)
            mixed = mixed - mixed.mean(dim=1, keepdim=True)
            current = current + self.recur(mixed)
            settled = settled + spike
        pooled = settled.mean(dim=1) / float(self.steps)
        return {
            "bias": self.read(pooled),
            "mem": mem,
            "spikes": settled,
        }
