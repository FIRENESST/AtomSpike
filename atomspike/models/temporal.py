"""30Hz temporal residual adapter.

Perception/reasoning run at 5Hz. Without a cheap residual, the policy would
see a stale scene for ~200ms. A tiny conv on the current frame (and optional
frame-diff) supplies motion that the 30Hz head can act on immediately.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import TemporalAdapterConfig


class TemporalResidualAdapter(nn.Module):
    def __init__(self, cfg: TemporalAdapterConfig):
        super().__init__()
        self.enabled = cfg.enabled
        self.d_model = cfg.d_model
        if not cfg.enabled:
            self.net = None
            return
        hidden = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, cfg.d_model)

    def forward(self, frames: Tensor, prev_frames: Tensor | None = None) -> Tensor:
        """frames: [B, C, H, W]. Returns [B, D] residual (zeros if disabled)."""
        b = frames.size(0)
        if not self.enabled or self.net is None:
            return frames.new_zeros(b, self.d_model)
        x = frames.float()
        if x.max() > 1.5:
            x = x / 255.0
        if prev_frames is None:
            prev = x
        else:
            prev = prev_frames.float()
            if prev.max() > 1.5:
                prev = prev / 255.0
        inp = torch.cat([x, x - prev], dim=1)
        h = self.net(inp).flatten(1)
        return self.proj(h)
