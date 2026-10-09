"""30Hz temporal residual adapter.

Perception/reasoning run at 10Hz (5Hz in smoke). Without a cheap residual, the
policy would see a stale scene between reason ticks. A tiny conv on the current
frame (and optional frame-diff) supplies motion that the 30Hz head can act on
immediately.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import TemporalAdapterConfig
from atomspike.models.activations import Act
from atomspike.models.encoder import fit_frames
from atomspike.models.iterator import as_hold_mask


class TemporalResidualAdapter(nn.Module):
    def __init__(self, cfg: TemporalAdapterConfig):
        super().__init__()
        self.enabled = cfg.enabled
        self.d_model = cfg.d_model
        self.last_live: Tensor | None = None
        self.down_hw = None
        if not cfg.enabled:
            self.net = None
            return
        hidden = cfg.hidden
        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, 5, stride=2, padding=2),
            Act("relu"),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            Act("relu"),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, cfg.d_model)
        self.gui_scale = nn.Parameter(torch.zeros(1))
        dh = int(getattr(cfg, "down_h", 0) or 0)
        dw = int(getattr(cfg, "down_w", 0) or 0)
        self.down_hw = (max(8, dh), max(8, dw)) if dh >= 8 and dw >= 8 else None

    def forward(
        self,
        frames: Tensor,
        prev_frames: Tensor | None = None,
        hold_mask: Tensor | None = None,
        overlay: bool | None = None,
    ) -> Tensor:
        """frames: [B, C, H, W]. Returns [B, D] residual (zeros if disabled).

        hold_mask: GUI/pause rows freeze appearance on the previous (world) frame
        and zero the frame-diff so inventory pixels are not treated as world motion.
        A zero-init `gui_scale` residual still sees the live GUI so mouse motion
        inside inventories can be learned without changing old checkpoints.
        """
        b = frames.size(0)
        self.last_live = None
        if not self.enabled or self.net is None:
            return torch.zeros(b, self.d_model, device=frames.device, dtype=torch.float32)
        if self.down_hw is not None and (
            int(frames.size(-2)) > self.down_hw[0] or int(frames.size(-1)) > self.down_hw[1]
        ):
            frames = fit_frames(frames, self.down_hw)
            if prev_frames is not None:
                prev_frames = fit_frames(prev_frames, self.down_hw)
        x = _to_unit_float(frames)
        prev = x if prev_frames is None else _to_unit_float(prev_frames)
        hold = as_hold_mask(hold_mask, b, x)
        diff = x - prev
        x_world = x
        diff_world = diff
        if hold is not None:
            vis = (~hold).to(dtype=x.dtype).view(b, 1, 1, 1)
            x_world = x * vis + prev * (1.0 - vis)
            diff_world = diff * vis
        residual = self.proj(self.net(torch.cat([x_world, diff_world], dim=1)).flatten(1))
        want_live = False
        if hold is not None:
            if overlay is True:
                want_live = True
            elif overlay is False:
                want_live = False
            else:
                want_live = bool(self.training) or bool(hold.any().item())
        if want_live:
            live = self.proj(self.net(torch.cat([x, diff], dim=1)).flatten(1))
            gated = live * hold.to(dtype=live.dtype).view(b, 1)
            self.last_live = gated
            residual = residual + self.gui_scale * gated
        return residual


def _to_unit_float(frames: Tensor) -> Tensor:
    x = frames.float()
    if frames.dtype == torch.uint8:
        return x / 255.0
    return x
