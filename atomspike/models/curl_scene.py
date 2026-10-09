"""10Hz scene fingerprint: curl of a spatial LIF vector field.

Visual tokens are a matrix. Each cell's spikes are mapped to a 2D flow
(vx, vy); discrete curl (∂vy/∂x − ∂vx/∂y) is a rotation signature that
stays more stable across lighting than raw pixels. Bias out is zero so
old checkpoints do not change until this head learns.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import SceneConfig
from atomspike.models.iterator import as_hold_mask, spatial_grid
from atomspike.models.lif import LIFCell

# Vanilla survival HUD is the bottom strip (hotbar / hearts / XP).
# Our overlay is WDA_EXCLUDEFROMCAPTURE, so it is not in the grabbed frame —
# masking the top or right here would delete world cells, not chrome.
HUD_TOP_FRAC = 0.0
HUD_BOTTOM_FRAC = 0.18
HUD_RIGHT_FRAC = 0.0


def chrome_keep_map(h: int, w: int):
    """True on world cells; False on top bar / hotbar / right HUD panel."""
    import numpy as np

    h = max(1, int(h))
    w = max(1, int(w))
    yy = (np.arange(h, dtype=np.float32)[:, None] + 0.5) / float(h)
    xx = (np.arange(w, dtype=np.float32)[None, :] + 0.5) / float(w)
    keep = (yy >= HUD_TOP_FRAC) & (yy <= (1.0 - HUD_BOTTOM_FRAC)) & (xx <= (1.0 - HUD_RIGHT_FRAC))
    if not bool(keep.any()):
        keep = np.ones((h, w), dtype=bool)
    return keep


def chrome_keep_tensor(h: int, w: int, *, device, dtype) -> Tensor:
    keep = torch.as_tensor(chrome_keep_map(h, w), device=device)
    return keep.to(dtype=dtype)


def discrete_curl(vx: Tensor, vy: Tensor) -> Tensor:
    """vx, vy: [B, H, W] → curl_z [B, H, W] (forward differences, edge-padded)."""
    dvy_dx = F.pad(vy[..., 1:] - vy[..., :-1], (0, 1))
    dvx_dy = F.pad(vx[..., 1:, :] - vx[..., :-1, :], (0, 0, 0, 1))
    return dvy_dx - dvx_dy


def _fit_cells(cells: Tensor, h: int, w: int) -> Tensor:
    """Pad/pool a token grid to exactly h×w cells."""
    n_cells = h * w
    if cells.size(1) == n_cells:
        return cells
    b = int(cells.size(0))
    n = int(cells.size(1))
    if n <= 1:
        return cells.mean(dim=1, keepdim=True).expand(b, n_cells, cells.size(-1)).contiguous()
    gh = max(1, int(round(n ** 0.5)))
    gw = max(1, (n + gh - 1) // gh)
    pad_n = gh * gw - n
    work = F.pad(cells, (0, 0, 0, pad_n)) if pad_n > 0 else cells
    pooled = work[:, : gh * gw].view(b, gh, gw, -1).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(pooled, (h, w))
    return pooled.flatten(2).transpose(1, 2).contiguous()


class SpikeCurlScene(nn.Module):
    """LIF on a small spatial grid, then curl stats → d_model residual."""

    def __init__(self, cfg: SceneConfig, d_model: int):
        super().__init__()
        self.spatial_h, self.spatial_w = spatial_grid(cfg)
        self.hw = self.spatial_h
        self.spatial_dim = max(8, int(cfg.spatial_dim))
        self.d_model = int(d_model)
        n_cells = self.spatial_h * self.spatial_w
        self.cell_proj = nn.Linear(self.d_model, self.spatial_dim)
        self.lif = LIFCell(
            tau=float(cfg.lif_tau),
            v_th=float(cfg.lif_v_th),
            thresh_shape=(self.spatial_dim,),
        )
        self.vec = nn.Linear(self.spatial_dim, 2, bias=False)
        # 4x6 curl map + mean / |mean| / std / max
        self.feat_dim = n_cells + 4
        self.bias_out = nn.Linear(self.feat_dim, self.d_model, bias=False)
        nn.init.zeros_(self.bias_out.weight)

    def forward(
        self,
        grid: Tensor,
        mem: Tensor | None = None,
        gui_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b = int(grid.size(0))
        cells = _fit_cells(self.cell_proj(grid), self.spatial_h, self.spatial_w)
        hold = as_hold_mask(gui_mask, b, cells)
        mem_in = mem
        if mem_in is not None and mem_in.shape != cells.shape:
            mem_in = None
        spike, mem_new = self.lif(cells, mem_in)
        if hold is not None:
            vis = (~hold).to(dtype=cells.dtype).view(b, 1, 1)
            if mem_in is not None:
                mem_new = mem_new * vis + mem_in * (1.0 - vis)
                spike = spike * vis
            else:
                spike = spike * vis
        # The vector field must be downstream of the LIF output.  Using the
        # analog ``cells`` here made membrane/threshold/spike changes invisible
        # to both the curl feature and the policy.  Centering the binary spike
        # map still prevents a uniformly bright screen becoming one arrow.
        dev = spike - spike.mean(dim=1, keepdim=True)
        field = self.vec(dev)
        vx = field[..., 0].view(b, self.spatial_h, self.spatial_w)
        vy = field[..., 1].view(b, self.spatial_h, self.spatial_w)
        keep = chrome_keep_tensor(
            self.spatial_h, self.spatial_w, device=vx.device, dtype=vx.dtype
        ).unsqueeze(0)
        mass = keep.sum().clamp_min(1.0)
        vx = vx - (vx * keep).sum(dim=(1, 2), keepdim=True) / mass
        vy = vy - (vy * keep).sum(dim=(1, 2), keepdim=True) / mass
        curl = discrete_curl(vx * keep, vy * keep)
        flat = curl.reshape(b, -1)
        stats = torch.stack(
            [
                curl.mean(dim=(1, 2)),
                curl.abs().mean(dim=(1, 2)),
                curl.flatten(1).std(dim=1, unbiased=False),
                curl.flatten(1).amax(dim=1),
            ],
            dim=-1,
        )
        feat = torch.cat([flat, stats], dim=-1)
        return {
            "bias": self.bias_out(feat),
            "mem": mem_new,
            "curl": curl,
            "vx": vx,
            "vy": vy,
            "spike": spike,
            "energy": stats[:, 1],
        }


def overlay_field(out: dict) -> dict:
    """2-channel spatial vector field for the in-game HUD."""
    pack = torch.stack((out["vx"][0].detach(), out["vy"][0].detach()), dim=0)
    return {
        "vx": pack[0],
        "vy": pack[1],
        "hw": (int(pack.shape[1]), int(pack.shape[2])),
    }
