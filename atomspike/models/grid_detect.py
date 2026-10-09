"""Unsupervised spike-activity grid (not a semantic object detector).

A Minecraft frame is a spatial prior, not a bag of tokens. Like YOLO:

1. Split the screen into a coarse SxT grid (default 6x10).
2. Each cell fires a LIF activity spike; without labels this is not objectness.
3. Firing cells carry an ungrounded latent-class vector. These are not
   semantic labels until a real concept target and loss are connected.
4. Keep the top-M cells. Silent cells do not enter the mix.

The cell's visual vector is not replaced, so which-block identity stays in
the pixels. Class is only a tag. Residual starts at zero.
10Hz only; 30Hz adds the cached bias.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import DetectConfig
from atomspike.models.activations import Act
from atomspike.models.iterator import as_hold_mask
from atomspike.models.lif import LIFCell
from atomspike.models.sea import compact_screen_tokens


SCREEN_KINDS: tuple[str, ...] = tuple(f"latent_{i}" for i in range(8))


class SpikeGridDetect(nn.Module):
    """Per-cell activity (LIF) + latent tag; no grounded class semantics."""

    def __init__(self, cfg: DetectConfig, d_model: int):
        super().__init__()
        d = int(d_model)
        self.grid_h = max(2, int(getattr(cfg, "grid_h", 6) or 6))
        self.grid_w = max(2, int(getattr(cfg, "grid_w", 10) or 10))
        self.n_classes = max(2, int(getattr(cfg, "n_classes", len(SCREEN_KINDS)) or len(SCREEN_KINDS)))
        self.n_keep = max(1, min(int(getattr(cfg, "n_keep", 8) or 8), self.grid_h * self.grid_w))
        hidden = max(8, int(getattr(cfg, "hidden", 32) or 32))
        tau = float(getattr(cfg, "lif_tau", 0.5) or 0.5)
        v_th = float(getattr(cfg, "lif_v_th", 1.0) or 1.0)
        self.cell = nn.Sequential(nn.Linear(d, hidden), Act("gelu"))
        self.obj = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, self.n_classes)
        self.pos = nn.Linear(2, hidden, bias=False)
        self.mix = nn.Linear(hidden + self.n_classes, d, bias=False)
        self.lif = LIFCell(tau=tau, v_th=v_th, thresh_shape=(1,))
        self.resid_scale = nn.Parameter(torch.zeros(d))
        gy, gx = torch.meshgrid(
            torch.linspace(0.0, 1.0, self.grid_h),
            torch.linspace(0.0, 1.0, self.grid_w),
            indexing="ij",
        )
        self.register_buffer("xy", torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1), persistent=False)

    def observe(
        self,
        tokens: Tensor,
        hw: tuple[int, int] | None = None,
        mem: Tensor | None = None,
        gui_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b, n, d = tokens.shape
        gh, gw = self.grid_h, self.grid_w
        want = gh * gw
        if hw is not None and int(hw[0]) * int(hw[1]) == n and int(hw[0]) > 1:
            grid = tokens.transpose(1, 2).reshape(b, d, int(hw[0]), int(hw[1]))
            grid = F.adaptive_avg_pool2d(grid, (gh, gw))
            compact = grid.flatten(2).transpose(1, 2).contiguous()
            layout = (gh, gw)
        else:
            compact, layout = compact_screen_tokens(tokens, hw, want)
            if compact.size(1) != want:
                g = compact.transpose(1, 2).unsqueeze(-1)
                g = F.adaptive_avg_pool2d(g, (want, 1))
                compact = g.squeeze(-1).transpose(1, 2).contiguous()
                layout = (gh, gw)
        n = compact.size(1)
        hold = as_hold_mask(gui_mask, b, compact)
        h = self.cell(compact)
        obj_in = self.obj(h)
        mem_in = mem if mem is not None and mem.shape == obj_in.shape else None
        spike, mem_new = self.lif(obj_in, mem_in)
        if hold is not None:
            vis = (~hold).to(dtype=spike.dtype).view(b, 1, 1)
            if mem_in is not None:
                mem_new = mem_new * vis + mem_in * (1.0 - vis)
            else:
                mem_new = mem_new * vis
            spike = spike * vis
        obj = spike.squeeze(-1)
        gh, gw = int(layout[0]), int(layout[1])
        keep = None
        if gh > 1 and gw > 1 and gh * gw == int(obj.size(1)):
            from atomspike.models.curl_scene import chrome_keep_tensor

            keep = chrome_keep_tensor(gh, gw, device=obj.device, dtype=obj.dtype).reshape(1, -1)
            obj = obj * keep
        logits = self.cls(h)
        k = min(self.n_keep, n)
        idx = obj.detach().topk(k, dim=1).indices
        # Display peaks: excess over the screen median, local maxima, not the
        # conv-border corners that win a flat top-k every frame.
        raw = obj_in.squeeze(-1).float()
        if keep is not None:
            raw = raw * keep.to(dtype=raw.dtype)
        med = raw.median(dim=1, keepdim=True).values
        excess = (raw - med).clamp_min(0.0)
        if gh > 1 and gw > 1 and gh * gw == int(excess.size(1)):
            grid = excess.view(b, gh, gw)
            pad = F.pad(grid, (1, 1, 1, 1), value=-1.0)
            peak = (
                (grid >= pad[:, :-2, 1:-1])
                & (grid >= pad[:, 2:, 1:-1])
                & (grid >= pad[:, 1:-1, :-2])
                & (grid >= pad[:, 1:-1, 2:])
            )
            inner = torch.zeros_like(peak)
            inner[:, 1:-1, 1:-1] = True
            excess = excess * (peak & inner).to(dtype=excess.dtype).reshape(b, -1)
        vis_idx = excess.detach().topk(k, dim=1).indices
        gather = idx.unsqueeze(-1)
        det_h = h.gather(1, gather.expand(-1, -1, h.size(-1)))
        det_cls = logits.gather(1, gather.expand(-1, -1, logits.size(-1)))
        det_obj = obj.gather(1, idx).unsqueeze(-1)
        xy = self.xy.to(device=h.device, dtype=h.dtype)
        if xy.size(0) != n:
            xy = compact.new_zeros(n, 2)
        det_xy = xy.unsqueeze(0).expand(b, -1, -1).gather(1, gather.expand(-1, -1, 2))
        # topk always has k indices, including on a silent frame.  Gate every
        # path into the residual by actual objectness so those bookkeeping
        # indices cannot inject an arbitrary silent-cell feature.
        feat = (det_h + self.pos(det_xy)) * det_obj
        feat = torch.cat([feat, det_cls * det_obj], dim=-1)
        cell_mix = self.mix(feat)
        delta = cell_mix.mean(dim=1)
        vis_gather = vis_idx.unsqueeze(-1)
        vis_logits = logits.gather(1, vis_gather.expand(-1, -1, logits.size(-1)))
        vis_cls = vis_logits.argmax(dim=-1)
        vis_conf = F.softmax(vis_logits.float(), dim=-1).gather(-1, vis_cls.unsqueeze(-1)).squeeze(-1)
        vis_obj = excess.gather(1, vis_idx)
        cls_idx = det_cls.argmax(dim=-1)
        conf = F.softmax(det_cls.float(), dim=-1).gather(-1, cls_idx.unsqueeze(-1)).squeeze(-1)
        infl = cell_mix.float().pow(2).mean(dim=-1).sqrt()
        mass = infl.sum(dim=-1, keepdim=True)
        infl = torch.where(
            mass > 1e-8,
            infl / mass.clamp_min(1e-8),
            torch.zeros_like(infl),
        )
        vis_infl = vis_obj / vis_obj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = compact.new_zeros(b, n)
        # Indices are always present for fixed-shape decoding; the mask is an
        # activity mask, so a silent LIF grid must remain entirely zero.
        mask.scatter_(1, idx, det_obj.squeeze(-1))
        return {
            "bias": self.resid_scale * delta,
            "mem": mem_new,
            "objectness": obj,
            "class_idx": cls_idx,
            "idx": idx,
            "conf": conf,
            "infl": infl,
            "vis_idx": vis_idx,
            "vis_cls": vis_cls,
            "vis_obj": vis_obj,
            "vis_conf": vis_conf,
            "vis_infl": vis_infl,
            "mask": mask,
            "k": k,
            "n": n,
            "hw": layout,
        }


def overlay_cells(out: dict) -> dict:
    """Tiny CPU snapshot of the latent grid (top-M cells). Safe at 10Hz."""
    idx = out.get("vis_idx", out["idx"])[0].detach()
    cls = out.get("vis_cls", out["class_idx"])[0].detach()
    src_obj = out.get("vis_obj", out["objectness"])
    obj = src_obj[0]
    if obj.ndim == 1 and int(obj.numel()) != int(idx.numel()):
        obj = obj.gather(0, idx)
    obj = obj.detach()
    infl = out.get("vis_infl", out["infl"])[0].detach()
    conf = out.get("vis_conf", out["conf"])[0].detach()
    pack = torch.stack((idx.float(), cls.float(), obj.float(), infl.float(), conf.float()), dim=0)
    hw = out.get("hw") or (0, 0)
    return {
        "idx": pack[0],
        "cls": pack[1],
        "obj": pack[2],
        "infl": pack[3],
        "conf": pack[4],
        "hw": (int(hw[0]), int(hw[1])),
        "n": int(out["n"]),
        "k": int(out["k"]),
    }
