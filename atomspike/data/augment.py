"""Photometric demo augmentation so BC cannot pin an action to one biome tint.

Same random draw is applied to (frame, prev_frame) so the 30Hz frame-diff
adapter still sees real motion. GUI frames are mixed back toward the raw
pixels because inventory icons are the signal. Eval / predict never call this.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_float01(frames: Tensor) -> Tensor:
    x = frames.float()
    if frames.dtype == torch.uint8:
        return x.div(255.0)
    if x.numel() and float(x.max()) > 1.5:
        return x.div(255.0)
    return x


def augment_frame_pair(
    frames: Tensor,
    prev_frames: Tensor,
    cfg: Any,
    gui: Tensor | None = None,
    protect_crosshair: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """frames/prev: [B, C, H, W]. Returns float in [0, 1]."""
    x0 = _as_float01(frames)
    p0 = _as_float01(prev_frames)
    b = x0.size(0)
    device = x0.device
    brightness = float(getattr(cfg, "aug_brightness", 0.0) or 0.0)
    contrast = float(getattr(cfg, "aug_contrast", 0.0) or 0.0)
    saturation = float(getattr(cfg, "aug_saturation", 0.0) or 0.0)
    noise = float(getattr(cfg, "aug_noise", 0.0) or 0.0)
    erase_p = float(getattr(cfg, "aug_erase", 0.0) or 0.0)
    erase_scale = float(getattr(cfg, "aug_erase_scale", 0.08) or 0.08)

    # All photometric terms, including the actual noise tensor and contrast
    # reference, must be shared by the current/previous image.  Independent
    # draws manufacture a false frame-difference signal for the motion adapter.
    pair_mean = 0.5 * (
        x0.mean(dim=(2, 3), keepdim=True) + p0.mean(dim=(2, 3), keepdim=True)
    )

    def _apply(
        x: Tensor,
        br: Tensor,
        ct: Tensor,
        sat: Tensor,
        nscale: Tensor,
        noise_map: Tensor | None,
    ) -> Tensor:
        y = (x * br).clamp(0.0, 1.0)
        # Use a pair-shared reference rather than each image's own mean.
        mean = (pair_mean * br).clamp(0.0, 1.0)
        y = (y * ct + mean * (1.0 - ct)).clamp(0.0, 1.0)
        gray = y.mean(dim=1, keepdim=True)
        y = (y * sat + gray * (1.0 - sat)).clamp(0.0, 1.0)
        if noise_map is not None and float(nscale.max()) > 1e-8:
            y = (y + noise_map * nscale).clamp(0.0, 1.0)
        return y

    br = x0.new_ones(b, 1, 1, 1)
    ct = x0.new_ones(b, 1, 1, 1)
    sat = x0.new_ones(b, 1, 1, 1)
    nscale = x0.new_zeros(b, 1, 1, 1)
    if brightness > 0:
        br = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * brightness
    if contrast > 0:
        ct = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * contrast
    if saturation > 0:
        sat = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * saturation
    if noise > 0:
        nscale = torch.rand(b, 1, 1, 1, device=device) * noise
    noise_map = torch.randn_like(x0) if float(nscale.max()) > 1e-8 else None
    x = _apply(x0, br, ct, sat, nscale, noise_map)
    p = _apply(p0, br, ct, sat, nscale, noise_map)
    if erase_p > 0 and x.size(-1) > 8 and x.size(-2) > 8:
        x, p = _erase_pair(x, p, erase_p, erase_scale, protect_crosshair=protect_crosshair)
    if gui is not None and gui.numel() == b:
        g = (gui.reshape(b, 1, 1, 1) > 0.5).to(dtype=x.dtype)
        # keep most of the raw GUI pixels; still allow a little lighting noise
        x = x * (1.0 - 0.7 * g) + x0 * (0.7 * g)
        p = p * (1.0 - 0.7 * g) + p0 * (0.7 * g)
    return x.clamp(0.0, 1.0), p.clamp(0.0, 1.0)


def _erase_pair(
    x: Tensor, p: Tensor, prob: float, scale: float, protect_crosshair: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    b, _, h, w = x.shape
    out_x = x.clone()
    out_p = p.clone()
    eh = max(1, int(h * scale))
    ew = max(1, int(w * scale))
    for i in range(b):
        # Erasure could remove the only labelled pixel while leaving its
        # crosshair concept target unchanged. Keep supervised examples intact.
        if protect_crosshair is not None and bool(protect_crosshair.reshape(-1)[i]):
            continue
        if float(torch.rand((), device=x.device)) >= prob:
            continue
        top = int(torch.randint(0, max(1, h - eh + 1), (1,), device=x.device))
        left = int(torch.randint(0, max(1, w - ew + 1), (1,), device=x.device))
        fill = torch.rand(1, x.size(1), 1, 1, device=x.device)
        out_x[i : i + 1, :, top : top + eh, left : left + ew] = fill
        out_p[i : i + 1, :, top : top + eh, left : left + ew] = fill
    return out_x, out_p


def maybe_augment_frames(
    frames: Tensor,
    prev_frames: Tensor,
    cfg: Any,
    game_state: Tensor | None = None,
    protect_crosshair: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    brightness = float(getattr(cfg, "aug_brightness", 0.0) or 0.0)
    contrast = float(getattr(cfg, "aug_contrast", 0.0) or 0.0)
    saturation = float(getattr(cfg, "aug_saturation", 0.0) or 0.0)
    noise = float(getattr(cfg, "aug_noise", 0.0) or 0.0)
    erase_p = float(getattr(cfg, "aug_erase", 0.0) or 0.0)
    if max(brightness, contrast, saturation, noise, erase_p) <= 1e-8:
        return frames, prev_frames
    gui = None
    if game_state is not None:
        try:
            from atomspike.capture.mc_state import MC_STATE_FIELDS

            if "gui_open" in MC_STATE_FIELDS:
                gi = MC_STATE_FIELDS.index("gui_open")
                if game_state.size(-1) > gi:
                    gui = game_state[:, gi]
        except Exception:
            gui = None
    return augment_frame_pair(frames, prev_frames, cfg, gui=gui, protect_crosshair=protect_crosshair)
