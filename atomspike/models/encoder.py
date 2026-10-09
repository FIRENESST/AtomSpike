"""Lightweight Impala-style CNN that emits spatial tokens + a pooled vector.

Each stage keeps the original downsampling + residual block (old checkpoints
load into the same `stem.*` keys). Extra residual blocks are a side ModuleList
with the last conv zeroed, so they start as identity and only add 10Hz cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from atomspike.config import EncoderConfig
from atomspike.models.activations import Act

STEM_STRIDE = 8


def net_input_hw(
    src_h: int,
    src_w: int,
    frame_size: int,
    max_token_h: int = 24,
    max_token_w: int = 40,
) -> tuple[int, int]:
    """CNN input size helper for the 30Hz adapter (encoder keeps native capture size)."""
    src_h = max(STEM_STRIDE, int(src_h))
    src_w = max(STEM_STRIDE, int(src_w))
    cap_h = max(STEM_STRIDE, int(max_token_h) * STEM_STRIDE)
    cap_w = max(STEM_STRIDE, int(max_token_w) * STEM_STRIDE)
    want = max(STEM_STRIDE, int(frame_size) or src_h)
    th = min(src_h, cap_h, want)
    tw = int(round(th * src_w / max(1, src_h)))
    tw = min(max(STEM_STRIDE, tw), cap_w)
    if tw == cap_w and src_w > 0:
        th = min(cap_h, max(STEM_STRIDE, int(round(tw * src_h / src_w))))
    th = max(STEM_STRIDE, th - (th % 2))
    tw = max(STEM_STRIDE, tw - (tw % 2))
    return int(th), int(tw)


def fit_frames(frames: Tensor, hw: tuple[int, int]) -> Tensor:
    """Resize to net_hw. Returns float in [0, 1]."""
    th, tw = int(hw[0]), int(hw[1])
    x = frames.float()
    if frames.dtype == torch.uint8:
        x = x / 255.0
    if int(x.size(-2)) == th and int(x.size(-1)) == tw:
        return x
    return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, identity_init: bool = False):
        super().__init__()
        self.identity_init = bool(identity_init)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act1 = Act("relu")
        self.act2 = Act("relu")
        if self.identity_init:
            nn.init.zeros_(self.conv2.weight)
            if self.conv2.bias is not None:
                nn.init.zeros_(self.conv2.bias)
        self._skip_eval: bool | None = None

    def is_eval_identity(self) -> bool:
        if self.training or not self.identity_init:
            self._skip_eval = None
            return False
        if self._skip_eval is None:
            self._skip_eval = bool((self.conv2.weight.detach() == 0).all().item())
        return self._skip_eval

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.conv1(x))
        h = self.conv2(h)
        if self.identity_init:
            return x + self.act2(h)
        return self.act2(x + h)


class VisualEncoder(nn.Module):
    """RGB frame -> spatial tokens [B, N, D] and pooled [B, D]."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c1, c2, c3 = cfg.stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c1, 3, stride=2, padding=1),
            Act("relu"),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            Act("relu"),
            ResidualBlock(c2),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            Act("relu"),
            ResidualBlock(c3),
        )
        extra = max(0, int(getattr(cfg, "n_blocks_per_stage", 1) or 1) - 1)
        self.extra_blocks = nn.ModuleList(
            nn.ModuleList(ResidualBlock(ch, identity_init=True) for _ in range(extra))
            for ch in (c1, c2, c3)
        )
        self.proj = nn.Conv2d(c3, cfg.d_model, 1)
        self.d_model = cfg.d_model
        self.frame_size = cfg.frame_size
        self.max_token_h = int(cfg.max_token_h)
        self.max_token_w = int(cfg.max_token_w)
        self.last_hw: tuple[int, int] = (1, 1)

    def net_hw(self, src_h: int, src_w: int) -> tuple[int, int]:
        return net_input_hw(src_h, src_w, self.frame_size, self.max_token_h, self.max_token_w)

    def _stem(self, frames: Tensor) -> Tensor:
        if not any(len(stage) for stage in self.extra_blocks):
            return self.stem(frames)
        h = frames
        idx = 0
        skip_extra = not self.training
        for extras in self.extra_blocks:
            h = self.stem[idx](h)
            h = self.stem[idx + 1](h)
            h = self.stem[idx + 2](h)
            idx += 3
            for blk in extras:
                if skip_extra and blk.is_eval_identity():
                    continue
                h = blk(h)
        return h

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """frames: [B, C, H, W] float in [0, 1] or uint8."""
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.dtype != torch.float32 and frames.dtype != torch.float16:
            frames = frames.float()
        h = self._stem(frames)
        h = self.proj(h)
        gh, gw = int(h.shape[-2]), int(h.shape[-1])
        max_h, max_w = self.max_token_h, self.max_token_w
        if max_h > 0 and max_w > 0 and (gh > max_h or gw > max_w):
            h = F.adaptive_avg_pool2d(h, (min(gh, max_h), min(gw, max_w)))
        b, d, gh, gw = h.shape
        self.last_hw = (int(gh), int(gw))
        tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B, HW, D]
        pooled = tokens.mean(dim=1)
        return tokens, pooled
