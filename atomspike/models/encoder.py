"""Lightweight Impala-style CNN that emits spatial tokens + a pooled vector."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import EncoderConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class VisualEncoder(nn.Module):
    """RGB frame -> spatial tokens [B, N, D] and pooled [B, D]."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        c1, c2, c3 = cfg.stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c3),
        )
        self.proj = nn.Conv2d(c3, cfg.d_model, 1)
        self.d_model = cfg.d_model
        self.frame_size = cfg.frame_size

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """frames: [B, C, H, W] float in [0, 1] or uint8."""
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.dtype != torch.float32 and frames.dtype != torch.float16:
            frames = frames.float()
        h = self.stem(frames)
        h = self.proj(h)
        b, d, gh, gw = h.shape
        tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B, HW, D]
        pooled = tokens.mean(dim=1)
        return tokens, pooled
