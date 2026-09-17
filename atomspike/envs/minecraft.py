"""Minecraft is collected via the 30fps human capture tool, not a scripted expert."""

from __future__ import annotations


class MinecraftHumanEnv:
    spec_state_dim = 19
    frame_size = 160

    def __init__(self, **kwargs):
        raise RuntimeError(
            "Minecraft T0 is human capture, not a scripted expert. "
            "Open a singleplayer/LAN world then run: python -m atomspike collect-mc --out runs/mc.h5"
        )
