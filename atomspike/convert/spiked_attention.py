"""SpikedAttention conversion: flip Q/K spike flags on MultiHeadSelfAttention."""

from __future__ import annotations

from pathlib import Path

from atomspike.config import AtomSpikeConfig
from atomspike.models.attention import MultiHeadSelfAttention


def enable_spiked_attention(agent, threshold: float = 0.0) -> int:
    n = 0
    for m in agent.modules():
        if isinstance(m, MultiHeadSelfAttention):
            m.set_spike(True, threshold=threshold)
            n += 1
    return n


def convert_spiked_attention(
    cfg: AtomSpikeConfig,
    ckpt_path: str | Path,
    out_path: str | Path,
) -> dict:
    _ = (cfg, ckpt_path, out_path)
    raise ValueError(
        "SpikedAttention conversion is disabled: it has no per-layer calibration or fidelity evaluation in this "
        "project. Use calibrated PMSM with --data and --allow-lossy-snn, or implement and validate that path first."
    )
