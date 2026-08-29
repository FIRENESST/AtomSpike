"""SpikedAttention conversion: flip Q/K spike flags on MultiHeadSelfAttention."""

from __future__ import annotations

from pathlib import Path

from atomspike.config import AtomSpikeConfig
from atomspike.convert.pmsm import enable_pmsm
from atomspike.models.attention import MultiHeadSelfAttention
from atomspike.train.common import load_agent, resolve_device, save_checkpoint


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
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt_path, cfg, device)
    n_act = enable_pmsm(agent, threshold=0.0)
    n_mha = enable_spiked_attention(agent, threshold=0.0)
    extra = {
        "pmsm": {"enabled": True, "threshold": 0.0, "n_acts": n_act, "method": "spiked_attention"},
        "spiked_attention": {"enabled": True, "n_mha": n_mha},
        "method": "spiked_attention",
    }
    path = save_checkpoint(agent, out_path, extra=extra)
    return {"checkpoint": str(path), "spiked_mha": n_mha, "replaced": n_act, "method": "spiked_attention"}
