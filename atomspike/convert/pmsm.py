"""T6 training-free ANN → SNN conversion that survives checkpoint load.

Activations are `Act` modules with spike_mode/threshold buffers. Conversion
flips those buffers; load_state_dict restores them. No module-class swap.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.activations import Act, iter_acts
from atomspike.models.agent import AtomSpikeAgent
from atomspike.train.common import load_agent, resolve_device, save_checkpoint


def enable_pmsm(agent: nn.Module, threshold: float, scale: float = 1.0) -> int:
    n = 0
    for act in iter_acts(agent):
        act.set_spike(True, threshold=threshold, scale=scale)
        n += 1
    if n <= 0:
        raise RuntimeError("PMSM found 0 Act modules; encoder/reasoner must use Act")
    return n


def disable_pmsm(agent: nn.Module) -> int:
    n = 0
    for act in iter_acts(agent):
        act.set_spike(False)
        n += 1
    return n


def pmsm_status(agent: nn.Module) -> dict:
    acts = list(iter_acts(agent))
    if not acts:
        return {"enabled": False, "n_acts": 0, "threshold": 0.0}
    enabled = all(a.is_spike for a in acts)
    return {
        "enabled": enabled,
        "n_acts": len(acts),
        "n_spike": sum(1 for a in acts if a.is_spike),
        "threshold": float(acts[0].threshold.item()),
    }


@torch.no_grad()
def _calibrate_threshold(agent: AtomSpikeAgent, frames: Tensor, state: Tensor, percentile: float) -> float:
    collected: list[Tensor] = []

    def hook(_m, _inp, out):
        if isinstance(out, Tensor):
            collected.append(out.detach().float().abs().flatten())

    disable_pmsm(agent)
    handles = [m.register_forward_hook(hook) for m in iter_acts(agent)]
    agent.eval()
    agent.reset_runtime()
    agent(frames, state, reason_tick=True)
    for h in handles:
        h.remove()
    if not collected:
        return 0.0
    cat = torch.cat(collected)
    k = max(0, min(cat.numel() - 1, int(cat.numel() * percentile / 100.0)))
    return float(torch.kthvalue(cat, k + 1).values)


@torch.no_grad()
def measure_firing_rate(agent: AtomSpikeAgent, device: torch.device) -> float:
    rates: list[float] = []

    def hook(_m, _i, out):
        if isinstance(out, Tensor):
            rates.append(float((out > 0).float().mean().item()))

    handles = [m.register_forward_hook(hook) for m in iter_acts(agent) if m.is_spike]
    dummy_f = torch.rand(2, 3, agent.cfg.encoder.frame_size, agent.cfg.encoder.frame_size, device=device)
    dummy_s = torch.zeros(2, agent.cfg.reasoner.game_state_dim, device=device)
    agent.eval()
    agent.reset_runtime()
    agent(dummy_f, dummy_s, reason_tick=True)
    for h in handles:
        h.remove()
    if not rates:
        return 0.0
    return float(sum(rates) / len(rates))


def convert_checkpoint(
    cfg: AtomSpikeConfig,
    ckpt_path: str | Path,
    out_path: str | Path,
    data_path: str | Path | None = None,
) -> dict:
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt_path, cfg, device)
    if data_path is not None and Path(data_path).exists():
        ds = H5DemoDataset(data_path)
        n = min(8, len(ds))
        frames = torch.stack([torch.as_tensor(ds[i]["frame"]) for i in range(n)]).to(device)
        state = torch.stack([torch.as_tensor(ds[i]["game_state"]) for i in range(n)]).to(device)
        thr = _calibrate_threshold(agent, frames, state, cfg.convert.threshold_percentile)
    else:
        thr = 0.0
    n_replaced = enable_pmsm(agent, threshold=thr)
    sparsity = measure_firing_rate(agent, device)
    extra = {
        "pmsm": {"enabled": True, "threshold": thr, "n_acts": n_replaced, "method": "pmsm"},
        "sparsity": sparsity,
        "method": "pmsm",
        "time_steps": cfg.convert.time_steps,
    }
    path = save_checkpoint(agent, out_path, extra=extra)
    return {
        "checkpoint": str(path),
        "threshold": thr,
        "replaced": n_replaced,
        "sparsity": sparsity,
        "method": "pmsm",
        "time_steps": cfg.convert.time_steps,
        "status": pmsm_status(agent),
    }
