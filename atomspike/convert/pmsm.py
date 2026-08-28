"""T6 training-free ANN → SNN conversion (PMSM-style T=1).

PMSM maps dense activations to polarity-aware spikes at a single time-step.
We calibrate thresholds on a demo batch, replace ReLU/GELU with a hard spike,
and keep Linear/Conv weights unchanged. No backprop.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent
from atomspike.train.common import load_agent, resolve_device, save_checkpoint


class SpikeReLU(nn.Module):
    def __init__(self, threshold: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.threshold = float(threshold)
        self.scale = float(scale)

    def forward(self, x: Tensor) -> Tensor:
        spike = (x > self.threshold).to(x.dtype)
        return spike * self.scale


def _replace_activations(module: nn.Module, threshold: float) -> int:
    n = 0
    for m in list(module.modules()):
        if isinstance(m, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            if not isinstance(getattr(m, "activation", None), SpikeReLU):
                m.activation = SpikeReLU(threshold=threshold, scale=1.0)
                n += 1
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.ReLU, nn.GELU, nn.SiLU)):
                setattr(parent, name, SpikeReLU(threshold=threshold, scale=1.0))
                n += 1
    return n


@torch.no_grad()
def _calibrate_threshold(agent: AtomSpikeAgent, frames: Tensor, state: Tensor, percentile: float) -> float:
    acts: list[Tensor] = []

    def hook(_m, _inp, out):
        if isinstance(out, Tensor):
            acts.append(out.detach().float().abs().flatten())

    handles = []
    for m in agent.modules():
        if isinstance(m, (nn.ReLU, nn.GELU)):
            handles.append(m.register_forward_hook(hook))
    agent.eval()
    agent(frames, state, reason_tick=True)
    for h in handles:
        h.remove()
    if not acts:
        return 0.0
    cat = torch.cat(acts)
    k = max(0, min(cat.numel() - 1, int(cat.numel() * percentile / 100.0)))
    return float(torch.kthvalue(cat, k + 1).values)


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
    n_replaced = _replace_activations(agent, threshold=thr)
    sparsity = _measure_sparsity(agent, device)
    path = save_checkpoint(agent, out_path, extra={"pmsm_threshold": thr, "replaced": n_replaced, "sparsity": sparsity})
    return {
        "checkpoint": str(path),
        "threshold": thr,
        "replaced": n_replaced,
        "sparsity": sparsity,
        "method": "pmsm",
        "time_steps": cfg.convert.time_steps,
    }


@torch.no_grad()
def _measure_sparsity(agent: AtomSpikeAgent, device: torch.device) -> float:
    spikes = []

    def hook(m, _i, out):
        if isinstance(out, Tensor):
            spikes.append(float((out > 0).float().mean().item()))

    handles = [m.register_forward_hook(hook) for m in agent.modules() if isinstance(m, SpikeReLU)]
    dummy_f = torch.rand(2, 3, agent.cfg.encoder.frame_size, agent.cfg.encoder.frame_size, device=device)
    dummy_s = torch.zeros(2, agent.cfg.reasoner.game_state_dim, device=device)
    agent.eval()
    agent(dummy_f, dummy_s, reason_tick=True)
    for h in handles:
        h.remove()
    if not spikes:
        return 0.0
    return float(sum(spikes) / len(spikes))
