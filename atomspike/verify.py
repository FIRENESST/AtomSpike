"""Regression checks for LoRA reload, PMSM persistence, and dual-rate time."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from atomspike.config import load_config
from atomspike.convert.pmsm import enable_pmsm, pmsm_status
from atomspike.models.agent import AtomSpikeAgent
from atomspike.models.lora import LoRALinear, apply_lora
from atomspike.runtime.scheduler import DualRateClock
from atomspike.train.common import load_agent, save_checkpoint


def _dummy_batch(agent: AtomSpikeAgent, device: torch.device):
    n = agent.cfg.encoder.frame_size
    frames = torch.rand(2, 3, n, n, device=device)
    state = torch.rand(2, agent.cfg.reasoner.game_state_dim, device=device)
    return frames, state


def _logits(agent: AtomSpikeAgent, frames, state) -> torch.Tensor:
    agent.eval()
    agent.reset_runtime()
    with torch.no_grad():
        out = agent(frames, state, reason_tick=True)
    return torch.cat([s.flatten() for s in out["logits"]])


def check_pmsm_survives_save(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    frames, state = _dummy_batch(agent, device)
    before = _logits(agent, frames, state)
    n = enable_pmsm(agent, threshold=0.05)
    after = _logits(agent, frames, state)
    if torch.allclose(before, after, atol=1e-6):
        raise AssertionError("PMSM conversion did not change network outputs")
    if not pmsm_status(agent)["enabled"]:
        raise AssertionError("PMSM status enabled=False after convert")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "snn.pt"
        save_checkpoint(agent, path, extra={"pmsm": {"enabled": True, "threshold": 0.05}})
        loaded = load_agent(path, cfg, device)
    status = pmsm_status(loaded)
    if not status["enabled"]:
        raise AssertionError("PMSM disabled after load_agent — conversion evaporated")
    reloaded = _logits(loaded, frames, state)
    if not torch.allclose(after, reloaded, atol=1e-5):
        raise AssertionError("PMSM outputs changed after save/load")
    return {"ok": True, "n_acts": n, "status": status}


def check_lora_roundtrip(cfg_path: Path) -> dict:
    cfg = load_config(cfg_path)
    device = torch.device("cpu")
    agent = AtomSpikeAgent(cfg).to(device)
    meta = apply_lora(agent, r=4, alpha=8)
    if meta["wrapped"] <= 0:
        raise AssertionError("T4 LoRA wrapped 0 layers")
    if not any(isinstance(m, LoRALinear) for m in agent.reasoner.modules()):
        raise AssertionError("reasoner has no LoRALinear after apply_lora")
    if any(p.requires_grad for p in agent.encoder.parameters()):
        raise AssertionError("encoder was not frozen")
    lora_train = [n for n, p in agent.named_parameters() if p.requires_grad and "lora_" in n]
    if not lora_train:
        raise AssertionError("no trainable lora_* parameters")
    frames, state = _dummy_batch(agent, device)
    y = _logits(agent, frames, state)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "peft.pt"
        save_checkpoint(agent, path)
        loaded = load_agent(path, cfg, device)
    if not any(isinstance(m, LoRALinear) for m in loaded.reasoner.modules()):
        raise AssertionError("LoRA wrappers missing after load")
    y2 = _logits(loaded, frames, state)
    if not torch.allclose(y, y2, atol=1e-5):
        raise AssertionError("LoRA outputs changed after save/load")
    return {"ok": True, **meta}


def check_dual_rate_sim() -> dict:
    clock = DualRateClock(perception_hz=5.0, policy_hz=30.0, mode="sim", pace=False)
    perc = 0
    for _ in range(30):
        d = clock.start_tick()
        if not d.run_policy:
            raise AssertionError("sim mode must issue a policy tick every call")
        perc += int(d.run_perception)
        clock.finish_tick(d)
    if clock.n_policy != 30:
        raise AssertionError(f"expected 30 policy ticks, got {clock.n_policy}")
    if perc != 5 or clock.n_perception != 5:
        raise AssertionError(f"expected 5 perception ticks/s, got {clock.n_perception} (flags={perc})")
    hz = clock.scheduled_hz()
    if abs(hz["policy_hz"] - 30.0) > 1e-3:
        raise AssertionError(f"sim policy_hz {hz['policy_hz']} != 30")
    if abs(hz["perception_hz"] - 5.0) > 1e-3:
        raise AssertionError(f"sim perception_hz {hz['perception_hz']} != 5")
    return {"ok": True, **{k: hz[k] for k in ("policy_hz", "perception_hz", "elapsed_s")}}


def check_dual_rate_realtime(seconds: float = 0.35) -> dict:
    import time

    clock = DualRateClock(perception_hz=5.0, policy_hz=30.0, mode="realtime", pace=True)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        d = clock.start_tick()
        clock.finish_tick(d)
    hz = clock.scheduled_hz()
    if not (18.0 <= hz["policy_hz"] <= 42.0):
        raise AssertionError(f"realtime policy_hz {hz['policy_hz']:.1f} not near 30")
    if not (2.0 <= hz["perception_hz"] <= 10.0):
        raise AssertionError(f"realtime perception_hz {hz['perception_hz']:.1f} not near 5")
    return {"ok": True, "policy_hz": hz["policy_hz"], "perception_hz": hz["perception_hz"], "elapsed_s": hz["elapsed_s"]}


def run_verify(cfg_path: Path) -> dict:
    return {
        "pmsm": check_pmsm_survives_save(cfg_path),
        "lora": check_lora_roundtrip(cfg_path),
        "dual_rate_sim": check_dual_rate_sim(),
        "dual_rate_realtime": check_dual_rate_realtime(),
    }
