"""Control / timing / efficiency metrics from the README eval table."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from atomspike.config import AtomSpikeConfig
from atomspike.convert.pmsm import pmsm_status
from atomspike.data.io import H5DemoDataset
from atomspike.envs import make_env
from atomspike.models.agent import AtomSpikeAgent
from atomspike.runtime.loop import DualRateLoop
from atomspike.train.common import load_agent, resolve_device


def top1_accuracy(pred: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    if pred.size == 0:
        return 0.0
    return float((pred == target).mean())


def macro_f1(pred: NDArray[np.int64], target: NDArray[np.int64], n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.logical_and(pred == c, target == c).sum()
        fp = np.logical_and(pred == c, target != c).sum()
        fn = np.logical_and(pred != c, target == c).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate_policy(
    cfg: AtomSpikeConfig,
    ckpt: str | Path,
    episodes: int = 12,
    deterministic: bool = True,
) -> dict:
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt, cfg, device)
    agent.eval()
    env = make_env(
        cfg.env.kind,
        frame_size=cfg.env.frame_size,
        max_steps=cfg.env.max_steps,
        seed=cfg.env.seed + 1000,
        action_cfg=cfg.action,
    )
    loop = DualRateLoop(agent, cfg, device, mode="sim", pace=False)
    successes = 0
    returns = []
    expert_match = []
    for ep in range(episodes):
        loop.reset()
        obs = env.reset(seed=cfg.env.seed + 2000 + ep)
        ep_ret = 0.0
        while True:
            tokens, _info = loop.tick(obs.frame, obs.game_state, deterministic=deterministic)
            if hasattr(env, "expert_tokens"):
                expert = env.expert_tokens()
                expert_match.append(float(np.mean(tokens == expert)))
            nxt = env.step(tokens)
            ep_ret += float(nxt.reward)
            obs = nxt
            if nxt.done:
                successes += int(bool(nxt.info.get("success", False)))
                returns.append(ep_ret)
                break
    env.close()
    hz = loop.clock.scheduled_hz()
    status = pmsm_status(agent)
    return {
        "success_rate": successes / max(1, episodes),
        "return_mean": float(np.mean(returns) if returns else 0.0),
        "latency_p95_ms": loop.stats.p95_ms(),
        "n_policy_ticks": loop.stats.n_policy,
        "n_reason_ticks": loop.stats.n_reason,
        "scheduled_policy_hz": hz["policy_hz"],
        "scheduled_perception_hz": hz["perception_hz"],
        "clock_mode": loop.clock.mode,
        "pmsm_enabled": status["enabled"],
        "pmsm_n_acts": status.get("n_acts", 0),
        "expert_token_acc": float(np.mean(expert_match) if expert_match else 0.0),
        "episodes": episodes,
        "params": agent.parameter_count(),
    }


@torch.no_grad()
def evaluate_offline(
    cfg: AtomSpikeConfig,
    ckpt: str | Path,
    data_path: str | Path,
    max_steps: int | None = None,
) -> dict:
    """Token accuracy on recorded demos. Used when there is no scripted env (Minecraft)."""
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt, cfg, device)
    agent.eval()
    ds = H5DemoDataset.from_config(data_path, cfg)
    ds.assert_matches(cfg)
    n = len(ds) if max_steps is None else min(len(ds), int(max_steps))
    batch_size = max(1, int(cfg.train.batch_size))
    pred_all: list[np.ndarray] = []
    tgt_all: list[np.ndarray] = []
    gui_all: list[np.ndarray] = []
    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        sl = range(start, min(start + batch_size, n))
        frames = torch.stack([torch.from_numpy(ds[j]["frame"]) for j in sl]).to(device)
        prev = torch.stack([torch.from_numpy(ds[j]["prev_frame"]) for j in sl]).to(device)
        state = torch.stack([torch.from_numpy(ds[j]["game_state"]) for j in sl]).to(device)
        tgt = torch.stack([torch.from_numpy(ds[j]["action"]) for j in sl]).to(device)
        clock = torch.stack([torch.from_numpy(ds[j]["clock"]) for j in sl]).to(device)
        out = agent(frames, state, prev_frames=prev, reason_tick=True, clock=clock)
        pred = torch.stack([lg.argmax(dim=-1) for lg in out["logits"]], dim=1)
        pred_all.append(pred.cpu().numpy())
        tgt_all.append(tgt.cpu().numpy())
        if ds.gui_index is not None:
            gui_all.append(state[:, ds.gui_index].cpu().numpy())
    pred = np.concatenate(pred_all)
    tgt = np.concatenate(tgt_all)
    eq = pred == tgt
    slot_acc = eq.mean(axis=0).tolist()
    names = list(agent.spec.key_names) + ["mouse_x", "mouse_y"] + ["LMB", "RMB"][: agent.spec.n_mouse_buttons]
    slot_named = {names[i] if i < len(names) else f"slot{i}": float(slot_acc[i]) for i in range(len(slot_acc))}
    mx_i = agent.spec.n_keys
    center = int(agent.spec.mouse_center)
    gui_acc = look_acc = gui_mouse = look_mouse = None
    look_mouse_move = look_mouse_sign = None
    if gui_all:
        gui = np.concatenate(gui_all) > 0.5
        if gui.any():
            gui_acc = float(eq[gui].mean())
            gui_mouse = [float(eq[gui, mx_i].mean()), float(eq[gui, mx_i + 1].mean())]
        look = ~gui
        if look.any():
            look_acc = float(eq[look].mean())
            look_mouse = [float(eq[look, mx_i].mean()), float(eq[look, mx_i + 1].mean())]
            moving = look & ((tgt[:, mx_i] != center) | (tgt[:, mx_i + 1] != center))
            if moving.any():
                look_mouse_move = [float(eq[moving, mx_i].mean()), float(eq[moving, mx_i + 1].mean())]
            look_mouse_sign = [
                float((np.sign(pred[look, mx_i] - center) == np.sign(tgt[look, mx_i] - center)).mean()),
                float((np.sign(pred[look, mx_i + 1] - center) == np.sign(tgt[look, mx_i + 1] - center)).mean()),
            ]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(1, n)
    status = pmsm_status(agent)
    return {
        "token_acc": float(eq.mean()),
        "n": int(n),
        "slot_acc": slot_named,
        "gui_acc": gui_acc,
        "look_acc": look_acc,
        "gui_mouse_acc": gui_mouse,
        "look_mouse_acc": look_mouse,
        "look_mouse_move_acc": look_mouse_move,
        "look_mouse_sign_acc": look_mouse_sign,
        "ms_per_step": elapsed_ms,
        "pmsm_enabled": status["enabled"],
        "params": agent.parameter_count(),
        "mouse_bins": int(agent.spec.mouse_bins),
        "data": str(data_path),
    }


def proxy_energy(sparsity: float, ann_ops: int = 1) -> dict:
    """Very rough energy proxy: spike rate * synaptic ops vs dense ANN."""
    snn_ops = max(0.0, sparsity) * ann_ops
    return {"ann_ops": ann_ops, "snn_ops": snn_ops, "ratio": snn_ops / max(1e-9, ann_ops)}
