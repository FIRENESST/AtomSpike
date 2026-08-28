"""Control / timing / efficiency metrics from the README eval table."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from atomspike.config import AtomSpikeConfig
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
    loop = DualRateLoop(agent, cfg, device)
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
    return {
        "success_rate": successes / max(1, episodes),
        "return_mean": float(np.mean(returns) if returns else 0.0),
        "latency_p95_ms": loop.stats.p95_ms(),
        "n_policy_ticks": loop.stats.n_policy,
        "n_reason_ticks": loop.stats.n_reason,
        "expert_token_acc": float(np.mean(expert_match) if expert_match else 0.0),
        "episodes": episodes,
        "params": agent.parameter_count(),
    }


def proxy_energy(sparsity: float, ann_ops: int = 1) -> dict:
    """Very rough energy proxy: spike rate * synaptic ops vs dense ANN."""
    snn_ops = max(0.0, sparsity) * ann_ops
    return {"ann_ops": ann_ops, "snn_ops": snn_ops, "ratio": snn_ops / max(1e-9, ann_ops)}
