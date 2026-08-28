"""T3 lightweight offline RL: advantage-weighted BC + KL to the BC teacher.

Online PPO would occupy a GPU sampling forever. AW-BC / AWAC-style updates
on logged episodes (optionally self-play dumps) match the README's
'offline first' constraint and stay stable with a KL anchor.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent
from atomspike.train.common import load_agent, make_loader, resolve_device, save_checkpoint, set_seed


def _logprob(logits: list[torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    lp = []
    for i, logit in enumerate(logits):
        logp = torch.log_softmax(logit, dim=-1)
        lp.append(logp.gather(1, tokens[:, i : i + 1]).squeeze(1))
    return torch.stack(lp, dim=1).sum(dim=1)


def train_offline_rl(
    cfg: AtomSpikeConfig,
    data_path: str | Path,
    teacher_ckpt: str | Path,
    out_path: str | Path,
    epochs: int | None = None,
) -> dict:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ds = H5DemoDataset(data_path)
    loader = make_loader(ds, cfg)
    agent = load_agent(teacher_ckpt, cfg, device)
    teacher = deepcopy(agent).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.train.lr * 0.5, weight_decay=cfg.train.weight_decay)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    history: list[float] = []
    agent.train()
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"awbc {epoch+1}/{n_epochs}", leave=False):
            frames = batch["frame"].to(device)
            prev = batch["prev_frame"].to(device)
            state = batch["game_state"].to(device)
            tokens = batch["action"].to(device)
            reward = batch["reward"].to(device)
            out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
            with torch.no_grad():
                t_out = teacher(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                t_lp = _logprob(t_out["logits"], tokens)
                adv = reward - out["value"].detach()
                w = torch.exp((adv / max(cfg.train.advantage_temp, 1e-3)).clamp(-8, 8))
            lp = _logprob(out["logits"], tokens)
            aw = -(w * lp).mean()
            vloss = nn.functional.mse_loss(out["value"], reward)
            kl = (t_lp - lp).pow(2).mean()
            loss = aw + 0.5 * vloss + cfg.train.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(loss.item()) * frames.size(0)
            n += frames.size(0)
        history.append(running / max(1, n))
    ckpt = save_checkpoint(agent, out_path, extra={"offline_loss": history})
    return {"checkpoint": str(ckpt), "loss": history}
