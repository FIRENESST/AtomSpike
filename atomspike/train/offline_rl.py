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
from atomspike.train.common import (
    autocast_ctx,
    build_scheduler_if_needed,
    load_agent,
    make_loader,
    maybe_compile,
    optimizer_step,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from atomspike.train.optim import build_optimizer


def _logprob(logits: list[torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    lp = []
    for i, logit in enumerate(logits):
        logp = torch.log_softmax(logit.float(), dim=-1)
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
    agent = maybe_compile(agent, cfg)
    teacher = deepcopy(agent).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    opt = build_optimizer(cfg.train, agent.parameters())
    # lower LR than BC: AW-BC on top of a warm-started policy is a fine-tune
    for g in opt.param_groups:
        g["lr"] = g["lr"] * 0.5
    n_epochs = cfg.train.epochs if epochs is None else epochs
    accum = max(1, cfg.train.grad_accum)
    total_steps = n_epochs * max(1, (len(loader) + accum - 1) // accum)
    sched = build_scheduler_if_needed(opt, cfg, total_steps)
    history: list[float] = []
    agent.train()
    params = [p for p in agent.parameters() if p.requires_grad]
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"awbc {epoch+1}/{n_epochs}", leave=False)):
            frames = batch["frame"].to(device, non_blocking=True)
            prev = batch["prev_frame"].to(device, non_blocking=True)
            state = batch["game_state"].to(device, non_blocking=True)
            tokens = batch["action"].to(device, non_blocking=True)
            reward = batch["reward"].to(device, non_blocking=True)
            with autocast_ctx(device, cfg.train.precision):
                out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                with torch.no_grad():
                    t_out = teacher(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                    t_lp = _logprob(t_out["logits"], tokens)
                    adv = reward - out["value"].detach().float()
                    w = torch.exp((adv / max(cfg.train.advantage_temp, 1e-3)).clamp(-8, 8))
                lp = _logprob(out["logits"], tokens)
                aw = -(w * lp).mean()
                vloss = nn.functional.mse_loss(out["value"].float(), reward)
                kl = (t_lp - lp).pow(2).mean()
                loss = (aw + 0.5 * vloss + cfg.train.kl_coef * kl) / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * accum
            n += frames.size(0)
        history.append(running / max(1, n))
    ckpt = save_checkpoint(
        agent,
        out_path,
        extra={"offline_loss": history},
        optimizer=opt,
        scheduler=sched,
        keep_k=cfg.train.ckpt_keep_k,
        verify=cfg.train.ckpt_verify,
    )
    return {"checkpoint": str(ckpt), "loss": history}
