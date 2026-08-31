"""T1 behavior cloning warm-start."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent, default_slot_weights
from atomspike.train.common import (
    autocast_ctx,
    build_scheduler_if_needed,
    make_loader,
    maybe_compile,
    optimizer_step,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from atomspike.train.optim import build_optimizer


def train_bc(
    cfg: AtomSpikeConfig,
    data_path: str | Path,
    out_path: str | Path,
    weights=None,
    epochs: int | None = None,
) -> dict:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ds = H5DemoDataset(data_path, weights=weights)
    loader = make_loader(ds, cfg)
    agent = AtomSpikeAgent(cfg).to(device)
    agent = maybe_compile(agent, cfg)
    opt = build_optimizer(cfg.train, agent.parameters())
    n_epochs = cfg.train.epochs if epochs is None else epochs
    accum = max(1, cfg.train.grad_accum)
    total_steps = n_epochs * max(1, (len(loader) + accum - 1) // accum)
    sched = build_scheduler_if_needed(opt, cfg, total_steps)
    history: list[float] = []
    slot_w = default_slot_weights(agent.spec).to(device)
    agent.train()
    params = [p for p in agent.parameters() if p.requires_grad]
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"bc {epoch+1}/{n_epochs}", leave=False)):
            frames = batch["frame"].to(device, non_blocking=True)
            prev = batch["prev_frame"].to(device, non_blocking=True)
            state = batch["game_state"].to(device, non_blocking=True)
            tokens = batch["action"].to(device, non_blocking=True)
            w = batch["weight"].to(device, non_blocking=True)
            with autocast_ctx(device, cfg.train.precision):
                out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                ce_mat = torch.stack(
                    [
                        nn.functional.cross_entropy(logit.float(), tokens[:, i], reduction="none")
                        for i, logit in enumerate(out["logits"])
                    ],
                    dim=1,
                )
                slot = (ce_mat * slot_w).sum(dim=1) / slot_w.sum()
                loss = (slot * w).mean() / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * accum
            n += frames.size(0)
        avg = running / max(1, n)
        history.append(avg)
    ckpt = save_checkpoint(
        agent,
        out_path,
        extra={"bc_loss": history},
        optimizer=opt,
        scheduler=sched,
        keep_k=cfg.train.ckpt_keep_k,
        verify=cfg.train.ckpt_verify,
    )
    return {"checkpoint": str(ckpt), "loss": history, "params": agent.parameter_count()}
