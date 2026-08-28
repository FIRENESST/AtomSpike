"""T1 behavior cloning warm-start."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent, default_slot_weights
from atomspike.train.common import make_loader, resolve_device, save_checkpoint, set_seed


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
    opt = torch.optim.AdamW(agent.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    history: list[float] = []
    slot_w = default_slot_weights(agent.spec).to(device)
    agent.train()
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"bc {epoch+1}/{n_epochs}", leave=False):
            frames = batch["frame"].to(device)
            prev = batch["prev_frame"].to(device)
            state = batch["game_state"].to(device)
            tokens = batch["action"].to(device)
            w = batch["weight"].to(device)
            out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
            ce_mat = torch.stack(
                [
                    nn.functional.cross_entropy(logit, tokens[:, i], reduction="none")
                    for i, logit in enumerate(out["logits"])
                ],
                dim=1,
            )
            slot = (ce_mat * slot_w).sum(dim=1) / slot_w.sum()
            loss = (slot * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.train.grad_clip)
            opt.step()
            running += float(loss.item()) * frames.size(0)
            n += frames.size(0)
        avg = running / max(1, n)
        history.append(avg)
    ckpt = save_checkpoint(agent, out_path, extra={"bc_loss": history})
    return {"checkpoint": str(ckpt), "loss": history, "params": agent.parameter_count()}
