"""T4 LoRA/DoRA-style per-game PEFT and T5 tiny distillation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent, action_ce_loss
from atomspike.models.lora import freeze_module, inject_lora, trainable_parameter_count
from atomspike.train.common import load_agent, make_loader, resolve_device, save_checkpoint, set_seed


def train_peft(
    cfg: AtomSpikeConfig,
    data_path: str | Path,
    backbone_ckpt: str | Path,
    out_path: str | Path,
    epochs: int | None = None,
) -> dict:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    agent = load_agent(backbone_ckpt, cfg, device)
    freeze_module(agent.encoder)
    freeze_module(agent.reasoner)
    n_wrap = inject_lora(agent.reasoner, cfg.train.lora_r, cfg.train.lora_alpha, ("linear", "in_proj", "out"))
    n_wrap += inject_lora(agent.policy, cfg.train.lora_r, cfg.train.lora_alpha, ("qkv", "out", "pre", "heads"))
    # unfreeze action heads always
    for p in agent.policy.parameters():
        if p.ndim == 2 and p.requires_grad is False and p.shape[-1] <= 64:
            p.requires_grad = True
    train_n, total_n = trainable_parameter_count(agent)
    ds = H5DemoDataset(data_path)
    loader = make_loader(ds, cfg)
    opt = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=cfg.train.lr)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    history: list[float] = []
    agent.train()
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"peft {epoch+1}/{n_epochs}", leave=False):
            frames = batch["frame"].to(device)
            prev = batch["prev_frame"].to(device)
            state = batch["game_state"].to(device)
            tokens = batch["action"].to(device)
            out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
            loss = action_ce_loss(out["logits"], tokens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * frames.size(0)
            n += frames.size(0)
        history.append(running / max(1, n))
    ckpt = save_checkpoint(
        agent,
        out_path,
        extra={"peft_loss": history, "lora_wrapped": n_wrap, "trainable": train_n, "total": total_n},
    )
    return {"checkpoint": str(ckpt), "loss": history, "trainable": train_n, "total": total_n, "lora_wrapped": n_wrap}


def distill_ann_to_small(
    cfg: AtomSpikeConfig,
    data_path: str | Path,
    teacher_ckpt: str | Path,
    student_cfg: AtomSpikeConfig,
    out_path: str | Path,
    epochs: int | None = None,
) -> dict:
    """T5: train a smaller student to match teacher action logits."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    teacher = load_agent(teacher_ckpt, cfg, device)
    teacher.eval()
    student = AtomSpikeAgent(student_cfg).to(device)
    ds = H5DemoDataset(data_path)
    loader = make_loader(ds, cfg)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.train.lr)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    history: list[float] = []
    student.train()
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"distill {epoch+1}/{n_epochs}", leave=False):
            frames = batch["frame"].to(device)
            prev = batch["prev_frame"].to(device)
            state = batch["game_state"].to(device)
            tokens = batch["action"].to(device)
            with torch.no_grad():
                t_out = teacher(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
            s_out = student(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
            loss = 0.0
            for t_logit, s_logit in zip(t_out["logits"], s_out["logits"]):
                t_p = torch.softmax(t_logit, dim=-1)
                loss = loss + nn.functional.kl_div(
                    torch.log_softmax(s_logit, dim=-1), t_p, reduction="batchmean"
                )
            loss = loss / len(s_out["logits"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.item()) * frames.size(0)
            n += frames.size(0)
        history.append(running / max(1, n))
    ckpt = save_checkpoint(student, out_path, extra={"distill_loss": history})
    return {"checkpoint": str(ckpt), "loss": history}
