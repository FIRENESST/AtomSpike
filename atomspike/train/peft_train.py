"""T4 LoRA/DoRA-style per-game PEFT and T5 tiny distillation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.models.agent import AtomSpikeAgent, action_ce_loss
from atomspike.models.lora import apply_lora, trainable_parameter_count
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
    if getattr(agent, "_lora_meta", None) is None:
        meta = apply_lora(agent, cfg.train.lora_r, cfg.train.lora_alpha)
    else:
        meta = agent._lora_meta
    train_n, total_n = trainable_parameter_count(agent)
    if train_n <= 0:
        raise RuntimeError("T4 LoRA left 0 trainable parameters")
    agent = maybe_compile(agent, cfg)
    ds = H5DemoDataset(data_path)
    loader = make_loader(ds, cfg)
    opt = build_optimizer(cfg.train, (p for p in agent.parameters() if p.requires_grad))
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
        for step, batch in enumerate(tqdm(loader, desc=f"peft {epoch+1}/{n_epochs}", leave=False)):
            frames = batch["frame"].to(device, non_blocking=True)
            prev = batch["prev_frame"].to(device, non_blocking=True)
            state = batch["game_state"].to(device, non_blocking=True)
            tokens = batch["action"].to(device, non_blocking=True)
            with autocast_ctx(device, cfg.train.precision):
                out = agent(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                loss = action_ce_loss(out["logits"], tokens) / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * accum
            n += frames.size(0)
        history.append(running / max(1, n))
    meta = {**meta, "trainable": train_n, "total": total_n}
    agent._lora_meta = meta
    ckpt = save_checkpoint(
        agent,
        out_path,
        extra={"peft_loss": history, "lora": meta},
        optimizer=opt,
        scheduler=sched,
        keep_k=cfg.train.ckpt_keep_k,
        verify=cfg.train.ckpt_verify,
    )
    return {
        "checkpoint": str(ckpt),
        "loss": history,
        "trainable": train_n,
        "total": total_n,
        "lora_wrapped": meta["wrapped"],
        "lora_params": meta["lora_params"],
    }


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
    student = maybe_compile(student, cfg)
    ds = H5DemoDataset(data_path)
    loader = make_loader(ds, cfg)
    opt = build_optimizer(cfg.train, student.parameters())
    n_epochs = cfg.train.epochs if epochs is None else epochs
    accum = max(1, cfg.train.grad_accum)
    total_steps = n_epochs * max(1, (len(loader) + accum - 1) // accum)
    sched = build_scheduler_if_needed(opt, cfg, total_steps)
    history: list[float] = []
    student.train()
    params = [p for p in student.parameters() if p.requires_grad]
    for epoch in range(n_epochs):
        running = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"distill {epoch+1}/{n_epochs}", leave=False)):
            frames = batch["frame"].to(device, non_blocking=True)
            prev = batch["prev_frame"].to(device, non_blocking=True)
            state = batch["game_state"].to(device, non_blocking=True)
            tokens = batch["action"].to(device, non_blocking=True)
            with autocast_ctx(device, cfg.train.precision):
                with torch.no_grad():
                    t_out = teacher(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                s_out = student(frames, state, tokens=tokens, prev_frames=prev, reason_tick=True)
                loss = 0.0
                for t_logit, s_logit in zip(t_out["logits"], s_out["logits"]):
                    t_p = torch.softmax(t_logit.float(), dim=-1)
                    loss = loss + nn.functional.kl_div(
                        torch.log_softmax(s_logit.float(), dim=-1), t_p, reduction="batchmean"
                    )
                loss = loss / (len(s_out["logits"]) * accum)
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * accum
            n += frames.size(0)
        history.append(running / max(1, n))
    ckpt = save_checkpoint(
        student,
        out_path,
        extra={"distill_loss": history},
        optimizer=opt,
        scheduler=sched,
        keep_k=cfg.train.ckpt_keep_k,
        verify=cfg.train.ckpt_verify,
    )
    return {"checkpoint": str(ckpt), "loss": history}
