"""T1 behavior cloning warm-start."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.data.rabc import rabc_weights
from atomspike.data.sessions import stride_split_segments
from atomspike.models.agent import AtomSpikeAgent, default_slot_weights
from atomspike.models.planner import plan_aux_loss
from atomspike.train.common import (
    autocast_ctx,
    build_scheduler_if_needed,
    load_checkpoint,
    make_loader,
    maybe_compile,
    optimizer_step,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from atomspike.train.optim import build_optimizer


def stride_split(n: int, val_frac: float, segment_id=None) -> tuple[np.ndarray, np.ndarray]:
    """Spread validation frames through each recording (not across session cuts)."""
    return stride_split_segments(n, val_frac, segment_id)


def slot_class_weights(
    actions: np.ndarray,
    vocabs: tuple[int, ...],
    kinds: tuple[str, ...] | None = None,
    ratio_cap: float = 4.0,
) -> list[torch.Tensor]:
    """Mild sqrt-inverse class weights. Mouse bins stay uniform (idle/center is correct)."""
    ws: list[torch.Tensor] = []
    for i, vocab in enumerate(vocabs):
        if kinds is not None and kinds[i] == "mouse_axis":
            ws.append(torch.ones(int(vocab), dtype=torch.float32))
            continue
        counts = np.bincount(actions[:, i], minlength=int(vocab)).astype(np.float32)
        w = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
        w *= float(vocab) / float(w.sum())
        seen = w[counts >= 1.0]
        if seen.size and ratio_cap > 1.0:
            med = float(np.median(seen))
            w = np.clip(w, med / ratio_cap, med * ratio_cap)
            w *= float(vocab) / float(w.sum())
        ws.append(torch.tensor(w, dtype=torch.float32))
    return ws


def _apply_sample_weights(ds: H5DemoDataset, gui_w: float, motion_w: float) -> None:
    w = ds.weights.copy()
    gui = np.zeros(w.shape[0], dtype=bool)
    if ds.gui_index is not None:
        gui = ds.states[:, ds.gui_index] > 0.5
        if gui_w > 1.0001:
            w[gui] *= float(gui_w)
    if motion_w > 1.0001 and ds.spec is not None:
        mx_i = ds.spec.n_keys
        center = ds.spec.mouse_center
        moving = ~gui & ((ds.actions[:, mx_i] != center) | (ds.actions[:, mx_i + 1] != center))
        w[moving] *= float(motion_w)
    mean = float(w.mean()) or 1.0
    ds.weights = (w / mean).astype(np.float32)


def _plan_tensors(batch, device):
    out = {
        "future_state": batch["future_state"].to(device, non_blocking=True),
        "time_frac": batch["time_frac"].to(device, non_blocking=True),
    }
    if "future_state_near" in batch:
        out["future_state_near"] = batch["future_state_near"].to(device, non_blocking=True)
    if "future_state_far" in batch:
        out["future_state_far"] = batch["future_state_far"].to(device, non_blocking=True)
    if "clock" in batch:
        out["clock"] = batch["clock"].to(device, non_blocking=True)
    return out


def _total_loss(out, tokens, w, slot_w, class_w, cfg, batch_plan, spec=None):
    ce_mat = _slot_ce(out["logits"], tokens, class_w)
    boost = float(getattr(cfg.train, "look_move_ce_boost", 0.0) or 0.0)
    if boost > 0.0 and spec is not None:
        mx = spec.n_keys
        center = spec.mouse_center
        moving = ((tokens[:, mx] != center) | (tokens[:, mx + 1] != center)).to(dtype=ce_mat.dtype)
        scale = 1.0 + boost * moving
        ce_mat = ce_mat.clone()
        ce_mat[:, mx] = ce_mat[:, mx] * scale
        ce_mat[:, mx + 1] = ce_mat[:, mx + 1] * scale
    slot = (ce_mat * slot_w).sum(dim=1) / slot_w.sum()
    loss = (slot * w).mean()
    if cfg.planner.enabled and "plan" in out:
        loss = loss + plan_aux_loss(out["plan"], batch_plan, cfg.planner)
    return loss


@torch.no_grad()
def _eval_loader(agent: AtomSpikeAgent, loader, device, cfg: AtomSpikeConfig, slot_w, class_w) -> dict:
    agent.eval()
    running = 0.0
    n = 0
    hit = 0
    slot_hit: list[int] | None = None
    slot_n = 0
    entropy = 0.0
    plan_n = 0
    agree = 0
    agree_n = 0
    for batch in loader:
        frames = batch["frame"].to(device, non_blocking=True)
        prev = batch["prev_frame"].to(device, non_blocking=True)
        state = batch["game_state"].to(device, non_blocking=True)
        tokens = batch["action"].to(device, non_blocking=True)
        w = batch["weight"].to(device, non_blocking=True)
        plan_batch = _plan_tensors(batch, device)
        with autocast_ctx(device, cfg.train.precision):
            out = agent(
                frames,
                state,
                tokens=tokens,
                prev_frames=prev,
                reason_tick=True,
                future_state=plan_batch["future_state"],
                clock=plan_batch.get("clock"),
            )
            loss = _total_loss(out, tokens, w, slot_w, class_w, cfg, plan_batch, spec=agent.spec)
        pred = torch.stack([lg.argmax(dim=-1) for lg in out["logits"]], dim=1)
        running += float(loss.item()) * frames.size(0)
        n += frames.size(0)
        eq = pred == tokens
        hit += int(eq.sum().item())
        slot_n += int(eq.shape[0])
        cur = eq.sum(dim=0).cpu()
        slot_hit = cur.tolist() if slot_hit is None else [a + int(b) for a, b in zip(slot_hit, cur.tolist())]
        if "plan" in out:
            mix = out["plan"]["mix"].float().clamp_min(1e-8)
            entropy += float((-(mix * mix.log()).sum(-1)).mean().item()) * int(mix.size(0))
            plan_n += int(mix.size(0))
            if "teacher_mix" in out["plan"]:
                agree += int((out["plan"]["next_logits"].argmax(-1) == out["plan"]["teacher_mix"].argmax(-1)).sum().item())
                agree_n += int(mix.size(0))
    agent.train()
    n_slots = int(slot_w.numel())
    overall = hit / max(1, n_slots * n)
    per_slot = [c / max(1, slot_n) for c in (slot_hit or [0] * n_slots)]
    return {
        "loss": running / max(1, n),
        "token_acc": overall,
        "slot_acc": per_slot,
        "phase_acc": entropy / max(1, plan_n),
        "next_acc": agree / max(1, agree_n),
        "n": n,
    }


def _slot_ce(logits: list, tokens: torch.Tensor, class_w: list[torch.Tensor] | None) -> torch.Tensor:
    parts = []
    for i, logit in enumerate(logits):
        kw = {}
        if class_w is not None:
            kw["weight"] = class_w[i].to(device=logit.device, dtype=torch.float32)
        parts.append(nn.functional.cross_entropy(logit.float(), tokens[:, i], reduction="none", **kw))
    return torch.stack(parts, dim=1)


def train_bc(
    cfg: AtomSpikeConfig,
    data_path: str | Path,
    out_path: str | Path,
    weights=None,
    epochs: int | None = None,
    rabc: bool = False,
    reset: bool = False,
) -> dict:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ds = H5DemoDataset.from_config(data_path, cfg, weights=weights)
    ds.assert_matches(cfg)
    if weights is None and rabc:
        ds.weights = rabc_weights(ds)
    _apply_sample_weights(ds, cfg.train.gui_sample_weight, cfg.train.motion_sample_weight)

    n_full = int(ds.frames.shape[0])
    train_idx, val_idx = stride_split(n_full, cfg.train.val_frac, getattr(ds, "session_id", None))
    train_ds = ds.subset(train_idx)
    val_ds = ds.subset(val_idx) if val_idx.size else None

    loader = make_loader(train_ds, cfg)
    val_loader = make_loader(val_ds, cfg, shuffle=False) if val_ds is not None else None
    out_path = Path(out_path)
    resume_path = None if reset else (out_path if out_path.is_file() else None)
    ckpt_blob = None
    if resume_path is not None:
        agent, ckpt_blob = load_checkpoint(resume_path, cfg, device, restore_rng=False)
        print(f"resume {resume_path}  (pass --reset to wipe weights and start over)", flush=True)
    else:
        agent = AtomSpikeAgent(cfg).to(device)
        if reset and out_path.is_file():
            print(f"reset: ignoring existing {out_path}", flush=True)
        else:
            print("init: no checkpoint at --out, training from scratch", flush=True)
    agent = maybe_compile(agent, cfg)
    opt = build_optimizer(cfg.train, agent.parameters())
    if ckpt_blob is not None and "optimizer" in ckpt_blob:
        try:
            opt.load_state_dict(ckpt_blob["optimizer"])
        except (ValueError, RuntimeError):
            print("resume: optimizer state incompatible, keeping loaded weights only", flush=True)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    accum = max(1, cfg.train.grad_accum)
    total_steps = n_epochs * max(1, (len(loader) + accum - 1) // accum)
    sched = build_scheduler_if_needed(opt, cfg, total_steps)
    history: list[float] = []
    val_history: list[float] = []
    slot_w = default_slot_weights(agent.spec).to(device)
    class_w = None
    if cfg.train.slot_balance:
        class_w = [
            t.to(device)
            for t in slot_class_weights(
                ds.actions[train_idx], agent.spec.slot_vocabs, kinds=agent.spec.slot_kinds
            )
        ]
    agent.train()
    params = [p for p in agent.parameters() if p.requires_grad]
    best_val = float("inf")
    best_epoch = -1
    extra = {"bc_loss": history, "val_loss": val_history}
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
            plan_batch = _plan_tensors(batch, device)
            with autocast_ctx(device, cfg.train.precision):
                out = agent(
                    frames,
                    state,
                    tokens=tokens,
                    prev_frames=prev,
                    reason_tick=True,
                    future_state=plan_batch["future_state"],
                    clock=plan_batch.get("clock"),
                )
                loss = _total_loss(out, tokens, w, slot_w, class_w, cfg, plan_batch, spec=agent.spec) / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * accum
            n += frames.size(0)
        avg = running / max(1, n)
        history.append(avg)
        msg = f"epoch {epoch+1}/{n_epochs}  train={avg:.4f}"
        if val_loader is not None:
            metrics = _eval_loader(agent, val_loader, device, cfg, slot_w, class_w)
            val_history.append(float(metrics["loss"]))
            mx = my = 0.0
            if len(metrics["slot_acc"]) > agent.spec.n_keys + 1:
                mx = metrics["slot_acc"][agent.spec.n_keys]
                my = metrics["slot_acc"][agent.spec.n_keys + 1]
            msg += (
                f"  val={metrics['loss']:.4f}  acc={metrics['token_acc']:.3f}  "
                f"mouse={mx:.3f}/{my:.3f}  plan_H={metrics['phase_acc']:.2f}  "
                f"self_next={metrics['next_acc']:.3f}  n_val={metrics['n']}"
            )
            if metrics["loss"] < best_val:
                best_val = float(metrics["loss"])
                best_epoch = epoch + 1
                extra["best_epoch"] = best_epoch
                extra["best_val_loss"] = best_val
                extra["best_val_acc"] = metrics["token_acc"]
                extra["best_slot_acc"] = metrics["slot_acc"]
                save_checkpoint(
                    agent,
                    out_path,
                    extra=extra,
                    optimizer=opt,
                    scheduler=sched,
                    keep_k=cfg.train.ckpt_keep_k,
                    verify=cfg.train.ckpt_verify,
                )
        tqdm.write(msg)
    if val_loader is None or best_epoch < 0:
        save_checkpoint(
            agent,
            out_path,
            extra=extra,
            optimizer=opt,
            scheduler=sched,
            keep_k=cfg.train.ckpt_keep_k,
            verify=cfg.train.ckpt_verify,
        )
    return {
        "checkpoint": str(out_path),
        "loss": history,
        "val_loss": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_epoch < 0 else best_val,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds) if val_ds is not None else 0),
        "n_sessions": int(getattr(ds, "n_sessions", 1) or 1),
        "session_lengths": list(getattr(ds, "session_lengths", []) or []),
        "params": agent.parameter_count(),
        "mouse_bins": int(agent.spec.mouse_bins),
        "resumed": resume_path is not None,
        "reset": bool(reset),
    }
