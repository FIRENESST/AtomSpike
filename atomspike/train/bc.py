"""T1 behavior cloning warm-start."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.data.visual_concepts import CONCEPT_KINDS
from atomspike.data.effects import EFFECT_EVENTS
from atomspike.data.rabc import rabc_weights
from atomspike.data.sessions import stride_split_segments
from atomspike.eval.effect_metrics import EffectBrierMeter
from atomspike.models.agent import AtomSpikeAgent, default_slot_weights
from atomspike.models.iterator import task_forward_kwargs
from atomspike.models.inventory import inventory_forward_kwargs
from atomspike.models.sea import block_pulse_forward_kwargs
from atomspike.models.planner import plan_aux_loss
from atomspike.data.augment import maybe_augment_frames
from atomspike.train.common import (
    autocast_ctx,
    backward_loss,
    build_grad_scaler,
    build_scheduler_if_needed,
    grad_accum_divisor,
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
    """Reserve complete collector recordings for validation."""
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
    if ds.spec is not None and ds.spec.has_wheel:
        spinning = ds.actions[:, ds.spec.wheel_index] != ds.spec.wheel_center
        w[spinning] *= 2.0
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


def _total_loss(
    out, tokens, w, slot_w, class_w, cfg, batch_plan, spec=None, game_state=None,
    look_target=None, effect_target=None, effect_valid=None, effect_active=None, effect_pos_weight=None,
    train_visual_baseline=False,
):
    ce_mat = _slot_ce(out["logits"], tokens, class_w, label_smoothing=float(getattr(cfg.train, "label_smoothing", 0.0) or 0.0))
    boost = float(getattr(cfg.train, "look_move_ce_boost", 0.0) or 0.0)
    if boost > 0.0 and spec is not None:
        mx = spec.n_keys
        center = spec.mouse_center
        moving = ((tokens[:, mx] != center) | (tokens[:, mx + 1] != center)).to(dtype=ce_mat.dtype)
        scale = 1.0 + boost * moving
        ce_mat = ce_mat.clone()
        ce_mat[:, mx] = ce_mat[:, mx] * scale
        ce_mat[:, mx + 1] = ce_mat[:, mx + 1] * scale
    gui_boost = float(getattr(cfg.train, "gui_mouse_ce_boost", 0.0) or 0.0)
    if gui_boost > 0.0 and spec is not None and game_state is not None and game_state.size(-1) > 0:
        from atomspike.capture.mc_state import MC_STATE_FIELDS

        if "gui_open" in MC_STATE_FIELDS:
            gi = MC_STATE_FIELDS.index("gui_open")
            if game_state.size(-1) > gi:
                gui = (game_state[:, gi] > 0.5).to(dtype=ce_mat.dtype)
                mx = spec.n_keys
                scale = 1.0 + gui_boost * gui
                ce_mat = ce_mat.clone()
                ce_mat[:, mx] = ce_mat[:, mx] * scale
                ce_mat[:, mx + 1] = ce_mat[:, mx + 1] * scale
                for bi in range(int(spec.n_mouse_buttons)):
                    ce_mat[:, mx + 2 + bi] = ce_mat[:, mx + 2 + bi] * scale
                if spec.has_wheel:
                    ce_mat[:, spec.wheel_index] = ce_mat[:, spec.wheel_index] * scale
    if spec is not None and spec.has_wheel:
        wi = spec.wheel_index
        spinning = (tokens[:, wi] != spec.wheel_center).to(dtype=ce_mat.dtype)
        ce_mat = ce_mat.clone()
        ce_mat[:, wi] = ce_mat[:, wi] * (1.0 + 2.0 * spinning)
    slot = (ce_mat * slot_w).sum(dim=1) / slot_w.sum()
    loss = (slot * w).mean()
    if "pointer_logits" in out and game_state is not None:
        from atomspike.capture.mc_state import MC_STATE_FIELDS

        if "gui_open" in MC_STATE_FIELDS and "hovered_slot" in MC_STATE_FIELDS:
            gi = MC_STATE_FIELDS.index("gui_open")
            hi = MC_STATE_FIELDS.index("hovered_slot")
            if game_state.size(-1) > max(gi, hi):
                gui = game_state[:, gi] > 0.5
                idx = (game_state[:, hi] * 90.0 - 1.0).round().long()
                pl = out["pointer_logits"]
                valid = gui & (idx >= 0) & (idx < pl.size(-1))
                if bool(valid.any().item()):
                    loss = loss + 0.25 * nn.functional.cross_entropy(
                        pl[valid], idx[valid].clamp(0, pl.size(-1) - 1)
                    )
    if cfg.planner.enabled and "plan" in out:
        loss = loss + plan_aux_loss(out["plan"], batch_plan, cfg.planner)
    if look_target is not None and "look_concept_logits" in out:
        target = look_target.reshape(-1).long()
        valid = target >= 0
        if bool(valid.any().item()):
            # -1 is an absent/stale teacher, never the "air" class.
            loss = loss + float(cfg.train.semantic_loss_weight) * nn.functional.cross_entropy(
                out["look_concept_logits"][valid].float(), target[valid]
            )
    if (
        effect_target is not None and effect_valid is not None and effect_active is not None
        and bool(effect_active.any().item()) and "observed_effect_logits" in out
    ):
        valid_effect = effect_valid.reshape(-1) > 0
        if bool(valid_effect.any().item()):
            active = effect_active.to(device=effect_target.device, dtype=torch.bool)
            target = effect_target[valid_effect][:, active].float()
            for name in ("observed_effect_logits", "visual_only_effect_logits"):
                if name != "observed_effect_logits" and (not train_visual_baseline or name not in out):
                    continue
                logits = out[name][valid_effect][:, active].float()
                pos_weight = None if effect_pos_weight is None else effect_pos_weight.to(logits.device)[active]
                loss = loss + float(cfg.train.effect_loss_weight) * nn.functional.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=pos_weight
                )
    return loss


@torch.no_grad()
def _eval_loader(
    agent: AtomSpikeAgent, loader, device, cfg: AtomSpikeConfig, slot_w, class_w,
    effect_active=None, effect_pos_weight=None, effect_prior=None,
) -> dict:
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
    look_hit = 0
    look_n = 0
    look_confusion = np.zeros((len(CONCEPT_KINDS), len(CONCEPT_KINDS)), dtype=np.int64)
    active_np = np.zeros(len(EFFECT_EVENTS), dtype=bool) if effect_active is None else effect_active.cpu().numpy().astype(bool)
    prior_np = np.zeros(len(EFFECT_EVENTS), dtype=np.float64) if effect_prior is None else effect_prior.cpu().numpy()
    effect_meter = EffectBrierMeter(active_np, prior_np)
    for batch in loader:
        # Evaluation batches are independently shuffled/packed examples, not a
        # continuous rollout.  Never let recurrent runtime state leak from one
        # batch into the next (or from a preceding train batch).
        agent.reset_runtime()
        frames = batch["frame"].to(device, non_blocking=True)
        prev = batch["prev_frame"].to(device, non_blocking=True)
        state = batch["game_state"].to(device, non_blocking=True)
        tokens = batch["action"].to(device, non_blocking=True)
        look_target = batch["look_concept"].to(device, non_blocking=True)
        effect_target = batch["effect_target"].to(device, non_blocking=True)
        effect_valid = batch["effect_valid"].to(device, non_blocking=True)
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
                **task_forward_kwargs(batch, device),
                **inventory_forward_kwargs(batch, device),
                **block_pulse_forward_kwargs(batch, device),
            )
            loss = _total_loss(
                out, tokens, w, slot_w, class_w, cfg, plan_batch,
                spec=agent.spec, game_state=state, look_target=look_target,
                effect_target=effect_target, effect_valid=effect_valid,
                effect_active=effect_active, effect_pos_weight=effect_pos_weight,
            )
        if effect_active is not None and bool(effect_active.any().item()) and "observed_effect_logits" in out and "visual_only_effect_logits" in out:
            ev = effect_valid.reshape(-1) > 0
            if bool(ev.any().item()):
                effect_meter.update(
                    effect_target[ev].float().cpu().numpy(),
                    out["observed_effect_logits"][ev].float().sigmoid().cpu().numpy(),
                    out["visual_only_effect_logits"][ev].float().sigmoid().cpu().numpy(),
                )
        valid_look = look_target.reshape(-1) >= 0
        if bool(valid_look.any().item()) and "look_concept_logits" in out:
            look_pred = out["look_concept_logits"].argmax(-1)[valid_look]
            look_true = look_target.reshape(-1)[valid_look]
            look_hit += int((look_pred == look_true).sum().item())
            look_n += int(valid_look.sum().item())
            np.add.at(look_confusion, (look_true.cpu().numpy(), look_pred.cpu().numpy()), 1)
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
    look_support = look_confusion.sum(axis=1)
    look_predicted = look_confusion.sum(axis=0)
    look_f1 = 2.0 * np.diag(look_confusion) / np.maximum(1, look_support + look_predicted)
    look_macro_f1 = float(look_f1[look_support > 0].mean()) if look_n else 0.0
    effect_report = effect_meter.report()
    return {
        "loss": running / max(1, n),
        "token_acc": overall,
        "slot_acc": per_slot,
        "phase_acc": entropy / max(1, plan_n),
        "next_acc": agree / max(1, agree_n),
        "n": n,
        "look_acc": look_hit / max(1, look_n),
        "look_n": look_n,
        "look_macro_f1": look_macro_f1,
        "look_majority_acc": float(look_support.max()) / max(1, look_n),
        "effect_brier": effect_report["conditioned"],
        "effect_visual_brier": effect_report["visual_only"],
        "effect_prior_brier": effect_report["prior"],
        "effect_n": effect_report["n"],
        "effect_by_event": effect_report["by_event"],
    }


def _slot_ce(
    logits: list,
    tokens: torch.Tensor,
    class_w: list[torch.Tensor] | None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    parts = []
    smooth = float(max(0.0, label_smoothing))
    for i, logit in enumerate(logits):
        kw = {}
        if class_w is not None:
            kw["weight"] = class_w[i].to(device=logit.device, dtype=torch.float32)
        parts.append(
            nn.functional.cross_entropy(
                logit.float(),
                tokens[:, i],
                reduction="none",
                label_smoothing=smooth,
                **kw,
            )
        )
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
    train_idx, val_idx = stride_split(n_full, cfg.train.val_frac, getattr(ds, "recording_id", None))
    train_ds = ds.subset(train_idx)
    val_ds = ds.subset(val_idx) if val_idx.size else None
    n_look = int((ds.look_concept[train_idx] >= 0).sum())
    print(f"[semantic] labelled train rays={n_look}; legacy/unobserved rows are ignored", flush=True)
    train_effect_valid = ds.effect_valid[train_idx] > 0
    train_effect = ds.effect_target[train_idx][train_effect_valid]
    n_effect = int(train_effect.shape[0])
    effect_pos = train_effect.sum(axis=0) if n_effect else np.zeros(len(EFFECT_EVENTS), dtype=np.float32)
    effect_neg = n_effect - effect_pos
    # A column with no observed positive examples must not be trained or
    # advertised as a useful event predictor. Eight is a minimal smoke gate,
    # not evidence of statistical reliability.
    active_np = (effect_pos >= 8) & (effect_neg >= 8)
    effect_active = torch.tensor(active_np, device=device, dtype=torch.bool)
    effect_pos_weight = torch.tensor(
        np.clip(effect_neg / np.maximum(1, effect_pos), 1.0, 20.0), device=device, dtype=torch.float32
    )
    effect_prior = torch.tensor(effect_pos / max(1, n_effect), device=device, dtype=torch.float32)
    active_names = [name for name, active in zip(EFFECT_EVENTS, active_np) if active]
    print(f"[effect] valid train windows={n_effect}; active observed events={active_names}", flush=True)

    loader = make_loader(train_ds, cfg)
    val_loader = make_loader(val_ds, cfg, shuffle=False) if val_ds is not None else None
    out_path = Path(out_path)
    latest_path = out_path.with_name(f"{out_path.stem}.last{out_path.suffix}")
    resume_path = None
    if not reset:
        candidates = [path for path in (out_path, latest_path) if path.is_file()]
        if candidates:
            # Normally .last is newer.  Choosing by publication time also
            # avoids a stale .last after switching to a no-validation run.
            resume_path = max(
                candidates,
                key=lambda path: (path.stat().st_mtime_ns, path == latest_path),
            )
    ckpt_blob = None
    if resume_path is not None:
        agent, ckpt_blob = load_checkpoint(resume_path, cfg, device, restore_rng=True)
        print(f"resume {resume_path}  (pass --reset to wipe weights and start over)", flush=True)
        print("resume: overlapping weights loaded; new heads (WHL / extra events) start cold", flush=True)
    else:
        agent = AtomSpikeAgent(cfg).to(device)
        if reset and out_path.is_file():
            print(f"reset: ignoring existing {out_path}", flush=True)
        else:
            print("init: no checkpoint at --out, training from scratch", flush=True)
    agent = maybe_compile(agent, cfg)
    n_epochs = cfg.train.epochs if epochs is None else epochs
    accum = max(1, cfg.train.grad_accum)
    steps_per_epoch = max(1, (len(loader) + accum - 1) // accum)
    resume_extra = (ckpt_blob or {}).get("extra") or {}
    prior_stage_epoch = int(resume_extra.get("stage_epoch") or 0)
    prior_stage_epochs = int(resume_extra.get("stage_epochs") or 0)
    prior_steps_per_epoch = int(resume_extra.get("stage_steps_per_epoch") or 0)
    prior_stage_complete = bool(resume_extra.get("stage_complete", True))
    resume_stage = bool(
        ckpt_blob is not None
        and not prior_stage_complete
        and 0 < prior_stage_epoch < int(n_epochs)
        and prior_stage_epochs == int(n_epochs)
        and prior_steps_per_epoch == int(steps_per_epoch)
        and (cfg.train.lr_schedule != "wsd" or "scheduler" in ckpt_blob)
    )
    start_epoch = prior_stage_epoch if resume_stage else 0
    total_epochs_before = int(resume_extra.get("training_epochs_total") or 0)
    if resume_stage:
        stage_base_epoch = max(0, total_epochs_before - prior_stage_epoch)
        stage_index = int(resume_extra.get("stage_index") or 1)
        print(
            f"resume: exact stage {stage_index} epoch {start_epoch}/{n_epochs} from {resume_path}",
            flush=True,
        )
    else:
        stage_base_epoch = total_epochs_before
        stage_index = int(resume_extra.get("stage_index") or 0) + 1
        if ckpt_blob is not None:
            print(f"resume: starting fine-tune stage {stage_index} for {n_epochs} epochs", flush=True)
    opt = build_optimizer(cfg.train, agent.parameters())
    fresh_lrs = [float(group["lr"]) for group in opt.param_groups]
    scaler = build_grad_scaler(device, cfg.train.precision)
    if ckpt_blob is not None and "optimizer" in ckpt_blob:
        try:
            opt.load_state_dict(ckpt_blob["optimizer"])
        except (ValueError, RuntimeError):
            print("resume: optimizer state incompatible, keeping loaded weights only", flush=True)
    if not resume_stage:
        # A completed WSD stage ends at zero LR.  Optimizer momentum is useful
        # across fine-tune stages, but the old terminal LR is not.
        for group, lr in zip(opt.param_groups, fresh_lrs):
            group["lr"] = lr
    if ckpt_blob is not None and scaler.is_enabled() and "scaler" in ckpt_blob:
        try:
            scaler.load_state_dict(ckpt_blob["scaler"])
        except (KeyError, ValueError, RuntimeError):
            print("resume: fp16 scaler state incompatible, using a fresh scale", flush=True)
    total_steps = n_epochs * steps_per_epoch
    sched = build_scheduler_if_needed(opt, cfg, total_steps)
    if resume_stage and sched is not None and ckpt_blob is not None:
        try:
            sched.load_state_dict(ckpt_blob["scheduler"])
        except (ValueError, RuntimeError):
            print("resume: scheduler state incompatible, restarting schedule", flush=True)
            for group, lr in zip(opt.param_groups, fresh_lrs):
                group["lr"] = lr
            sched = build_scheduler_if_needed(opt, cfg, total_steps)
            start_epoch = 0
            stage_base_epoch = total_epochs_before
            stage_index += 1
            resume_stage = False
    history: list[float] = [float(v) for v in resume_extra.get("bc_loss", [])]
    val_history: list[float] = [float(v) for v in resume_extra.get("val_loss", [])]
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
    best_val = float(resume_extra.get("best_val_loss", float("inf")))
    best_epoch = int(resume_extra.get("best_epoch", -1))
    extra = {
        "bc_loss": history,
        "val_loss": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }
    for epoch in range(start_epoch, n_epochs):
        running = 0.0
        n = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loader, desc=f"bc {epoch+1}/{n_epochs}", leave=False)):
            frames = batch["frame"].to(device, non_blocking=True)
            prev = batch["prev_frame"].to(device, non_blocking=True)
            state = batch["game_state"].to(device, non_blocking=True)
            look_target = batch["look_concept"].to(device, non_blocking=True)
            effect_target = batch["effect_target"].to(device, non_blocking=True)
            effect_valid = batch["effect_valid"].to(device, non_blocking=True)
            frames, prev = maybe_augment_frames(
                frames, prev, cfg.train, game_state=state, protect_crosshair=look_target >= 0
            )
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
                    **task_forward_kwargs(batch, device),
                    **inventory_forward_kwargs(batch, device),
                    **block_pulse_forward_kwargs(batch, device),
                )
                divisor = grad_accum_divisor(step, len(loader), accum)
                loss = _total_loss(
                    out, tokens, w, slot_w, class_w, cfg, plan_batch,
                    spec=agent.spec, game_state=state, look_target=look_target,
                    effect_target=effect_target, effect_valid=effect_valid,
                    effect_active=effect_active, effect_pos_weight=effect_pos_weight,
                    train_visual_baseline=True,
                ) / divisor
            backward_loss(loss, scaler)
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                optimizer_step(opt, sched, cfg.train.grad_clip, params, scaler=scaler)
                opt.zero_grad(set_to_none=True)
            running += float(loss.item()) * frames.size(0) * divisor
            n += frames.size(0)
        avg = running / max(1, n)
        history.append(avg)
        stage_done = epoch + 1
        total_done = stage_base_epoch + stage_done
        extra.update(
            {
                "stage_index": stage_index,
                "stage_epoch": stage_done,
                "stage_epochs": int(n_epochs),
                "stage_steps_per_epoch": int(steps_per_epoch),
                "stage_complete": stage_done >= int(n_epochs),
                "training_epochs_total": total_done,
            }
        )
        msg = f"epoch {epoch+1}/{n_epochs}  train={avg:.4f}"
        if val_loader is not None:
            metrics = _eval_loader(
                agent, val_loader, device, cfg, slot_w, class_w,
                effect_active=effect_active, effect_pos_weight=effect_pos_weight, effect_prior=effect_prior,
            )
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
            if metrics["look_n"]:
                msg += (
                    f"  look_acc={metrics['look_acc']:.3f}"
                    f" macro_f1={metrics['look_macro_f1']:.3f}"
                    f" majority={metrics['look_majority_acc']:.3f} n_look={metrics['look_n']}"
                )
            if metrics["effect_n"]:
                msg += (
                    f"  effect_Brier={metrics['effect_brier']:.3f}"
                    f" visual_only={metrics['effect_visual_brier']:.3f}"
                    f" prior={metrics['effect_prior_brier']:.3f} n_effect={metrics['effect_n']}"
                )
                for event_name, event in metrics["effect_by_event"].items():
                    msg += (
                        f"  {event_name}[+{event['positive']}/-{event['negative']}]="
                        f"{event['conditioned']:.3f}/{event['visual_only']:.3f}/{event['prior']:.3f}"
                    )
            if metrics["loss"] < best_val:
                best_val = float(metrics["loss"])
                best_epoch = total_done
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
                    scaler=scaler,
                    keep_k=cfg.train.ckpt_keep_k,
                    verify=cfg.train.ckpt_verify,
                )
            # Latest state is distinct from the best validation model and is
            # the only file used for an exact continuation.
            save_checkpoint(
                agent,
                latest_path,
                extra=extra,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                keep_k=None,
                verify=cfg.train.ckpt_verify,
            )
        else:
            save_checkpoint(
                agent,
                out_path,
                extra=extra,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                keep_k=cfg.train.ckpt_keep_k,
                verify=cfg.train.ckpt_verify,
            )
        tqdm.write(msg)
    return {
        "checkpoint": str(out_path),
        "latest_checkpoint": str(latest_path) if val_loader is not None else str(out_path),
        "loss": history,
        "val_loss": val_history,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_epoch < 0 else best_val,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds) if val_ds is not None else 0),
        "n_sessions": int(getattr(ds, "n_sessions", 1) or 1),
        "session_lengths": list(getattr(ds, "session_lengths", []) or []),
        "n_recordings": int(getattr(ds, "n_recordings", 1) or 1),
        "recording_lengths": list(getattr(ds, "recording_lengths", []) or []),
        "params": agent.parameter_count(),
        "mouse_bins": int(agent.spec.mouse_bins),
        "resumed": resume_path is not None,
        "resumed_stage": resume_stage,
        "stage_index": stage_index,
        "training_epochs_total": stage_base_epoch + int(n_epochs),
        "reset": bool(reset),
    }
