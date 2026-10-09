"""10Hz Minecraft iterator: spatial grid, task semantics, temporal GRU.

The CNN/reasoner stay frozen during live play. This module is the cheap
plastic loop that (1) keeps a 4x6 map of frozen visual tokens and their
frame-to-frame change, (2) embeds the current advancement as the working
task, and (3) carries a GRU memory across perception ticks. It emits a
residual bias the 30Hz policy can read without another transformer pass.

Inventory / pause overlays are held out of the spatial map and the visual
side of the GRU: last world cells stay put, pixel delta is zeroed, and
pooled CNN features are not mixed in. Task aux / game_state still update.
GUI mouse goes through InventorySense instead.

Bias out is zero-initialized so old checkpoints behave as before until
the iterator actually learns.
"""

from __future__ import annotations

import zlib
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import IteratorConfig
from atomspike.models.activations import Act

TASK_AUX_DIM = 8

# Vanilla story / nether / end / adventure / husbandry ids the sidecar uses.
NAMED_TASKS: tuple[str, ...] = (
    "",
    "story/root",
    "story/mine_stone",
    "story/upgrade_tools",
    "story/smelt_iron",
    "story/obtain_armor",
    "story/lava_bucket",
    "story/iron_tools",
    "story/deflect_arrow",
    "story/form_obsidian",
    "story/enter_the_nether",
    "story/follow_ender_eye",
    "story/enter_the_end",
    "nether/root",
    "nether/find_fortress",
    "nether/obtain_blaze_rod",
    "nether/obtain_crying_obsidian",
    "nether/charge_respawn_anchor",
    "nether/brew_potion",
    "nether/summon_wither",
    "end/root",
    "end/kill_dragon",
    "end/enter_end_gateway",
    "end/find_end_city",
    "adventure/root",
    "adventure/sleep_in_bed",
    "adventure/kill_a_mob",
    "adventure/shoot_arrow",
    "adventure/trade",
    "husbandry/root",
    "husbandry/plant_seed",
    "husbandry/tame_an_animal",
    "husbandry/breed_an_animal",
)

_NAMED_INDEX = {name: i for i, name in enumerate(NAMED_TASKS)}


def normalize_adv(name: str | None) -> str:
    s = str(name or "").strip().lower()
    if s.startswith("minecraft:"):
        s = s[10:]
    return s


def task_id_from_name(name: str | None, n_buckets: int = 16) -> int:
    key = normalize_adv(name)
    if key in _NAMED_INDEX:
        return int(_NAMED_INDEX[key])
    if not key:
        return 0
    n_named = len(NAMED_TASKS)
    n_buckets = max(1, int(n_buckets))
    return n_named + (zlib.crc32(key.encode("utf-8")) % n_buckets)


def task_label_from_id(task_id: int, n_buckets: int = 16) -> str:
    """Short name for the live window; unknown ids collapse to a bucket label."""
    tid = int(task_id)
    if 0 <= tid < len(NAMED_TASKS):
        name = NAMED_TASKS[tid]
        return (name.split("/")[-1] if name else "空任务")
    extra = tid - len(NAMED_TASKS)
    n_buckets = max(1, int(n_buckets))
    if extra >= 0:
        return f"其它#{extra % n_buckets}"
    return "空任务"


def task_aux_from_signal(signal: dict[str, Any] | None) -> list[float]:
    raw = signal or {}

    def _b(key: str) -> float:
        v = raw.get(key)
        if isinstance(v, str):
            return 1.0 if v.strip().lower() in {"1", "true", "yes"} else 0.0
        return 1.0 if bool(v) else 0.0

    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(raw[key]) if key in raw and raw[key] is not None else default
        except (TypeError, ValueError):
            return default

    gui = str(raw.get("gui") or "").lower()
    return [
        max(0.0, min(1.0, _f("adv_next_frac"))),
        1.0 if (_b("gui_open") or "inventory" in gui) else 0.0,
        1.0 if (_b("paused") or "pause" in gui) else 0.0,
        1.0 if (_b("inv_open") or "inventory" in gui) else 0.0,
        max(0.0, min(1.0, _f("blocks_seen") / 32.0)),
        1.0 if (_b("dead") or _f("hp", 20.0) <= 0.0) else 0.0,
        1.0 if _b("look_miss") else 0.0,
        max(0.0, min(1.0, _f("adv_done") / 40.0)),
    ]


def task_tensors_from_signal(
    signal: dict[str, Any] | None,
    *,
    device: torch.device,
    batch: int = 1,
    n_buckets: int = 16,
) -> tuple[Tensor, Tensor]:
    tid = task_id_from_name(None if signal is None else signal.get("adv_next"), n_buckets)
    aux = task_aux_from_signal(signal)
    task_id = torch.full((max(1, int(batch)),), int(tid), device=device, dtype=torch.long)
    task_aux = torch.tensor(aux, device=device, dtype=torch.float32).unsqueeze(0).expand(max(1, int(batch)), -1).contiguous()
    return task_id, task_aux


def task_forward_kwargs(batch: dict, device: torch.device) -> dict[str, Tensor]:
    """Pull iterator task tensors off a demo batch if present."""
    out: dict[str, Tensor] = {}
    if "task_id" in batch:
        out["task_id"] = batch["task_id"].to(device, non_blocking=True)
    if "task_aux" in batch:
        out["task_aux"] = batch["task_aux"].to(device, non_blocking=True)
    return out


def as_hold_mask(mask: Tensor | None, batch: int, ref: Tensor) -> Tensor | None:
    """Broadcast a GUI/pause hold flag to [B] bool, or None if inactive."""
    if mask is None:
        return None
    m = mask.reshape(-1).to(device=ref.device)
    if m.dtype != torch.bool:
        m = m > 0.5
    if int(m.numel()) == 1 and batch > 1:
        m = m.expand(batch)
    elif int(m.numel()) != batch:
        m = m[:1].expand(batch) if int(m.numel()) else ref.new_zeros(batch, dtype=torch.bool)
    return m.contiguous()


def overlay_hold_from_aux(task_aux: Tensor | None, batch: int, ref: Tensor) -> Tensor:
    """task_aux[:, 1]=gui, [:, 2]=paused. Missing aux -> no hold."""
    hold = ref.new_zeros(batch, dtype=torch.bool)
    if task_aux is None:
        return hold
    aux = task_aux.float()
    if aux.dim() == 1:
        aux = aux.unsqueeze(0)
    if aux.size(0) != batch:
        aux = aux[:1].expand(batch, -1) if aux.size(0) >= 1 else aux.new_zeros(batch, int(aux.size(-1)))
    if aux.size(-1) > 1:
        hold = hold | (aux[:, 1] > 0.5)
    if aux.size(-1) > 2:
        hold = hold | (aux[:, 2] > 0.5)
    return hold


def spatial_grid(cfg) -> tuple[int, int]:
    """(height, width) cells. Prefer spatial_h/w; else a square spatial_hw."""
    h = int(getattr(cfg, "spatial_h", 0) or 0)
    w = int(getattr(cfg, "spatial_w", 0) or 0)
    if h >= 2 and w >= 2:
        return h, w
    side = max(2, int(getattr(cfg, "spatial_hw", 4) or 4))
    return side, side


def tokens_to_grid(tokens: Tensor, hw: tuple[int, int] | None, size: int | tuple[int, int]) -> Tensor:
    """[B, N, D] visual tokens -> [B, H*W, D] spatial cells."""
    if isinstance(size, (tuple, list)):
        out_h, out_w = max(1, int(size[0])), max(1, int(size[1]))
    else:
        out_h = out_w = max(1, int(size))
    b, n, d = tokens.shape
    if hw is not None:
        gh, gw = int(hw[0]), int(hw[1])
        if gh > 0 and gw > 0 and gh * gw == n:
            grid = tokens.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()
            pooled = F.adaptive_avg_pool2d(grid, (out_h, out_w))
            return pooled.flatten(2).transpose(1, 2).contiguous()
    side = int(n ** 0.5)
    if side * side == n and side > 1:
        grid = tokens.view(b, side, side, d).permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(grid, (out_h, out_w))
        return pooled.flatten(2).transpose(1, 2).contiguous()
    pooled = tokens.mean(dim=1, keepdim=True)
    return pooled.expand(b, out_h * out_w, d).contiguous()


class MinecraftIterator(nn.Module):
    """One GRU step of spatial + task + state. Returns a d_model residual bias."""

    def __init__(self, cfg: IteratorConfig, d_model: int, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(d_model)
        self.hidden = max(16, int(cfg.hidden))
        self.spatial_h, self.spatial_w = spatial_grid(cfg)
        self.spatial_hw = self.spatial_h
        self.n_cells = self.spatial_h * self.spatial_w
        self.spatial_dim = max(8, int(cfg.spatial_dim))
        self.n_named = len(NAMED_TASKS)
        self.n_buckets = max(1, int(cfg.n_task_buckets))
        n_tasks = self.n_named + self.n_buckets
        self.cell_proj = nn.Linear(self.d_model, self.spatial_dim)
        self.spatial_in = nn.Linear(self.spatial_dim * 2, self.hidden)
        self.task_embed = nn.Embedding(n_tasks, max(8, int(cfg.task_dim)))
        self.task_in = nn.Sequential(
            nn.Linear(max(8, int(cfg.task_dim)) + TASK_AUX_DIM, self.hidden),
            Act("gelu"),
        )
        self.state_in = nn.Linear(int(state_dim), self.hidden)
        self.pooled_in = nn.Linear(self.d_model, self.hidden)
        self.mix = nn.Sequential(
            nn.Linear(self.hidden * 4, self.hidden),
            Act("gelu"),
        )
        self.gru = nn.GRUCell(self.hidden, self.hidden)
        self.bias_out = nn.Linear(self.hidden, self.d_model)
        self.change_head = nn.Linear(self.hidden, 1)
        nn.init.zeros_(self.bias_out.weight)
        nn.init.zeros_(self.bias_out.bias)
        nn.init.zeros_(self.change_head.weight)
        nn.init.zeros_(self.change_head.bias)

    def n_task_ids(self) -> int:
        return self.n_named + self.n_buckets

    def step(
        self,
        grid: Tensor,
        pooled: Tensor,
        state: Tensor,
        task_id: Tensor,
        task_aux: Tensor,
        h: Tensor | None = None,
        prev_cells: Tensor | None = None,
        gui_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b = pooled.size(0)
        cells_now = self.cell_proj(grid)
        if cells_now.size(0) != b:
            cells_now = (
                cells_now[:1].expand(b, -1, -1).contiguous()
                if cells_now.size(0) >= 1
                else cells_now.new_zeros(b, cells_now.size(1), cells_now.size(2))
            )
        prev = cells_now if prev_cells is None else prev_cells
        if prev.shape != cells_now.shape:
            prev = cells_now
        hold = as_hold_mask(gui_mask, b, pooled)
        if hold is not None:
            keep = (1.0 - hold.to(dtype=cells_now.dtype)).view(b, 1, 1)
            cells = cells_now * keep + prev * (1.0 - keep)
            delta = (cells_now - prev) * keep
            vis = (1.0 - hold.to(dtype=pooled.dtype)).view(b, 1)
        else:
            cells = cells_now
            delta = cells_now - prev
            vis = None
        spatial = self.spatial_in(torch.cat([cells.mean(dim=1), delta.abs().mean(dim=1)], dim=-1))
        tid = task_id.reshape(-1).to(dtype=torch.long)
        if tid.size(0) != b:
            tid = tid[:1].expand(b) if tid.numel() else pooled.new_zeros(b, dtype=torch.long)
        tid = tid.clamp(0, self.n_task_ids() - 1)
        aux = task_aux.float()
        if aux.dim() == 1:
            aux = aux.unsqueeze(0)
        if aux.size(0) != b:
            aux = aux[:1].expand(b, -1) if aux.size(0) >= 1 else aux.new_zeros(b, TASK_AUX_DIM)
        if aux.size(-1) != TASK_AUX_DIM:
            pad = aux.new_zeros(b, TASK_AUX_DIM)
            n = min(TASK_AUX_DIM, int(aux.size(-1)))
            pad[:, :n] = aux[:, :n]
            aux = pad
        task = self.task_in(torch.cat([self.task_embed(tid), aux], dim=-1))
        pooled_feat = self.pooled_in(pooled)
        if vis is not None:
            spatial = spatial * vis
            pooled_feat = pooled_feat * vis
        x = self.mix(
            torch.cat(
                [
                    spatial,
                    task,
                    self.state_in(state.float()),
                    pooled_feat,
                ],
                dim=-1,
            )
        )
        if h is None or h.shape != (b, self.hidden):
            h = x.new_zeros(b, self.hidden)
        h_new = self.gru(x, h)
        change_logit = self.change_head(h_new).squeeze(-1)
        change = torch.sigmoid(change_logit)
        if vis is not None:
            change = change * vis.squeeze(-1)
        return {
            "h": h_new,
            "bias": self.bias_out(h_new),
            "cells": cells,
            "change_logit": change_logit,
            "change": change,
        }

    def forward(
        self,
        grid: Tensor,
        pooled: Tensor,
        state: Tensor,
        task_id: Tensor,
        task_aux: Tensor,
        h: Tensor | None = None,
        prev_cells: Tensor | None = None,
        gui_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.step(
            grid,
            pooled,
            state,
            task_id,
            task_aux,
            h=h,
            prev_cells=prev_cells,
            gui_mask=gui_mask,
        )
