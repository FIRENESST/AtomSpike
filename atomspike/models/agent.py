"""Full AtomSpike agent: encoder + reasoner (10Hz) + residual + policy (30Hz)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from atomspike.capture.mc_state import MC_STATE_FIELDS
from atomspike.config import AtomSpikeConfig
from atomspike.models.actions import ActionSpec
from atomspike.models.encoder import VisualEncoder
from atomspike.models.grow import GrowBank
from atomspike.models.iterator import (
    MinecraftIterator,
    TASK_AUX_DIM,
    overlay_hold_from_aux,
    task_label_from_id,
    tokens_to_grid,
)
from atomspike.models.curl_scene import SpikeCurlScene, overlay_field
from atomspike.models.group_attn import SpikeGroupCoalition
from atomspike.models.sea import (
    ScreenEventCoincidenceAttention,
    apply_innate_block_pulse,
    overlay_events,
)
from atomspike.models.grid_detect import SCREEN_KINDS, SpikeGridDetect, overlay_cells
from atomspike.models.event_seq import GlobalEventSequence
from atomspike.models.thought import SpikeThought
from atomspike.models.semantic_probe import CrosshairConceptProbe
from atomspike.models.effect_probe import ConditionalEffectProbe
from atomspike.models.inventory import InventorySense, item_label_from_id
from atomspike.models.planner import SemanticPlanner
from atomspike.models.policy_ann import ANNPolicy
from atomspike.models.policy_spike import SpikePolicy
from atomspike.models.reasoner import SemanticReasoner
from atomspike.models.temporal import TemporalResidualAdapter
from atomspike.data.sessions import CLOCK_DIM
from atomspike.models.activations import Act


def _num(value, default: float = 0.0) -> float:
    """First finite number. Materialized traces are often length-1 arrays."""
    if value is None or isinstance(value, bool):
        return float(default)
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size == 0:
            return float(default)
        value = flat[0]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


class AtomSpikeAgent(nn.Module):
    def __init__(self, cfg: AtomSpikeConfig):
        super().__init__()
        self.cfg = cfg
        self.spec = ActionSpec.from_config(cfg.action)
        self.encoder = VisualEncoder(cfg.encoder)
        # Screen-only probe. Its logits have no semantic meaning until trained
        # on a recording with explicit look_concept labels and evaluated.
        self.semantic_probe = CrosshairConceptProbe(cfg.encoder.d_model)
        self.effect_probe = ConditionalEffectProbe(cfg.encoder.d_model)
        # Equal-capacity action-blind control; diagnostic only, never policy input.
        self.effect_visual_baseline = ConditionalEffectProbe(cfg.encoder.d_model)
        self.reasoner = SemanticReasoner(cfg.reasoner)
        self.adapter = TemporalResidualAdapter(cfg.adapter)
        if cfg.policy.kind == "spike":
            self.policy: nn.Module = SpikePolicy(cfg.policy, self.spec)
        elif cfg.policy.kind == "ann":
            self.policy = ANNPolicy(cfg.policy, self.spec)
        else:
            raise ValueError(f"unknown policy kind: {cfg.policy.kind}")
        self.planner = (
            SemanticPlanner(cfg.planner, cfg.encoder.d_model, cfg.reasoner.game_state_dim)
            if cfg.planner.enabled
            else None
        )
        self.iterator = (
            MinecraftIterator(cfg.iterator, cfg.encoder.d_model, cfg.reasoner.game_state_dim)
            if cfg.iterator.enabled
            else None
        )
        self.inventory = (
            InventorySense(cfg.inventory, cfg.encoder.d_model, cfg.reasoner.game_state_dim)
            if cfg.inventory.enabled
            else None
        )
        self.scene = (
            SpikeCurlScene(cfg.scene, cfg.encoder.d_model)
            if cfg.scene.enabled
            else None
        )
        self.groups = (
            SpikeGroupCoalition(cfg.groups, cfg.encoder.d_model)
            if cfg.groups.enabled
            else None
        )
        self.sea = (
            ScreenEventCoincidenceAttention(cfg.sea, cfg.encoder.d_model)
            if cfg.sea.enabled
            else None
        )
        self.detect = (
            SpikeGridDetect(cfg.detect, cfg.encoder.d_model)
            if getattr(cfg.detect, "enabled", True)
            else None
        )
        self.event_seq = (
            GlobalEventSequence(
                cfg.event_seq,
                cfg.encoder.d_model,
                event_dim=int(cfg.planner.latent_dim) if cfg.planner.enabled else cfg.encoder.d_model,
            )
            if cfg.event_seq.enabled
            else None
        )
        self.thought = (
            SpikeThought(cfg.thought, cfg.encoder.d_model)
            if getattr(cfg.thought, "enabled", True)
            else None
        )
        self.grow = GrowBank(
            cfg.encoder.d_model,
            cfg.reasoner.game_state_dim,
            max_layers=int(cfg.learn.max_new_layers),
            hidden=int(cfg.learn.layer_hidden),
        )
        d = cfg.encoder.d_model
        self.clock_in = nn.Sequential(
            nn.Linear(CLOCK_DIM, d),
            Act("gelu"),
            nn.Linear(d, d),
        )
        self._context: Tensor | None = None
        self._plan_bias: Tensor | None = None
        self._iter_h: Tensor | None = None
        self._iter_cells: Tensor | None = None
        self._iter_bias: Tensor | None = None
        self._last_iter_h: Tensor | None = None
        self._last_iter_h_in: Tensor | None = None
        self._last_iter_grid: Tensor | None = None
        self._last_iter_prev_cells: Tensor | None = None
        self._last_task_id: Tensor | None = None
        self._last_task_aux: Tensor | None = None
        self._iter_world_grid: Tensor | None = None
        self._iter_world_pooled: Tensor | None = None
        self._last_iter: dict | None = None
        self._overlay_iter = None
        self._chg_ema = None
        self._inv_mem: Tensor | None = None
        self._inv_bias: Tensor | None = None
        self._last_inv_ids: Tensor | None = None
        self._last_hand_id: Tensor | None = None
        self._last_hover_id: Tensor | None = None
        self._last_carry_id: Tensor | None = None
        self._last_inv_mem_in: Tensor | None = None
        self._last_inv: dict | None = None
        self._curl_mem: Tensor | None = None
        self._curl_bias: Tensor | None = None
        self._last_curl_grid: Tensor | None = None
        self._last_curl_mem_in: Tensor | None = None
        self._last_curl: dict | None = None
        self._overlay_curl: dict | None = None
        self._group_bias: Tensor | None = None
        self._group_mem: Tensor | None = None
        self._group_mem_q: Tensor | None = None
        self._group_mem_k: Tensor | None = None
        self._last_group_mem_in: Tensor | None = None
        self._last_group_mem_q_in: Tensor | None = None
        self._last_group_mem_k_in: Tensor | None = None
        self._last_group_energy: Tensor | None = None
        self._last_group: dict | None = None
        self._sea_bias: Tensor | None = None
        self._sea_mem: Tensor | None = None
        self._sea_mem_q: Tensor | None = None
        self._sea_mem_k: Tensor | None = None
        self._sea_prev_spike: Tensor | None = None
        self._last_sea_map: Tensor | None = None
        self._last_sea_mem_in: Tensor | None = None
        self._last_sea_mem_q_in: Tensor | None = None
        self._last_sea_mem_k_in: Tensor | None = None
        self._last_sea_prev_spike: Tensor | None = None
        self._last_sea: dict | None = None
        self._overlay_sea: dict | None = None
        self._last_block_pulse: Tensor | None = None
        self._detect_bias: Tensor | None = None
        self._detect_mem: Tensor | None = None
        self._last_detect_mem_in: Tensor | None = None
        self._last_detect: dict | None = None
        self._overlay_detect: dict | None = None
        self._event_bias: Tensor | None = None
        self._event_slots: Tensor | None = None
        self._event_ages: Tensor | None = None
        self._event_mem_q: Tensor | None = None
        self._event_mem_k: Tensor | None = None
        self._last_event_slots: Tensor | None = None
        self._last_event_ages: Tensor | None = None
        self._last_event_mem_q: Tensor | None = None
        self._last_event_mem_k: Tensor | None = None
        self._last_event_tokens: Tensor | None = None
        self._last_event: dict | None = None
        self._inv_pointer_logits: Tensor | None = None
        self._activity_trace = False
        self._overlay_trace = True
        self._last_plan: dict | None = None
        self._last_pooled: Tensor | None = None
        self._concept_logits: Tensor | None = None
        self._last_fused_base: Tensor | None = None
        self._thought_mem: Tensor | None = None
        self._thought_bias: Tensor | None = None
        self._last_gui_live: Tensor | None = None

    def set_overlay_trace(self, enabled: bool) -> None:
        """HUD off -> skip the per-tick GPU->CPU overlay snapshots entirely."""
        self._overlay_trace = bool(enabled)
        if not enabled:
            self._overlay_iter = None
            self._overlay_curl = None
            self._overlay_sea = None
            self._overlay_detect = None

    def set_activity_trace(self, enabled: bool) -> None:
        self._activity_trace = bool(enabled)
        if hasattr(self.policy, "trace_enabled"):
            self.policy.trace_enabled = bool(enabled)
        if not enabled:
            self._last_plan = None
            self._last_iter = None
            self._last_curl = None
            self._last_group = None
            self._last_sea = None
            self._last_event = None
            if hasattr(self.policy, "last_trace"):
                self.policy.last_trace = None

    def reset_runtime(self) -> None:
        self._context = None
        self._plan_bias = None
        self._last_pooled = None
        self._concept_logits = None
        self._last_fused_base = None
        self._last_policy_input = None
        self._thought_mem = None
        self._thought_bias = None
        self.reset_iterator()
        self.reset_inventory()
        self.reset_scene()
        self.reset_groups()
        self.reset_sea()
        self.reset_detect()
        self.reset_event_seq()
        reset = getattr(self.policy, "reset_state", None)
        if callable(reset):
            reset()

    def reset_iterator(self) -> None:
        self._iter_h = None
        self._iter_cells = None
        self._iter_bias = None
        self._last_iter_h = None
        self._last_iter_h_in = None
        self._last_iter_grid = None
        self._last_iter_prev_cells = None
        self._last_task_id = None
        self._last_task_aux = None
        self._iter_world_grid = None
        self._iter_world_pooled = None
        self._last_iter = None
        self._overlay_iter = None
        self._chg_ema = None

    def reset_inventory(self) -> None:
        self._inv_mem = None
        self._inv_bias = None
        self._last_inv_ids = None
        self._last_hand_id = None
        self._last_hover_id = None
        self._last_carry_id = None
        self._last_inv_mem_in = None
        self._last_inv = None

    def reset_scene(self) -> None:
        self._curl_mem = None
        self._curl_bias = None
        self._last_curl_grid = None
        self._last_curl_mem_in = None
        self._last_curl = None
        self._overlay_curl = None

    def reset_groups(self) -> None:
        self._group_bias = None
        self._group_mem = None
        self._group_mem_q = None
        self._group_mem_k = None
        self._last_group_mem_in = None
        self._last_group_mem_q_in = None
        self._last_group_mem_k_in = None
        self._last_group_energy = None
        self._last_group = None

    def reset_sea(self) -> None:
        self._sea_bias = None
        self._sea_mem = None
        self._sea_mem_q = None
        self._sea_mem_k = None
        self._sea_prev_spike = None
        self._last_sea_map = None
        self._last_sea_mem_in = None
        self._last_sea_mem_q_in = None
        self._last_sea_mem_k_in = None
        self._last_sea_prev_spike = None
        self._last_sea = None
        self._last_block_pulse = None
        self._overlay_sea = None

    def reset_detect(self) -> None:
        self._detect_bias = None
        self._detect_mem = None
        self._last_detect_mem_in = None
        self._last_detect = None
        self._overlay_detect = None

    def reset_event_seq(self) -> None:
        self._event_bias = None
        self._event_slots = None
        self._event_ages = None
        self._event_mem_q = None
        self._event_mem_k = None
        self._last_event_slots = None
        self._last_event_ages = None
        self._last_event_mem_q = None
        self._last_event_mem_k = None
        self._last_event_tokens = None
        self._last_event = None

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _gui_open(self, game_state: Tensor) -> Tensor:
        if "gui_open" not in MC_STATE_FIELDS:
            return game_state.new_zeros(game_state.size(0), dtype=torch.bool)
        idx = MC_STATE_FIELDS.index("gui_open")
        if idx >= game_state.size(-1):
            return game_state.new_zeros(game_state.size(0), dtype=torch.bool)
        return game_state[:, idx] > 0.5

    def _overlay_hold(self, game_state: Tensor, task_aux: Tensor | None) -> Tensor:
        """True = inventory/pause overlay; freeze world spatial/temporal perception."""
        hold = self._gui_open(game_state)
        return hold | overlay_hold_from_aux(task_aux, int(game_state.size(0)), game_state)

    @staticmethod
    def _flag_any(mask: Tensor, known: bool | None) -> bool:
        if known is not None:
            return bool(known)
        return bool(mask.any().item())

    def _pointer_feat(self, game_state: Tensor) -> Tensor:
        feat = game_state.new_zeros(game_state.size(0), 4)
        for i, name in enumerate(("cursor_x", "cursor_y", "gui_open", "hovered_slot")):
            if name in MC_STATE_FIELDS:
                j = MC_STATE_FIELDS.index(name)
                if j < game_state.size(-1):
                    feat[:, i] = game_state[:, j]
        return feat

    def _planner_event_tokens(self, mix: Tensor) -> Tensor:
        planner = self.planner
        if planner is None or self.event_seq is None:
            return mix.new_zeros(mix.size(0), 1, 1)
        w = planner._task_weight().to(device=mix.device, dtype=mix.dtype)
        return mix.unsqueeze(-1) * w.unsqueeze(0)

    def _cursor_xy(self, game_state: Tensor) -> Tensor | None:
        if "cursor_x" not in MC_STATE_FIELDS:
            return None
        cx = MC_STATE_FIELDS.index("cursor_x")
        if cx + 1 >= game_state.size(-1):
            return None
        return game_state[:, cx : cx + 2]

    @staticmethod
    def _hold_update(prev: Tensor | None, new: Tensor, hold: Tensor, overlay: bool | None = None) -> Tensor | None:
        """Keep overlay rows from prev; reset if batch/shape changed."""
        if overlay is True:
            return prev if prev is not None and prev.shape == new.shape else None
        if overlay is False or prev is None or prev.shape != new.shape:
            return new
        view = [hold.size(0)] + [1] * (new.dim() - 1)
        return torch.where(hold.view(*view), prev, new)

    def _inventory_bias(
        self,
        game_state: Tensor,
        inv_ids: Tensor | None,
        hand_id: Tensor | None,
        hover_id: Tensor | None,
        carry_id: Tensor | None,
    ) -> Tensor:
        d = game_state.size(0)
        zeros = game_state.new_zeros(d, self.cfg.encoder.d_model)
        if self.inventory is None:
            return zeros
        persist = (not self.training) and bool(
            getattr(self.cfg.inventory, "persistent_inference", False)
        )
        mem = None if not persist else self._inv_mem
        if mem is not None and mem.size(0) != d:
            mem = None
        self._last_inv_mem_in = None if mem is None else mem.detach()
        out = self.inventory(inv_ids, hand_id, hover_id, carry_id, game_state, mem=mem)
        bias = out["bias"]
        self._inv_pointer_logits = out.get("pointer_logits")
        self._inv_mem = out["mem"].detach() if persist else None
        # Cache the current 10Hz observation for intervening 30Hz policy ticks;
        # this is not recurrent state and remains valid in stateless mode.
        self._inv_bias = bias.detach() if not self.training else None
        self._last_inv_ids = None if inv_ids is None else inv_ids.detach()
        self._last_hand_id = None if hand_id is None else hand_id.detach()
        self._last_hover_id = None if hover_id is None else hover_id.detach()
        self._last_carry_id = None if carry_id is None else carry_id.detach()
        if (not self.training) and self._activity_trace:
            self._pending_inv = {
                "hand_id": None if hand_id is None else hand_id.reshape(-1)[:1].detach(),
                "hover_id": None if hover_id is None else hover_id.reshape(-1)[:1].detach(),
                "carry_id": None if carry_id is None else carry_id.reshape(-1)[:1].detach(),
                "n_items": out["n_items"].reshape(-1)[:1].detach(),
                "spike": out["spike"][0].detach().float().mean(),
                "bias_rms": bias[0].detach().float().pow(2).mean().sqrt(),
            }
        return bias

    def fuse(
        self,
        frames: Tensor,
        game_state: Tensor,
        prev_frames: Tensor | None = None,
        reason_tick: bool = True,
        plan_sample: bool = False,
        explore: float = 0.0,
        plan_temperature: float = 1.0,
        future_state: Tensor | None = None,
        clock: Tensor | None = None,
        task_id: Tensor | None = None,
        task_aux: Tensor | None = None,
        inv_ids: Tensor | None = None,
        hand_id: Tensor | None = None,
        hover_id: Tensor | None = None,
        carry_id: Tensor | None = None,
        overlay: bool | None = None,
        gui_open: bool | None = None,
        block_pulse: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        if clock is None:
            clock_feat = torch.zeros(frames.size(0), CLOCK_DIM, device=frames.device, dtype=torch.float32)
        else:
            clock_feat = clock.float().reshape(frames.size(0), CLOCK_DIM)
        clock_vec = self.clock_in(clock_feat)
        hold = self._overlay_hold(game_state, task_aux)
        hold_any = self._flag_any(hold, overlay)
        residual = self.adapter(frames, prev_frames, hold_mask=hold, overlay=overlay)
        need_reason = reason_tick or self._context is None or self._context.size(0) != frames.size(0)
        plan_out = None
        pooled_out = None
        visual_tokens: Tensor | None = None
        visual_hw: tuple[int, int] | None = None
        if need_reason:
            tokens, pooled = self.encoder(frames)
            look_logits = self.semantic_probe(tokens, getattr(self.encoder, "last_hw", None))
            self._concept_logits = look_logits if self.training else look_logits.detach()
            amp = float(getattr(self.cfg.sea, "block_amp", 0.08) or 0.0)
            tokens = apply_innate_block_pulse(tokens, block_pulse, hw=getattr(self.encoder, "last_hw", None), amp=amp, hold=hold)
            visual_tokens = tokens
            visual_hw = getattr(self.encoder, "last_hw", None)
            pooled_out = pooled
            self._last_block_pulse = None if block_pulse is None else block_pulse.detach()
            plan_tokens = None
            plan_bias = residual.new_zeros(frames.size(0), residual.size(-1))
            if self.planner is not None:
                plan_out = self.planner(
                    pooled,
                    game_state,
                    sample=bool(self.training or plan_sample),
                    explore=0.0 if self.training else float(explore),
                    temperature=plan_temperature,
                    future_state=future_state,
                )
                plan_tokens = plan_out["tokens"]
                plan_bias = plan_out["bias"]
                if self._activity_trace and (not self.training):
                    mix = plan_out.get("mix")
                    z = plan_out.get("z")
                    self._last_plan = {
                        "mix": None if mix is None else mix[0].detach(),
                        "z": None if z is None else z[0].detach(),
                    }
                if (not self.training) and self._plan_bias is not None and self._plan_bias.size(0) == plan_bias.size(0):
                    plan_bias = 0.65 * self._plan_bias + 0.35 * plan_bias
            clock_tok = clock_vec.unsqueeze(1)
            extra = clock_tok if plan_tokens is None else torch.cat([plan_tokens, clock_tok], dim=1)
            ctx = self.reasoner(tokens, game_state, plan_tokens=extra)
            iter_bias = residual.new_zeros(frames.size(0), residual.size(-1))
            curl_bias = residual.new_zeros(frames.size(0), residual.size(-1))
            grid = None
            if self.iterator is not None or self.scene is not None:
                b = frames.size(0)
                hw = getattr(self.encoder, "last_hw", None)
                if self.iterator is not None:
                    grid_hw = (int(self.iterator.spatial_h), int(self.iterator.spatial_w))
                else:
                    grid_hw = (int(self.scene.spatial_h), int(self.scene.spatial_w))
                grid = tokens_to_grid(tokens, hw, grid_hw)
                # ``tokens`` already received the one-shot block tag before
                # pooling.  Applying it again here doubled the event strength.
                pooled_it = pooled
                if (not self.training) and hold_any:
                    g3 = hold.view(b, 1, 1)
                    if self._iter_world_grid is not None and self._iter_world_grid.shape == grid.shape:
                        grid = torch.where(g3, self._iter_world_grid, grid)
                    if self._iter_world_pooled is not None and self._iter_world_pooled.shape == pooled.shape:
                        pooled_it = torch.where(hold.view(b, 1), self._iter_world_pooled, pooled)
            if self.iterator is not None and grid is not None:
                if task_id is None:
                    task_id = tokens.new_zeros(b, dtype=torch.long)
                if task_aux is None:
                    task_aux = tokens.new_zeros(b, TASK_AUX_DIM)
                persist_iter = (not self.training) and bool(
                    getattr(self.cfg.iterator, "persistent_inference", False)
                )
                h_in = self._iter_h if persist_iter else None
                prev_cells = self._iter_cells if persist_iter else None
                it = self.iterator(
                    grid,
                    pooled_it,
                    game_state,
                    task_id,
                    task_aux,
                    h=h_in,
                    prev_cells=prev_cells,
                    gui_mask=hold if hold_any else None,
                )
                iter_bias = it["bias"]
                self._last_iter_grid = grid.detach()
                self._last_iter_h_in = None if h_in is None else h_in.detach()
                self._last_iter_prev_cells = None if prev_cells is None else prev_cells.detach()
                self._last_task_id = task_id.detach()
                self._last_task_aux = task_aux.float().detach()
                if self.training:
                    self._last_iter_h = it["h"]
                    self._iter_h = None
                    self._iter_cells = None
                    self._iter_bias = None
                    self._iter_world_grid = None
                    self._iter_world_pooled = None
                else:
                    self._iter_h = it["h"].detach() if persist_iter else None
                    self._iter_bias = iter_bias.detach()
                    self._iter_cells = self._hold_update(self._iter_cells, it["cells"].detach(), hold, overlay)
                    skip_world = overlay is True
                    if not skip_world:
                        self._iter_world_grid = self._hold_update(self._iter_world_grid, grid.detach(), hold, overlay)
                        self._iter_world_pooled = self._hold_update(self._iter_world_pooled, pooled.detach(), hold, overlay)
                    self._last_iter_h = self._last_iter_h_in
                    if self._overlay_trace or self._activity_trace:
                        cells = it["cells"]
                        prev = prev_cells if prev_cells is not None and prev_cells.shape == cells.shape else cells
                        dlt = (cells[0] - prev[0]).detach().float().abs().mean(dim=-1)
                        gh, gw = int(self.iterator.spatial_h), int(self.iterator.spatial_w)
                        self._pending_iter = {
                            "dlt": dlt.reshape(-1),
                            "gh": gh,
                            "gw": gw,
                            "change": it["change"].reshape(-1)[:1].detach(),
                            "h": it["h"][0].detach(),
                            "bias": iter_bias[0].detach(),
                            "task_aux": task_aux[0].detach(),
                            "task_id": task_id.reshape(-1)[:1].detach(),
                            "hold": bool(hold_any),
                        }
            if self.scene is not None and grid is not None:
                persist_scene = (not self.training) and bool(
                    getattr(self.cfg.scene, "persistent_inference", False)
                )
                mem_in = self._curl_mem if persist_scene else None
                self._last_curl_mem_in = None if mem_in is None else mem_in.detach()
                self._last_curl_grid = grid.detach()
                sc = self.scene(
                    grid,
                    mem=mem_in,
                    gui_mask=hold if hold_any else None,
                )
                curl_bias = sc["bias"]
                if self.training:
                    self._curl_mem = None
                    self._curl_bias = None
                else:
                    self._curl_mem = (
                        self._hold_update(self._curl_mem, sc["mem"].detach(), hold, overlay)
                        if persist_scene
                        else None
                    )
                    self._curl_bias = curl_bias.detach()
                    self._overlay_curl = overlay_field(sc) if self._overlay_trace else None
                    if self._activity_trace:
                        self._last_curl = {
                            "map": sc["curl"][0].detach(),
                            "energy": sc["energy"].reshape(-1)[:1].detach(),
                            "bias_rms": curl_bias[0].detach().float().pow(2).mean().sqrt(),
                            "hold": bool(hold_any),
                        }
            if self.training:
                self._context = ctx
                self._plan_bias = plan_bias
            else:
                self._context = ctx.detach()
                self._plan_bias = plan_bias.detach()
        else:
            ctx = self._context
            assert ctx is not None
            plan_bias = self._plan_bias
            if plan_bias is None:
                plan_bias = ctx.new_zeros(ctx.size(0), ctx.size(-1))
            iter_bias = self._iter_bias
            if iter_bias is None:
                iter_bias = ctx.new_zeros(ctx.size(0), ctx.size(-1))
            curl_bias = self._curl_bias
            if curl_bias is None:
                curl_bias = ctx.new_zeros(ctx.size(0), ctx.size(-1))
        inv_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.inventory is not None:
            gui = self._gui_open(game_state)
            run_inv = bool(need_reason or (gui_open if gui_open is not None else ((not self.training) and self._flag_any(gui, None))))
            if run_inv:
                inv_bias = self._inventory_bias(game_state, inv_ids, hand_id, hover_id, carry_id)
            else:
                self._inv_pointer_logits = None
                if self._inv_bias is not None and self._inv_bias.size(0) == frames.size(0):
                    inv_bias = self._inv_bias
        fused_base = ctx + residual + clock_vec
        pre = fused_base + plan_bias + iter_bias + inv_bias + curl_bias
        group_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.groups is not None:
            if need_reason:
                cache = not self.training
                persist = cache and bool(getattr(self.cfg.groups, "persistent_inference", False))
                mem = None if not persist else self._group_mem
                mem_q = None if not persist else self._group_mem_q
                mem_k = None if not persist else self._group_mem_k
                if mem is not None and mem.size(0) != frames.size(0):
                    mem = None
                    mem_q = None
                    mem_k = None
                self._last_group_mem_in = None if mem is None else mem.detach()
                self._last_group_mem_q_in = None if mem_q is None else mem_q.detach()
                self._last_group_mem_k_in = None if mem_k is None else mem_k.detach()
                go = self.groups.observe(pre, mem=mem, mem_q=mem_q, mem_k=mem_k)
                group_bias = go["bias"]
                self._last_group_energy = go["energy"].detach().mean(dim=0)
                self._group_mem = go["mem"].detach() if persist else None
                self._group_mem_q = go["mem_q"].detach() if persist else None
                self._group_mem_k = go["mem_k"].detach() if persist else None
                self._group_bias = group_bias.detach() if cache else None
                if cache and self._activity_trace:
                    self._last_group = {
                        "energy": go["energy"][0].detach(),
                        "n_alive": int(self.groups.n_alive()),
                        "n_groups": int(self.groups.n_groups),
                        "bias_rms": group_bias[0].detach().float().pow(2).mean().sqrt(),
                    }
            elif self._group_bias is not None and self._group_bias.size(0) == frames.size(0):
                group_bias = self._group_bias
        sea_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.sea is not None:
            if need_reason and visual_tokens is not None:
                cache = not self.training
                persist = cache and bool(getattr(self.cfg.sea, "persistent_inference", False))
                mem = None if not persist else self._sea_mem
                mem_q = None if not persist else self._sea_mem_q
                mem_k = None if not persist else self._sea_mem_k
                prev_spk = None if not persist else self._sea_prev_spike
                if mem is not None and mem.size(0) != frames.size(0):
                    mem = None
                    mem_q = None
                    mem_k = None
                    prev_spk = None
                self._last_sea_mem_in = None if mem is None else mem.detach()
                self._last_sea_mem_q_in = None if mem_q is None else mem_q.detach()
                self._last_sea_mem_k_in = None if mem_k is None else mem_k.detach()
                self._last_sea_prev_spike = None if prev_spk is None else prev_spk.detach()
                so = self.sea.observe(
                    visual_tokens,
                    pre,
                    hw=visual_hw,
                    mem=mem,
                    mem_q=mem_q,
                    mem_k=mem_k,
                    prev_spike=prev_spk,
                    gui_mask=hold if hold_any else None,
                    cursor_xy=self._cursor_xy(game_state),
                    cursor_w=self._gui_open(game_state).to(dtype=pre.dtype),
                    block_pulse=block_pulse,
                )
                sea_bias = so["bias"]
                self._last_sea_map = so["map"].detach()
                if cache:
                    self._sea_mem = so["mem"].detach() if persist else None
                    self._sea_mem_q = so["mem_q"].detach() if persist else None
                    self._sea_mem_k = so["mem_k"].detach() if persist else None
                    self._sea_prev_spike = so["spike"].detach() if persist else None
                    self._sea_bias = sea_bias.detach()
                    self._overlay_sea = overlay_events(so) if self._overlay_trace else None
                    if self._activity_trace:
                        self._last_sea = {
                            "mask": so["mask"][0].detach(),
                            "k": min(int(self.sea.k_events), int(so["mask"].size(1))),
                            "n": int(so["mask"].size(1)),
                            "energy": so["eventness"][0].detach().float().mean(),
                            "bias_rms": sea_bias[0].detach().float().pow(2).mean().sqrt(),
                        }
                else:
                    self._sea_mem = None
                    self._sea_mem_q = None
                    self._sea_mem_k = None
                    self._sea_prev_spike = None
                    self._sea_bias = None
                    self._overlay_sea = None
            elif self._sea_bias is not None and self._sea_bias.size(0) == frames.size(0):
                sea_bias = self._sea_bias
        detect_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.detect is not None:
            if need_reason and visual_tokens is not None:
                cache = not self.training
                persist = cache and bool(getattr(self.cfg.detect, "persistent_inference", False))
                mem = None if not persist else self._detect_mem
                if mem is not None and mem.size(0) != frames.size(0):
                    mem = None
                self._last_detect_mem_in = None if mem is None else mem.detach()
                do = self.detect.observe(
                    visual_tokens,
                    hw=visual_hw,
                    mem=mem,
                    gui_mask=hold if hold_any else None,
                )
                detect_bias = do["bias"]
                if cache:
                    self._detect_mem = do["mem"].detach() if persist else None
                    self._detect_bias = detect_bias.detach()
                    cells = None
                    if self._overlay_trace or self._activity_trace:
                        cells = overlay_cells(do)
                    self._overlay_detect = cells if self._overlay_trace else None
                    if self._activity_trace and cells is not None:
                        self._last_detect = {
                            "mask": do["mask"][0].detach(),
                            "k": int(cells["k"]),
                            "n": int(cells["n"]),
                            "cls": cells["cls"][:8].detach() if isinstance(cells["cls"], Tensor) else cells["cls"],
                            "obj": cells["obj"],
                        }
                else:
                    self._detect_mem = None
                    self._detect_bias = None
                    self._overlay_detect = None
            elif self._detect_bias is not None and self._detect_bias.size(0) == frames.size(0):
                detect_bias = self._detect_bias
        event_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.event_seq is not None:
            if need_reason:
                cache = not self.training
                persist = cache and bool(getattr(self.cfg.event_seq, "persistent_inference", False))
                evt = None
                if plan_out is not None and self.planner is not None and "mix" in plan_out:
                    evt = self._planner_event_tokens(plan_out["mix"])
                self._last_event_tokens = None if evt is None else evt.detach()
                slots = None if not persist else self._event_slots
                ages = None if not persist else self._event_ages
                mq = None if not persist else self._event_mem_q
                mk = None if not persist else self._event_mem_k
                if slots is not None and slots.size(0) != frames.size(0):
                    slots = ages = mq = mk = None
                self._last_event_slots = None if slots is None else slots.detach()
                self._last_event_ages = None if ages is None else ages.detach()
                self._last_event_mem_q = None if mq is None else mq.detach()
                self._last_event_mem_k = None if mk is None else mk.detach()
                eo = self.event_seq.observe(
                    pre + sea_bias + detect_bias,
                    event_tokens=evt,
                    slots=slots,
                    ages=ages,
                    mem_q=mq,
                    mem_k=mk,
                    pointer=self._pointer_feat(game_state),
                )
                event_bias = eo["bias"]
                self._event_slots = eo["slots"].detach() if persist else None
                self._event_ages = eo["ages"].detach() if persist else None
                self._event_mem_q = eo["mem_q"].detach() if persist else None
                self._event_mem_k = eo["mem_k"].detach() if persist else None
                self._event_bias = event_bias.detach() if cache else None
                if cache and self._activity_trace:
                    self._last_event = {
                        "order": eo["order"][0].detach(),
                        "n_slots": int(self.event_seq.n_slots),
                    }
            elif self._event_bias is not None and self._event_bias.size(0) == frames.size(0):
                event_bias = self._event_bias
        thought_bias = residual.new_zeros(frames.size(0), residual.size(-1))
        if self.thought is not None:
            if need_reason:
                persist = (not self.training) and bool(getattr(self.cfg.thought, "persistent_inference", True))
                mem = self._thought_mem if persist else None
                if mem is not None and mem.size(0) != frames.size(0):
                    mem = None
                thought = self.thought.observe(pre + sea_bias + detect_bias + event_bias, mem=mem)
                thought_bias = thought["bias"]
                if not self.training:
                    self._thought_mem = thought["mem"].detach() if persist else None
                    self._thought_bias = thought_bias.detach()
            elif self._thought_bias is not None and self._thought_bias.size(0) == frames.size(0):
                thought_bias = self._thought_bias
        fused = self.grow(pre + group_bias + sea_bias + detect_bias + event_bias + thought_bias)
        if self.groups is not None:
            fused = self.groups.apply_specialists(fused)
        if need_reason:
            self._last_pooled = None if pooled_out is None else pooled_out.detach()
            live = getattr(self.adapter, "last_live", None)
            if live is not None and live.shape == fused_base.shape:
                self._last_gui_live = live.detach()
                self._last_fused_base = (fused_base - self.adapter.gui_scale.detach() * live).detach()
            else:
                self._last_gui_live = None
                self._last_fused_base = fused_base.detach()
        return fused, plan_out

    def forward(
        self,
        frames: Tensor,
        game_state: Tensor,
        tokens: Tensor | None = None,
        prev_frames: Tensor | None = None,
        reason_tick: bool = True,
        plan_sample: bool = False,
        explore: float = 0.0,
        plan_temperature: float = 1.0,
        future_state: Tensor | None = None,
        clock: Tensor | None = None,
        task_id: Tensor | None = None,
        task_aux: Tensor | None = None,
        inv_ids: Tensor | None = None,
        hand_id: Tensor | None = None,
        hover_id: Tensor | None = None,
        carry_id: Tensor | None = None,
        overlay: bool | None = None,
        gui_open: bool | None = None,
        block_pulse: Tensor | None = None,
        sample_actions: bool = False,
        action_temperature: float = 1.0,
        deterministic_action: bool = False,
    ) -> dict[str, Any]:
        fused, plan_out = self.fuse(
            frames,
            game_state,
            prev_frames=prev_frames,
            reason_tick=reason_tick,
            plan_sample=plan_sample,
            explore=explore,
            plan_temperature=plan_temperature,
            future_state=future_state,
            clock=clock,
            task_id=task_id,
            task_aux=task_aux,
            inv_ids=inv_ids,
            hand_id=hand_id,
            hover_id=hover_id,
            carry_id=carry_id,
            overlay=overlay,
            gui_open=gui_open,
            block_pulse=block_pulse,
        )
        # Keep the exact policy input used for this action.  Online replay uses
        # it only for the frozen-teacher anchor; the learn graph is rebuilt from
        # the smaller module snapshots captured in ``fuse``.
        self._last_policy_input = fused.detach()
        out = self.policy(
            fused,
            tokens,
            sample_actions=sample_actions,
            action_temperature=action_temperature,
            deterministic_action=deterministic_action,
        )
        out["context"] = fused
        if self._concept_logits is not None:
            out["look_concept_logits"] = self._concept_logits
            # This is an observed-trajectory diagnostic, never a causal
            # explanation or a source of action selection.
            effect_action = tokens
            if effect_action is None:
                effect_action = out.get("sampled_tokens")
            if effect_action is None:
                effect_action = torch.stack([lg.argmax(dim=-1) for lg in out["logits"]], dim=1)
            effect_visual = self._last_pooled
            if effect_visual is not None:
                out["observed_effect_logits"] = self.effect_probe(
                    self._concept_logits, effect_visual, effect_action, self.spec
                )
                out["visual_only_effect_logits"] = self.effect_visual_baseline(
                    self._concept_logits, effect_visual, None, self.spec
                )
        if plan_out is not None:
            out["plan"] = plan_out
        if self._inv_pointer_logits is not None:
            out["pointer_logits"] = self._inv_pointer_logits
        return out

    @torch.no_grad()
    def act(
        self,
        frames: Tensor,
        game_state: Tensor,
        prev_frames: Tensor | None = None,
        temperature: float | None = None,
        deterministic: bool = False,
        reason_tick: bool = True,
        plan_sample: bool | None = None,
        explore: float | None = None,
        plan_temperature: float = 1.0,
        clock: Tensor | None = None,
    ) -> Tensor:
        """Return [B, n_slots] action tokens. Dual-rate is owned by DualRateClock."""
        sample_plan = bool(self.cfg.planner.creative) if plan_sample is None else bool(plan_sample)
        exp = 0.0 if not sample_plan else (self.cfg.planner.explore if explore is None else float(explore))
        temp = self.cfg.policy.temperature if temperature is None else temperature
        out = self.forward(
            frames,
            game_state,
            tokens=None,
            prev_frames=prev_frames,
            reason_tick=reason_tick,
            plan_sample=sample_plan,
            explore=exp,
            plan_temperature=plan_temperature,
            clock=clock,
            sample_actions=True,
            action_temperature=temp,
            deterministic_action=deterministic,
        )
        sampled = out.get("sampled_tokens")
        if not isinstance(sampled, Tensor):
            raise RuntimeError("policy did not return sampled_tokens")
        return sampled

    def materialize_live(self) -> None:
        """One GPU sync for HUD and the pulse window. The forward itself does not copy."""
        pending = getattr(self, "_pending_iter", None)
        if isinstance(pending, dict) and isinstance(pending.get("dlt"), Tensor):
            dlt = pending["dlt"]
            packed = [dlt]
            for key in ("change", "bias", "task_aux", "task_id"):
                value = pending.get(key)
                if isinstance(value, Tensor):
                    packed.append(value.reshape(-1).float())
            if dlt.is_cuda:
                torch.cuda.current_stream().synchronize()
            raw = dlt.detach().float().cpu().numpy()
            gh, gw = int(pending["gh"]), int(pending["gw"])
            need = gh * gw
            if int(raw.size) != need:
                pad = np.zeros(need, dtype=np.float32)
                ncopy = min(need, int(raw.size))
                pad[:ncopy] = raw[:ncopy]
                raw = pad
            med = float(np.median(raw))
            raw = np.clip(raw - med, 0.0, None)
            prev = getattr(self, "_chg_ema", None)
            if prev is not None and prev.shape == raw.shape:
                raw = 0.8 * prev + 0.2 * raw
            self._chg_ema = raw
            if self._overlay_trace:
                self._overlay_iter = {"chg": raw.astype(np.float16, copy=False).reshape(gh, gw), "hw": (gh, gw)}
            if self._activity_trace:
                tid = int(pending["task_id"].detach().cpu().reshape(-1)[0].item()) if isinstance(pending.get("task_id"), Tensor) else 0
                change = pending.get("change")
                bias = pending.get("bias")
                aux = pending.get("task_aux")
                h = pending.get("h")
                self._last_iter = {
                    "task_id": tid,
                    "task_name": task_label_from_id(tid, self.iterator.n_buckets) if self.iterator is not None else "",
                    "task_aux": None if not isinstance(aux, Tensor) else aux.detach().float().cpu().numpy(),
                    "change": 0.0 if not isinstance(change, Tensor) else float(change.detach().float().cpu().reshape(-1)[0]),
                    "grid_delta": raw.reshape(-1),
                    "h": None if not isinstance(h, Tensor) else h.detach().float().cpu().numpy(),
                    "bias_rms": 0.0 if not isinstance(bias, Tensor) else float(bias.detach().float().pow(2).mean().sqrt().cpu()),
                    "spatial_h": gh,
                    "spatial_w": gw,
                    "spatial_hw": gh,
                    "hold": bool(pending.get("hold")),
                }
            self._pending_iter = None
        inv = getattr(self, "_pending_inv", None)
        if isinstance(inv, dict):
            def _id(value: Tensor | None) -> int:
                if not isinstance(value, Tensor):
                    return 0
                return int(value.detach().cpu().reshape(-1)[0].item())
            n_buck = int(self.inventory.n_buckets) if self.inventory is not None else 1
            n_items = inv.get("n_items")
            spike = inv.get("spike")
            rms = inv.get("bias_rms")
            self._last_inv = {
                "hand": item_label_from_id(_id(inv.get("hand_id")), n_buck),
                "hover": item_label_from_id(_id(inv.get("hover_id")), n_buck),
                "carry": item_label_from_id(_id(inv.get("carry_id")), n_buck),
                "n_items": 0.0 if not isinstance(n_items, Tensor) else float(n_items.detach().float().cpu().reshape(-1)[0]),
                "spike": 0.0 if not isinstance(spike, Tensor) else float(spike.detach().float().cpu()),
                "bias_rms": 0.0 if not isinstance(rms, Tensor) else float(rms.detach().float().cpu()),
            }
            self._pending_inv = None

        def conv(value):
            if isinstance(value, Tensor):
                array = value.detach().float().cpu().numpy()
                return array
            if isinstance(value, dict):
                return {k: conv(v) for k, v in value.items()}
            if isinstance(value, list):
                return [conv(v) for v in value]
            return value

        for name in ("_overlay_curl", "_overlay_sea", "_overlay_detect", "_last_plan", "_last_curl", "_last_group", "_last_sea", "_last_detect", "_last_event"):
            current = getattr(self, name, None)
            if current is not None:
                setattr(self, name, conv(current))
        det = self._last_detect
        if isinstance(det, dict) and "kinds" not in det and "cls" in det:
            kinds = SCREEN_KINDS
            cls = np.asarray(det["cls"]).reshape(-1)
            det["kinds"] = [kinds[int(c) % len(kinds)] for c in cls[:8]]
            obj = np.asarray(det.get("obj")) if det.get("obj") is not None else np.zeros(0)
            det["energy"] = float(obj.mean()) if obj.size else 0.0
        trace = getattr(self.policy, "last_trace", None)
        if isinstance(trace, dict):
            self.policy.last_trace = conv(trace)
            if isinstance(self.policy.last_trace.get("sparsity"), np.ndarray):
                self.policy.last_trace["sparsity"] = float(self.policy.last_trace["sparsity"])
            if isinstance(self.policy.last_trace.get("v_th"), np.ndarray):
                self.policy.last_trace["v_th"] = float(np.mean(self.policy.last_trace["v_th"]))

    def activity_payload(self, info: dict | None = None) -> dict:
        """CPU-side snapshot for the live SNN window. Safe to call from the predict loop."""
        self.materialize_live()
        trace = dict(getattr(self.policy, "last_trace", None) or {})
        info = info or {}
        controls = info.get("controls") or {}
        keys = controls.get("keys") if isinstance(controls, dict) else {}
        mouse = controls.get("mouse") if isinstance(controls, dict) else {}
        payload = {
            "kind": trace.get("kind") or ("spike" if isinstance(self.policy, SpikePolicy) else "ann"),
            "in_spike": trace.get("in_spike"),
            "in_mem": trace.get("in_mem"),
            "layers": trace.get("layers") or [],
            "sparsity": _num(trace.get("sparsity")),
            "v_th": _num(trace.get("v_th"), 1.0),
            "plan_mix": None if self._last_plan is None else self._last_plan.get("mix"),
            "plan_z": None if self._last_plan is None else self._last_plan.get("z"),
            "iter_task": None,
            "iter_task_id": 0,
            "iter_aux": None,
            "iter_change": 0.0,
            "iter_grid": None,
            "iter_h": None,
            "iter_bias": 0.0,
            "iter_hw": 4,
            "iter_hold": False,
            "curl_energy": 0.0,
            "curl_map": None,
            "slot_names": list(self.spec.slot_names),
            "slot_kinds": list(self.spec.slot_kinds),
            "tokens": None,
            "keys": keys if isinstance(keys, dict) else {},
            "mouse": mouse if isinstance(mouse, dict) else {},
            "latency_ms": float(info.get("latency_ms") or 0.0),
            "perception": bool(info.get("run_perception")),
            "policy_hz": float(info.get("policy_hz") or 0.0),
            "perception_hz": float(info.get("perception_hz") or 0.0),
        }
        tokens = info.get("tokens")
        if tokens is not None:
            payload["tokens"] = np.asarray(tokens).reshape(-1).tolist()
        it = self._last_iter or {}
        adv = info.get("adv_next")
        payload["iter_task"] = str(adv).strip() if adv else it.get("task_name")
        payload["iter_task_id"] = int(_num(it.get("task_id")))
        payload["iter_aux"] = it.get("task_aux")
        payload["iter_change"] = _num(it.get("change"))
        payload["iter_grid"] = it.get("grid_delta")
        payload["iter_h"] = it.get("h")
        payload["iter_bias"] = _num(it.get("bias_rms"))
        payload["iter_hw"] = int(_num(it.get("spatial_h"), 4))
        payload["iter_gh"] = int(_num(it.get("spatial_h"), 4))
        payload["iter_gw"] = int(_num(it.get("spatial_w"), 6))
        payload["iter_hold"] = bool(it.get("hold") or False)
        curl = self._last_curl or {}
        payload["curl_energy"] = _num(curl.get("energy"))
        payload["curl_map"] = curl.get("map")
        if info.get("adv_next_frac") is not None:
            payload["iter_frac"] = float(info.get("adv_next_frac") or 0.0)
        inv = self._last_inv or {}
        payload["inv_hand"] = str(inv.get("hand") or "-")
        payload["inv_hover"] = str(inv.get("hover") or "-")
        payload["inv_carry"] = str(inv.get("carry") or "-")
        payload["inv_n"] = _num(inv.get("n_items"))
        payload["inv_spike"] = _num(inv.get("spike"))
        payload["inv_bias"] = _num(inv.get("bias_rms"))
        grp = self._last_group or {}
        payload["group_energy"] = grp.get("energy")
        payload["group_alive"] = int(grp.get("n_alive") or 0)
        payload["group_n"] = int(grp.get("n_groups") or 0)
        payload["group_bias"] = _num(grp.get("bias_rms"))
        sea = self._last_sea or {}
        payload["sea_mask"] = sea.get("mask")
        payload["sea_k"] = int(sea.get("k") or 0)
        payload["sea_n"] = int(sea.get("n") or 0)
        payload["sea_energy"] = _num(sea.get("energy"))
        det = self._last_detect or {}
        payload["detect_mask"] = det.get("mask")
        payload["detect_k"] = int(det.get("k") or 0)
        payload["detect_n"] = int(det.get("n") or 0)
        payload["detect_kinds"] = det.get("kinds") or []
        payload["detect_energy"] = _num(det.get("energy"))
        ev = self._last_event or {}
        payload["event_order"] = ev.get("order")
        payload["event_n"] = int(ev.get("n_slots") or 0)
        return payload

    def overlay_snapshot(self, info: dict | None = None) -> dict:
        """Cheap CPU dict for the in-game HUD. Does not enable activity_trace."""
        self.materialize_live()
        info = info or {}
        controls = info.get("controls") or {}
        keys = controls.get("keys") if isinstance(controls, dict) else {}
        mouse = controls.get("mouse") if isinstance(controls, dict) else {}
        buttons = mouse.get("buttons") if isinstance(mouse, dict) else {}
        pressed: list[str] = []
        if isinstance(keys, dict):
            pressed.extend(str(n) for n, st in keys.items() if str(st) in {"press", "hold", "down", "True"})
        if isinstance(buttons, dict):
            pressed.extend(str(n) for n, st in buttons.items() if str(st) in {"press", "hold", "down", "True"})
        det = self._overlay_detect or {}
        sea = self._overlay_sea or {}
        curl = self._overlay_curl or {}
        itov = self._overlay_iter or {}
        hw = det.get("hw") or (0, 0)
        task = str(info.get("adv_next") or "").strip()
        if "/" in task:
            task = task.split("/")[-1]
        return {
            "enabled": self.detect is not None,
            "hw": (int(hw[0] or 0), int(hw[1] or 0)),
            "idx": det.get("idx"),
            "cls": det.get("cls"),
            "obj": det.get("obj"),
            "infl": det.get("infl"),
            "conf": det.get("conf"),
            "n": int(det.get("n") or 0),
            "k": int(det.get("k") or 0),
            "sea_idx": sea.get("idx"),
            "sea_w": sea.get("w"),
            "sea_hw": sea.get("hw") or (0, 0),
            "vx": curl.get("vx"),
            "vy": curl.get("vy"),
            "vec_hw": curl.get("hw") or (0, 0),
            "chg": itov.get("chg"),
            "iter_hw": itov.get("hw") or (0, 0),
            "latency_ms": float(info.get("latency_ms") or 0.0),
            "perception": bool(info.get("run_perception")),
            "keys": pressed,
            "task": task,
        }


def sample_tokens(
    logits: list[Tensor],
    temperature: float = 1.0,
    deterministic: bool = False,
) -> Tensor:
    slots = []
    for logit in logits:
        if deterministic or temperature <= 0:
            slots.append(logit.argmax(dim=-1))
            continue
        scaled = logit / max(temperature, 1e-5)
        dist = torch.distributions.Categorical(logits=scaled)
        slots.append(dist.sample())
    return torch.stack(slots, dim=-1)


def action_ce_loss(logits: list[Tensor], tokens: Tensor, slot_weights: Tensor | None = None) -> Tensor:
    losses = []
    for i, logit in enumerate(logits):
        losses.append(nn.functional.cross_entropy(logit, tokens[:, i]))
    stacked = torch.stack(losses)
    if slot_weights is None:
        return stacked.mean()
    w = slot_weights.to(stacked.device, stacked.dtype)
    return (stacked * w).sum() / w.sum()


def default_slot_weights(spec: ActionSpec) -> Tensor:
    weights = []
    for kind in spec.slot_kinds:
        if kind == "mouse_axis":
            weights.append(3.0)
        elif kind == "mouse_btn":
            weights.append(2.5)
        elif kind == "wheel":
            weights.append(2.0)
        else:
            weights.append(1.0)
    return torch.tensor(weights)
