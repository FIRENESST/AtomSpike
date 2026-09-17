"""Full AtomSpike agent: encoder + reasoner (5Hz) + residual + policy (30Hz)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.models.actions import ActionSpec
from atomspike.models.encoder import VisualEncoder
from atomspike.models.planner import SemanticPlanner
from atomspike.models.policy_ann import ANNPolicy
from atomspike.models.policy_spike import SpikePolicy
from atomspike.models.reasoner import SemanticReasoner
from atomspike.models.temporal import TemporalResidualAdapter
from atomspike.data.sessions import CLOCK_DIM
from atomspike.models.activations import Act


class AtomSpikeAgent(nn.Module):
    def __init__(self, cfg: AtomSpikeConfig):
        super().__init__()
        self.cfg = cfg
        self.spec = ActionSpec.from_config(cfg.action)
        self.encoder = VisualEncoder(cfg.encoder)
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
        d = cfg.encoder.d_model
        self.clock_in = nn.Sequential(
            nn.Linear(CLOCK_DIM, d),
            Act("gelu"),
            nn.Linear(d, d),
        )
        self._context: Tensor | None = None
        self._plan_bias: Tensor | None = None
        self._activity_trace = False
        self._last_plan: dict | None = None

    def set_activity_trace(self, enabled: bool) -> None:
        self._activity_trace = bool(enabled)
        if hasattr(self.policy, "trace_enabled"):
            self.policy.trace_enabled = bool(enabled)
        if not enabled:
            self._last_plan = None
            if hasattr(self.policy, "last_trace"):
                self.policy.last_trace = None

    def reset_runtime(self) -> None:
        self._context = None
        self._plan_bias = None
        reset = getattr(self.policy, "reset_state", None)
        if callable(reset):
            reset()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        residual = self.adapter(frames, prev_frames)
        if clock is None:
            clock_feat = frames.new_zeros(frames.size(0), CLOCK_DIM)
        else:
            clock_feat = clock.float().reshape(frames.size(0), CLOCK_DIM)
        clock_vec = self.clock_in(clock_feat)
        need_reason = reason_tick or self._context is None or self._context.size(0) != frames.size(0)
        plan_out = None
        if need_reason:
            tokens, pooled = self.encoder(frames)
            plan_tokens = None
            plan_bias = frames.new_zeros(frames.size(0), residual.size(-1))
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
                        "mix": None if mix is None else mix[0].detach().float().cpu().numpy(),
                        "z": None if z is None else z[0].detach().float().cpu().numpy(),
                    }
                if (not self.training) and self._plan_bias is not None and self._plan_bias.size(0) == plan_bias.size(0):
                    plan_bias = 0.65 * self._plan_bias + 0.35 * plan_bias
            clock_tok = clock_vec.unsqueeze(1)
            extra = clock_tok if plan_tokens is None else torch.cat([plan_tokens, clock_tok], dim=1)
            ctx = self.reasoner(tokens, game_state, plan_tokens=extra)
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
        return ctx + residual + plan_bias + clock_vec, plan_out

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
        )
        out = self.policy(fused, tokens)
        out["context"] = fused
        if plan_out is not None:
            out["plan"] = plan_out
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
        )
        temp = self.cfg.policy.temperature if temperature is None else temperature
        return sample_tokens(out["logits"], temperature=temp, deterministic=deterministic)

    def activity_payload(self, info: dict | None = None) -> dict:
        """CPU-side snapshot for the live SNN window. Safe to call from the predict loop."""
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
            "sparsity": float(trace.get("sparsity") or 0.0),
            "v_th": float(trace.get("v_th") or 1.0),
            "plan_mix": None if self._last_plan is None else self._last_plan.get("mix"),
            "plan_z": None if self._last_plan is None else self._last_plan.get("z"),
            "slot_names": list(self.spec.key_names)
            + ["MX", "MY"]
            + ["LMB", "RMB"][: self.spec.n_mouse_buttons],
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
        return payload


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
        else:
            weights.append(1.0)
    return torch.tensor(weights)
