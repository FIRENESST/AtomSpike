"""Full AtomSpike agent: encoder + reasoner (5Hz) + residual + policy (30Hz)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.models.actions import ActionSpec
from atomspike.models.encoder import VisualEncoder
from atomspike.models.policy_ann import ANNPolicy
from atomspike.models.policy_spike import SpikePolicy
from atomspike.models.reasoner import SemanticReasoner
from atomspike.models.temporal import TemporalResidualAdapter


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
        self._context: Tensor | None = None

    def reset_runtime(self) -> None:
        self._context = None
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
    ) -> Tensor:
        residual = self.adapter(frames, prev_frames)
        need_reason = reason_tick or self._context is None or self._context.size(0) != frames.size(0)
        if need_reason:
            tokens, _ = self.encoder(frames)
            ctx = self.reasoner(tokens, game_state)
            if self.training:
                self._context = ctx
            else:
                self._context = ctx.detach()
        else:
            ctx = self._context
            assert ctx is not None
        return ctx + residual

    def forward(
        self,
        frames: Tensor,
        game_state: Tensor,
        tokens: Tensor | None = None,
        prev_frames: Tensor | None = None,
        reason_tick: bool = True,
    ) -> dict[str, Any]:
        fused = self.fuse(frames, game_state, prev_frames=prev_frames, reason_tick=reason_tick)
        out = self.policy(fused, tokens)
        out["context"] = fused
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
    ) -> Tensor:
        """Return [B, 8] action tokens. Dual-rate is owned by DualRateClock."""
        out = self.forward(
            frames,
            game_state,
            tokens=None,
            prev_frames=prev_frames,
            reason_tick=reason_tick,
        )
        temp = self.cfg.policy.temperature if temperature is None else temperature
        return sample_tokens(out["logits"], temperature=temp, deterministic=deterministic)


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
            weights.append(2.0)
        elif kind == "mouse_btn":
            weights.append(2.5)
        else:
            weights.append(1.0)
    return torch.tensor(weights)
