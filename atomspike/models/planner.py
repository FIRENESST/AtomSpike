"""Self-semantic planner: invented tasks, inferred importance, self-arranged order.

Recording sessions are split before this runs. Within a session the planner
predicts near/mid/far game_state and a session-local progress score. It does
not use a hard time-bin action curriculum.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import PlannerConfig
from atomspike.models.activations import Act


class SemanticPlanner(nn.Module):
    def __init__(self, cfg: PlannerConfig, d_model: int, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.state_dim = state_dim
        hidden = max(32, int(cfg.hidden))
        self.n_tasks = max(2, int(cfg.n_phases))
        self.enc = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden),
            Act("gelu"),
            nn.Linear(hidden, hidden),
            Act("gelu"),
        )
        self.mu = nn.Linear(hidden, cfg.latent_dim)
        self.logvar = nn.Linear(hidden, cfg.latent_dim)
        self.importance_head = nn.Linear(hidden, self.n_tasks)
        self.next_head = nn.Linear(hidden, self.n_tasks)
        self.future_head = nn.Linear(cfg.latent_dim, state_dim)
        self.future_near_head = nn.Linear(cfg.latent_dim, state_dim)
        self.future_far_head = nn.Linear(cfg.latent_dim, state_dim)
        self.time_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.time_head.weight)
        nn.init.zeros_(self.time_head.bias)
        self.task_embed = nn.Embedding(self.n_tasks, cfg.latent_dim)
        self.z_to_tokens = nn.Linear(cfg.latent_dim, d_model * cfg.n_plan_tokens)
        self.z_to_bias = nn.Linear(cfg.latent_dim, d_model)
        nn.init.zeros_(self.logvar.bias)
        nn.init.zeros_(self.logvar.weight)

    def _encode(self, pooled: Tensor, game_state: Tensor) -> Tensor:
        return self.enc(torch.cat([pooled, game_state], dim=-1))

    def forward(
        self,
        pooled: Tensor,
        game_state: Tensor,
        *,
        sample: bool = False,
        explore: float = 0.0,
        temperature: float = 1.0,
        future_state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        h = self._encode(pooled, game_state)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 4.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        now_logits = self.importance_head(h)
        next_logits = self.next_head(h)
        temp = max(1e-4, float(temperature))
        now_mix = (now_logits.float() / temp).softmax(-1)
        next_mix = (next_logits.float() / temp).softmax(-1)
        arrange = min(1.0, max(0.0, float(self.cfg.arrange)))
        blend = (1.0 - arrange) * now_mix + arrange * next_mix
        if explore > 0.0 and (not self.training):
            uni = torch.full_like(blend, 1.0 / self.n_tasks)
            jump = (torch.rand(blend.size(0), 1, device=blend.device) < float(explore)).float()
            blend = (1.0 - jump) * blend + jump * uni
        if sample:
            mix = F.gumbel_softmax(blend.clamp_min(1e-8).log(), tau=max(0.5, temp), hard=self.training)
        else:
            mix = blend
        z = z + mix @ self.task_embed.weight
        tokens = self.z_to_tokens(z).view(h.size(0), self.cfg.n_plan_tokens, self.d_model)
        bias = self.z_to_bias(z)
        teacher_mix = None
        if future_state is not None:
            h_fut = self._encode(pooled.detach(), future_state)
            teacher_mix = self.importance_head(h_fut).float().softmax(-1).detach()
        out = {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "phase_logits": now_logits,
            "next_logits": next_logits,
            "mix": mix,
            "now_mix": now_mix,
            "next_mix": next_mix,
            "phase": mix.argmax(dim=-1),
            "tokens": tokens,
            "bias": bias,
            "future": self.future_head(z),
            "future_near": self.future_near_head(z),
            "future_far": self.future_far_head(z),
            "time_pred": torch.sigmoid(self.time_head(h).squeeze(-1)),
        }
        if teacher_mix is not None:
            out["teacher_mix"] = teacher_mix
        return out


def plan_aux_loss(plan: dict[str, Tensor], batch: dict[str, Tensor], cfg: PlannerConfig) -> Tensor:
    """Unsupervised: future state (multi-horizon), session progress, VAE KL, next-task, diversity."""
    future = batch["future_state"].float()
    fut = F.mse_loss(plan["future"].float(), future)
    if "future_state_near" in batch and "future_near" in plan:
        fut = fut + 0.5 * F.mse_loss(plan["future_near"].float(), batch["future_state_near"].float())
    if "future_state_far" in batch and "future_far" in plan:
        fut = fut + 0.5 * F.mse_loss(plan["future_far"].float(), batch["future_state_far"].float())
    mu = plan["mu"].float()
    logvar = plan["logvar"].float()
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
    consist = plan["future"].new_zeros(())
    if "teacher_mix" in plan:
        log_next = F.log_softmax(plan["next_logits"].float(), dim=-1)
        consist = -(plan["teacher_mix"].float() * log_next).sum(-1).mean()
    mix = plan["mix"].float()
    mean_mix = mix.mean(dim=0).clamp_min(1e-8)
    k = float(mix.size(-1))
    div = (mean_mix * (mean_mix.log() + math.log(k))).sum()
    time_aux = plan["future"].new_zeros(())
    if float(cfg.time_coef) > 0.0 and "time_pred" in plan and "time_frac" in batch:
        tgt = batch["time_frac"].float().reshape(-1)
        pred = plan["time_pred"].float().reshape(-1)
        time_aux = F.mse_loss(pred, tgt)
    return (
        float(cfg.future_coef) * fut
        + float(cfg.kl_coef) * kl
        + float(cfg.next_coef) * consist
        + float(cfg.div_coef) * div
        + float(cfg.time_coef) * time_aux
    )


def build_episode_timeline(
    states: "object",
    dones: "object",
    *,
    n_phases: int,
    future_steps: int,
    timestamps_ns: "object | None" = None,
    gap_s: float = 2.0,
    future_near: int = 5,
    future_far: int = 45,
    fps: float = 30.0,
) -> dict[str, object]:
    """Per-recording clocks and clipped future state. See data.sessions."""
    from atomspike.data.sessions import build_episode_timeline as _build

    return _build(
        states,
        dones,
        n_phases=n_phases,
        future_steps=future_steps,
        timestamps_ns=timestamps_ns,
        gap_s=gap_s,
        future_near=future_near,
        future_far=future_far,
        fps=fps,
    )
