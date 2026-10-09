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
        self.n_core = max(2, int(cfg.n_phases))
        self.n_pool = max(0, int(getattr(cfg, "event_pool", 0) or 0))
        self.n_tasks = self.n_core + self.n_pool
        self.enc = nn.Sequential(
            nn.Linear(d_model + state_dim, hidden),
            Act("gelu"),
            nn.Linear(hidden, hidden),
            Act("gelu"),
        )
        self.mu = nn.Linear(hidden, cfg.latent_dim)
        self.logvar = nn.Linear(hidden, cfg.latent_dim)
        self.importance_head = nn.Linear(hidden, self.n_core)
        self.next_head = nn.Linear(hidden, self.n_core)
        self.extra_importance = nn.Linear(hidden, self.n_pool) if self.n_pool else None
        self.extra_next = nn.Linear(hidden, self.n_pool) if self.n_pool else None
        if self.extra_importance is not None:
            nn.init.zeros_(self.extra_importance.weight)
            nn.init.zeros_(self.extra_importance.bias)
            nn.init.zeros_(self.extra_next.weight)
            nn.init.zeros_(self.extra_next.bias)
        self.future_head = nn.Linear(cfg.latent_dim, state_dim)
        self.future_near_head = nn.Linear(cfg.latent_dim, state_dim)
        self.future_far_head = nn.Linear(cfg.latent_dim, state_dim)
        self.time_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.time_head.weight)
        nn.init.zeros_(self.time_head.bias)
        self.task_embed = nn.Embedding(self.n_core, cfg.latent_dim)
        self.extra_embed = nn.Embedding(self.n_pool, cfg.latent_dim) if self.n_pool else None
        self.register_buffer("event_alive", torch.zeros(self.n_pool, dtype=torch.float32))
        self.z_to_tokens = nn.Linear(cfg.latent_dim, d_model * cfg.n_plan_tokens)
        self.z_to_bias = nn.Linear(cfg.latent_dim, d_model)
        nn.init.zeros_(self.logvar.bias)
        nn.init.zeros_(self.logvar.weight)

    def live_mask(self) -> Tensor:
        core = self.importance_head.weight.new_ones(self.n_core)
        if self.n_pool <= 0:
            return core
        return torch.cat([core, (self.event_alive > 0.5).to(core.dtype)], dim=0)

    def n_born_events(self) -> int:
        if self.n_pool <= 0:
            return 0
        return int((self.event_alive > 0.5).sum().item())

    def birth_event(self) -> int | None:
        if self.n_pool <= 0 or self.extra_embed is None:
            return None
        dead = (self.event_alive < 0.5).nonzero(as_tuple=False).view(-1)
        if dead.numel() == 0:
            return None
        i = int(dead[0].item())
        with torch.no_grad():
            src = self.task_embed.weight.mean(dim=0)
            noise = 0.02 * torch.randn_like(src)
            self.extra_embed.weight[i].copy_(src + noise)
            if self.extra_importance is not None and self.extra_next is not None:
                self.extra_importance.weight[i].copy_(self.importance_head.weight[0])
                self.extra_importance.bias[i].copy_(self.importance_head.bias[0])
                self.extra_next.weight[i].copy_(self.next_head.weight[0])
                self.extra_next.bias[i].copy_(self.next_head.bias[0])
            self.event_alive[i] = 1.0
        return i

    def _event_logits(self, h: Tensor, core: nn.Linear, extra: nn.Linear | None) -> Tensor:
        logits = core(h)
        if extra is None or self.n_pool <= 0:
            return logits
        pool = extra(h)
        dead = self.event_alive < 0.5
        pool = pool.masked_fill(dead.unsqueeze(0), -40.0)
        return torch.cat([logits, pool], dim=-1)

    def _task_weight(self) -> Tensor:
        if self.extra_embed is None:
            return self.task_embed.weight
        return torch.cat([self.task_embed.weight, self.extra_embed.weight], dim=0)

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
        now_logits = self._event_logits(h, self.importance_head, self.extra_importance)
        next_logits = self._event_logits(h, self.next_head, self.extra_next)
        temp = max(1e-4, float(temperature))
        now_mix = (now_logits.float() / temp).softmax(-1)
        next_mix = (next_logits.float() / temp).softmax(-1)
        arrange = min(1.0, max(0.0, float(self.cfg.arrange)))
        blend = (1.0 - arrange) * now_mix + arrange * next_mix
        live = self.live_mask().unsqueeze(0).to(device=blend.device, dtype=blend.dtype)
        blend = blend * live
        blend = blend / blend.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if explore > 0.0 and (not self.training):
            uni = live / live.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            jump = (torch.rand(blend.size(0), 1, device=blend.device) < float(explore)).float()
            blend = (1.0 - jump) * blend + jump * uni
        if sample:
            mix = F.gumbel_softmax(blend.clamp_min(1e-8).log(), tau=max(0.5, temp), hard=self.training)
            mix = mix * live
            mix = mix / mix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            mix = blend
        z = z + mix @ self._task_weight()
        tokens = self.z_to_tokens(z).view(h.size(0), self.cfg.n_plan_tokens, self.d_model)
        bias = self.z_to_bias(z)
        teacher_mix = None
        if future_state is not None:
            h_fut = self._encode(pooled.detach(), future_state)
            teacher_mix = self._event_logits(h_fut, self.importance_head, self.extra_importance).float().softmax(-1).detach()
            t_live = self.live_mask().unsqueeze(0).to(device=teacher_mix.device, dtype=teacher_mix.dtype)
            teacher_mix = teacher_mix * t_live
            teacher_mix = teacher_mix / teacher_mix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
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
        log_next = torch.nan_to_num(log_next, nan=0.0, posinf=0.0, neginf=0.0)
        consist = -(plan["teacher_mix"].float() * log_next).sum(-1).mean()
    mix = plan["mix"].float()
    mean_mix = mix.mean(dim=0).clamp_min(1e-8)
    k = float(max(2, int((mean_mix > 1e-5).sum().item())))
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
