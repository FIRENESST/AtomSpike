"""Global event sequence: planner event slots + a short tape, learned order.

Time is not a clock scalar. Event slots sit on a global sequence; an order
head scores each slot and coincidence attention mixes them. The model
learns which events are earlier / later from action prediction.

Write scale and residual start at zero so old checkpoints stay put.
10Hz only; 30Hz adds the cached bias.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import EventSeqConfig
from atomspike.models.lif import LIFCell
from atomspike.models.sea import coincidence_weights


class GlobalEventSequence(nn.Module):
    def __init__(self, cfg: EventSeqConfig, d_model: int, event_dim: int | None = None):
        super().__init__()
        d = int(d_model)
        self.n_slots = max(2, int(getattr(cfg, "n_slots", 8) or 8))
        self.n_events = max(1, int(getattr(cfg, "n_events", 8) or 8))
        heads = max(1, int(getattr(cfg, "n_heads", 2) or 2))
        while heads > 1 and d % heads != 0:
            heads -= 1
        self.n_heads = heads
        self.head_dim = d // heads
        src = int(event_dim or d)
        self.event_proj = nn.Linear(src, d, bias=False) if src != d else nn.Identity()
        self.pointer_proj = nn.Linear(4, d, bias=False)
        self.cand = nn.Linear(d, d, bias=False)
        self.write_head = nn.Linear(d, self.n_slots)
        self.order_head = nn.Linear(d + 1, 1)
        self.qkv = nn.Linear(d, d * 3, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        tau = float(getattr(cfg, "lif_tau", 0.5) or 0.5)
        v_th = float(getattr(cfg, "lif_v_th", 1.0) or 1.0)
        self.lif_q = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.lif_k = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.write_scale = nn.Parameter(torch.zeros(()))
        self.resid_scale = nn.Parameter(torch.zeros(d))

    def _pad_events(self, evt: Tensor | None, decision: Tensor) -> Tensor:
        b, d = decision.shape
        if evt is None or evt.numel() == 0:
            return decision.new_zeros(b, self.n_events, d)
        if evt.size(-1) != d:
            evt = evt[..., :d]
        n = int(evt.size(1))
        if n == self.n_events:
            return evt
        if n > self.n_events:
            return evt[:, : self.n_events]
        return F.pad(evt, (0, 0, 0, self.n_events - n))

    def observe(
        self,
        decision: Tensor,
        event_tokens: Tensor | None = None,
        slots: Tensor | None = None,
        ages: Tensor | None = None,
        mem_q: Tensor | None = None,
        mem_k: Tensor | None = None,
        pointer: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b, d = decision.shape
        if slots is None or slots.size(0) != b or slots.size(1) != self.n_slots:
            slots = decision.new_zeros(b, self.n_slots, d)
        if ages is None or ages.shape != slots.shape[:2]:
            ages = decision.new_zeros(b, self.n_slots)
        cand = self.cand(decision)
        write = self.write_head(cand).softmax(dim=-1)
        slots = slots + self.write_scale * write.unsqueeze(-1) * cand.unsqueeze(1)
        ages = ages + 1.0
        ages = ages * (1.0 - write)
        evt = self.event_proj(event_tokens) if event_tokens is not None and event_tokens.numel() > 0 else None
        evt = self._pad_events(evt, decision)
        if pointer is None:
            ptr = decision.new_zeros(b, 1, d)
        else:
            feat = pointer.to(device=decision.device, dtype=decision.dtype).reshape(b, -1)
            if feat.size(-1) < 4:
                feat = F.pad(feat, (0, 4 - feat.size(-1)))
            ptr = self.pointer_proj(feat[:, :4]).unsqueeze(1)
        seq = torch.cat([decision.unsqueeze(1), slots, evt, ptr], dim=1)
        t = seq.size(1)
        age_pad = seq.new_zeros(b, t, 1)
        age_pad[:, 1 : 1 + self.n_slots, 0] = ages / 8.0
        order = self.order_head(torch.cat([seq, age_pad], dim=-1)).squeeze(-1)
        rank = order.softmax(dim=-1)
        qkv = self.qkv(seq).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_s, mem_q = self.lif_q(q, mem_q if mem_q is not None and mem_q.shape == q.shape else None)
        k_s, mem_k = self.lif_k(k, mem_k if mem_k is not None and mem_k.shape == k.shape else None)
        attn = coincidence_weights(q_s, k_s)
        attn = attn * rank.view(b, 1, 1, t)
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        mixed = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, d)
        delta = self.out(mixed)[:, 0]
        return {
            "bias": self.resid_scale * delta,
            "slots": slots,
            "ages": ages,
            "order": rank,
            "mem_q": mem_q,
            "mem_k": mem_k,
        }
