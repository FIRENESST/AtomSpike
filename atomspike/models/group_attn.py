"""Spike coalitions: activity groups + 10Hz group attention + specialist neurons.

Decision-layer channels are split into a few groups. A cheap spike
self-attention mixes those group tokens at perception rate. Highly active
groups can later birth a tiny specialist residual under the decision layer.

30Hz ticks only add the cached group residual and, if any specialist is
alive, a Dg-wide MLP on that slice. Until birth, the extra 30Hz cost is one
add of a cached zero vector.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import GroupConfig
from atomspike.models.lif import LIFCell


def choose_n_groups(d_model: int, want: int = 8) -> int:
    d = max(1, int(d_model))
    for g in (int(want), 8, 4, 2, 1):
        if g >= 1 and d % g == 0:
            return g
    return 1


class _GroupSpikeAttn(nn.Module):
    """G×G spike attention. Output projection is a normal Linear; the caller gates it."""

    def __init__(self, d_group: int, n_heads: int, tau: float, v_th: float):
        super().__init__()
        heads = max(1, int(n_heads))
        while heads > 1 and d_group % heads != 0:
            heads -= 1
        self.n_heads = heads
        self.head_dim = d_group // heads
        self.qkv = nn.Linear(d_group, d_group * 3, bias=False)
        self.out = nn.Linear(d_group, d_group, bias=False)
        self.lif_q = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.lif_k = LIFCell(tau=tau, v_th=v_th, thresh_shape=(heads, 1, self.head_dim))
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, mem_q: Tensor | None, mem_k: Tensor | None):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_s, mem_q = self.lif_q(q, mem_q)
        k_s, mem_k = self.lif_k(k, mem_k)
        attn = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.out(h), mem_q, mem_k


class _SliceSpecialist(nn.Module):
    def __init__(self, d_group: int, hidden: int):
        super().__init__()
        h = max(4, int(hidden))
        self.fc1 = nn.Linear(d_group, h)
        self.fc2 = nn.Linear(h, d_group)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SpikeGroupCoalition(nn.Module):
    """Activity groups + spike attention residual + on-demand specialists."""

    def __init__(self, cfg: GroupConfig, d_model: int):
        super().__init__()
        d = int(d_model)
        self.n_groups = choose_n_groups(d, int(getattr(cfg, "n_groups", 8) or 8))
        self.d_group = d // self.n_groups
        self.ema_decay = min(0.99, max(0.0, float(getattr(cfg, "ema", 0.9) or 0.9)))
        tau = float(getattr(cfg, "lif_tau", 0.5) or 0.5)
        v_th = float(getattr(cfg, "lif_v_th", 1.0) or 1.0)
        hidden = max(4, int(getattr(cfg, "specialist_hidden", 8) or 8))
        n_heads = max(1, int(getattr(cfg, "n_heads", 2) or 2))
        self.lif = LIFCell(tau=tau, v_th=v_th, thresh_shape=(d,))
        self.attn = _GroupSpikeAttn(self.d_group, n_heads, tau, v_th)
        self.resid_scale = nn.Parameter(torch.zeros(d))
        self.specialists = nn.ModuleList(
            _SliceSpecialist(self.d_group, hidden) for _ in range(self.n_groups)
        )
        self.register_buffer("alive", torch.zeros(self.n_groups, dtype=torch.float32))
        self.register_buffer("ema", torch.zeros(d))
        self._alive_i: list[int] = []
        self.register_load_state_dict_post_hook(SpikeGroupCoalition._sync_alive)

    def _sync_alive(self, *_args) -> None:
        a = self.alive.detach().cpu()
        self._alive_i = [i for i in range(int(a.numel())) if float(a[i]) > 0.5]

    def n_alive(self) -> int:
        return len(self._alive_i)

    def birth_specialist(self, energy: Tensor | None = None) -> int | None:
        """Turn on the most active group that does not yet have a specialist."""
        scores = self.ema.detach().view(self.n_groups, self.d_group).mean(dim=-1)
        if energy is not None:
            e = energy.detach().reshape(-1)
            if e.numel() == self.n_groups:
                scores = scores + e.to(device=scores.device, dtype=scores.dtype)
        order = torch.argsort(scores, descending=True)
        for idx in order.tolist():
            i = int(idx)
            if i in self._alive_i:
                continue
            self.alive[i] = 1.0
            self._alive_i.append(i)
            return i
        return None

    def observe(
        self,
        x: Tensor,
        mem: Tensor | None = None,
        mem_q: Tensor | None = None,
        mem_k: Tensor | None = None,
        *,
        track: bool = True,
    ) -> dict[str, Tensor]:
        """10Hz: LIF + group spike attention. Returns a residual that starts at 0."""
        spike, mem = self.lif(x, mem)
        tokens = spike.view(x.size(0), self.n_groups, self.d_group)
        h, mem_q, mem_k = self.attn(tokens, mem_q, mem_k)
        y = h.reshape(x.size(0), -1)
        energy = tokens.mean(dim=-1)
        if track:
            with torch.no_grad():
                rate = spike.detach().float().mean(dim=0)
                self.ema.mul_(self.ema_decay).add_(rate, alpha=1.0 - self.ema_decay)
        return {
            "bias": self.resid_scale * y,
            "mem": mem,
            "mem_q": mem_q,
            "mem_k": mem_k,
            "energy": energy,
            "spike": spike,
        }

    def apply_specialists(self, x: Tensor) -> Tensor:
        """30Hz: skip entirely until at least one group specialist is born."""
        if not self._alive_i:
            return x
        b = x.size(0)
        chunks = x.view(b, self.n_groups, self.d_group)
        out = list(chunks.unbind(dim=1))
        for i in self._alive_i:
            out[i] = out[i] + self.specialists[i](out[i])
        return torch.stack(out, dim=1).reshape(b, -1)
