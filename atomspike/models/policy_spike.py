"""Spiking Transformer policy: LIF dynamics + spike-driven attention.

Membrane state is carried across 30Hz ticks, which is isomorphic to
press/hold/release. Training can unroll a few internal time-steps;
runtime uses one step per policy tick with persistent membrane.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from atomspike.config import PolicyConfig
from atomspike.models.actions import ActionSpec
from atomspike.models.activations import Act
from atomspike.models.decoder import ActionDecoder
from atomspike.models.lif import LIFCell


class SpikeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, tau: float, v_th: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.lif_q = LIFCell(tau=tau, v_th=v_th)
        self.lif_k = LIFCell(tau=tau, v_th=v_th)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor, mem_q: Tensor | None, mem_k: Tensor | None):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, H, N, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_s, mem_q = self.lif_q(q, mem_q)
        k_s, mem_k = self.lif_k(k, mem_k)
        attn = torch.matmul(q_s, k_s.transpose(-2, -1)) * self.scale
        # spike attention is already sparse; still normalize for stability
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.out(out), mem_q, mem_k, q_s, k_s


class SpikeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, tau: float, v_th: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SpikeSelfAttention(d_model, n_heads, tau, v_th)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            Act("gelu"),
            nn.Linear(d_model * 2, d_model),
        )
        self.lif = LIFCell(tau=tau, v_th=v_th)

    def forward(self, x: Tensor, state: dict[str, Tensor] | None):
        state = state or {}
        h, mq, mk, q_s, k_s = self.attn(self.norm1(x), state.get("mq"), state.get("mk"))
        x = x + h
        mlp_out = self.mlp(self.norm2(x))
        spike, mem = self.lif(mlp_out, state.get("mem"))
        x = x + spike
        return x, {"mq": mq, "mk": mk, "mem": mem, "q_s": q_s, "k_s": k_s, "spike": spike}


class SpikePolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, spec: ActionSpec):
        super().__init__()
        self.time_steps = cfg.spike_time_steps
        self.blocks = nn.ModuleList(
            SpikeBlock(cfg.d_model, cfg.n_heads, cfg.lif_tau, cfg.lif_v_th)
            for _ in range(cfg.n_layers)
        )
        self.in_lif = LIFCell(tau=cfg.lif_tau, v_th=cfg.lif_v_th)
        self.decoder = ActionDecoder(cfg.d_model, spec, mode=cfg.action_decode)
        self.value_head = nn.Linear(cfg.d_model, 1)
        self._state: list[dict[str, Tensor]] | None = None
        self._in_mem: Tensor | None = None
        self.trace_enabled = False
        self.last_trace: dict | None = None
        self.v_th = float(cfg.lif_v_th)

    def reset_state(self) -> None:
        self._state = None
        self._in_mem = None
        self.last_trace = None

    def forward(self, context: Tensor, tokens: Tensor | None = None) -> dict[str, object]:
        x = context.unsqueeze(1)  # [B, 1, D]
        if self.training:
            state, in_mem = None, None
            steps = max(1, self.time_steps)
        else:
            state, in_mem = self._state, self._in_mem
            if in_mem is not None and in_mem.size(0) != x.size(0):
                state, in_mem = None, None
            steps = 1
        spike_rates: list[Tensor] = []
        want_trace = bool(self.trace_enabled) and (not self.training)
        last_in_spike: Tensor | None = None
        last_in_mem: Tensor | None = None
        last_layers: list[dict] = []
        for _ in range(steps):
            spiked, in_mem = self.in_lif(x, in_mem)
            spike_rates.append(spiked.detach().float().mean())
            last_in_spike, last_in_mem = spiked, in_mem
            h = spiked
            new_state: list[dict[str, Tensor]] = []
            last_layers = []
            for i, block in enumerate(self.blocks):
                st = None if state is None else state[i]
                h, st_out = block(h, st)
                persist = {k: st_out[k] for k in ("mq", "mk", "mem") if k in st_out}
                new_state.append(persist)
                if st_out.get("q_s") is not None:
                    spike_rates.append(st_out["q_s"].detach().float().mean())
                if st_out.get("spike") is not None:
                    spike_rates.append(st_out["spike"].detach().float().mean())
                if want_trace:
                    last_layers.append(
                        {
                            "mem": _vec(st_out.get("mem")),
                            "spike": _vec(st_out.get("spike")),
                            "q_s": _vec(st_out.get("q_s")),
                            "k_s": _vec(st_out.get("k_s")),
                            "q_mem": _vec(st_out.get("mq")),
                            "k_mem": _vec(st_out.get("mk")),
                        }
                    )
            x = h
            state = new_state
        if not self.training:
            self._state = _detach_state(state)
            self._in_mem = in_mem.detach() if in_mem is not None else None
        pooled = x.mean(dim=1)
        logits = self.decoder(pooled, tokens)
        value = self.value_head(pooled).squeeze(-1)
        sparsity = torch.stack(spike_rates).mean() if spike_rates else pooled.new_zeros(())
        if want_trace:
            self.last_trace = {
                "kind": "spike",
                "in_spike": _vec(last_in_spike),
                "in_mem": _vec(last_in_mem),
                "layers": last_layers,
                "sparsity": float(sparsity.detach().float().item()),
                "v_th": float(self.v_th),
            }
        return {"logits": logits, "value": value, "spikes": sparsity}


def _vec(t: Tensor | None) -> np.ndarray:
    if t is None:
        return np.zeros(1, dtype=np.float32)
    x = t.detach().float()
    if x.ndim >= 1:
        x = x[0]
    return np.ascontiguousarray(x.reshape(-1).cpu().numpy(), dtype=np.float32)


def _detach_state(state: list[dict[str, Tensor]] | None) -> list[dict[str, Tensor]] | None:
    if state is None:
        return None
    keep = ("mq", "mk", "mem")
    return [{k: v.detach() for k, v in st.items() if k in keep} for st in state]
