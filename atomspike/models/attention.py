"""Explicit Q/K/V attention so LoRA and SpikedAttention attach to real Linear modules.

nn.MultiheadAttention keeps in_proj as a fused Parameter, which made T4 wrap
count hit 0 and T6 swap impossible to reload.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.register_buffer("spike_attn", torch.zeros((), dtype=torch.uint8))
        self.register_buffer("spike_th", torch.zeros(()))
        self._spike = False
        self.register_load_state_dict_post_hook(MultiHeadSelfAttention._sync_spike)

    def _sync_spike(self, *_args) -> None:
        self._spike = bool(int(self.spike_attn.detach().cpu().item()))

    def set_spike(self, enabled: bool, threshold: float = 0.0) -> None:
        self._spike = bool(enabled)
        self.spike_attn.fill_(1 if enabled else 0)
        self.spike_th.fill_(float(threshold))

    @property
    def is_spike(self) -> bool:
        return self._spike

    def _heads(self, t: Tensor, b: int) -> Tensor:
        return t.view(b, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        b = x.size(0)
        q = self._heads(self.q(x), b)
        k = self._heads(self.k(x), b)
        v = self._heads(self.v(x), b)
        if self._spike:
            q = (q > self.spike_th).to(dtype=v.dtype)
            k = (k > self.spike_th).to(dtype=v.dtype)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            # Binary Q/K still need a normalized value mixture.  Raw overlap
            # grows with the number of co-firing channels and used to explode
            # as sequence length increased.
            attn = attn.softmax(dim=-1)
            attn = self.drop(attn)
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.drop(attn)
        h = torch.matmul(attn, v).transpose(1, 2).contiguous()
        h = h.view(b, -1, self.n_heads * self.head_dim)
        return self.out(h)
