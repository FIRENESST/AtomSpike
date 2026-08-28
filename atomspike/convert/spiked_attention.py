"""SpikedAttention-style conversion: spike Q/K, drop softmax.

Applied to MultiheadAttention in the reasoner when convert.method is
spiked_attention. Combined with PMSM activation swap.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from atomspike.config import AtomSpikeConfig
from atomspike.convert.pmsm import SpikeReLU, _replace_activations
from atomspike.models.agent import AtomSpikeAgent
from atomspike.train.common import load_agent, resolve_device, save_checkpoint


class SpikedMHA(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention, v_th: float = 0.0):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.out_proj = mha.out_proj
        if mha.in_proj_weight is not None:
            with torch.no_grad():
                self.in_proj.weight.copy_(mha.in_proj_weight)
                if mha.in_proj_bias is not None and self.in_proj.bias is not None:
                    self.in_proj.bias.copy_(mha.in_proj_bias)
        self.v_th = v_th
        self.batch_first = mha.batch_first

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor | tuple[Tensor, None]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, n, _ = x.shape
        qkv = self.in_proj(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = (q > self.v_th).to(v.dtype)
        k = (k > self.v_th).to(v.dtype)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / max(1, n)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n, self.embed_dim)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        if kwargs.get("need_weights", False):
            return out, None
        return out


def _swap_mha(module: nn.Module, v_th: float) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, SpikedMHA(child, v_th=v_th))
            n += 1
        else:
            n += _swap_mha(child, v_th)
    return n


def convert_spiked_attention(
    cfg: AtomSpikeConfig,
    ckpt_path: str | Path,
    out_path: str | Path,
) -> dict:
    device = resolve_device(cfg.train.device)
    agent = load_agent(ckpt_path, cfg, device)
    n_act = _replace_activations(agent, threshold=0.0)
    n_mha = _swap_mha(agent, v_th=0.0)
    path = save_checkpoint(agent, out_path, extra={"spiked_mha": n_mha, "replaced_act": n_act})
    return {"checkpoint": str(path), "spiked_mha": n_mha, "replaced": n_act, "method": "spiked_attention"}
