"""LoRA on real Linear layers. Reconstruct wrappers before load_state_dict."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scaling


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def wrap_linears(module: nn.Module, r: int, alpha: int) -> int:
    """Replace every nn.Linear descendant with LoRALinear. Returns wrap count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += wrap_linears(child, r, alpha)
    return count


def lora_param_count(module: nn.Module) -> int:
    n = 0
    for m in module.modules():
        if isinstance(m, LoRALinear):
            n += m.lora_a.numel() + m.lora_b.numel()
    return n


def apply_lora(agent: nn.Module, r: int, alpha: int) -> dict[str, Any]:
    """Freeze perception backbone, inject LoRA into the reasoner.

    Policy head stays fully trainable (small). Raises if nothing was wrapped.
    """
    encoder = getattr(agent, "encoder", None)
    reasoner = getattr(agent, "reasoner", None)
    if encoder is None or reasoner is None:
        raise RuntimeError("apply_lora expects an agent with encoder and reasoner")
    freeze_module(encoder)
    freeze_module(reasoner)
    n = wrap_linears(reasoner, r, alpha)
    if n <= 0:
        raise RuntimeError(
            "T4 LoRA wrapped 0 Linear layers. Reasoner must expose nn.Linear "
            "(q/k/v/out/fc), not fused MultiheadAttention.in_proj_weight."
        )
    meta = {
        "r": int(r),
        "alpha": int(alpha),
        "wrapped": int(n),
        "targets": ["reasoner"],
        "lora_params": lora_param_count(agent),
    }
    agent._lora_meta = meta  # type: ignore[attr-defined]
    return meta


def trainable_parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return train, total
