"""Minimal LoRA wrappers so per-game PEFT does not need HuggingFace PEFT."""

from __future__ import annotations

import math

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


def inject_lora(module: nn.Module, r: int, alpha: int, name_substr: tuple[str, ...] = ("qkv", "out", "pre")) -> int:
    """Replace matching Linear children with LoRALinear. Returns wrap count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(s in name.lower() for s in name_substr):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += inject_lora(child, r, alpha, name_substr)
    return count


def trainable_parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return train, total
