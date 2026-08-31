"""Muon + AdamW hybrid optimizer.

Muon (Keller Jordan's Momentum + Orthogonalization) replaces SGD-momentum's
update with the Newton-Schulz-orthogonalized momentum — on matrix-shaped
params it converges faster and is robust to larger LR. Per README-5070Ti
§4.2:

  Tier A-C default: Muon (matrix params) + AdamW (embeddings / heads /
  scalars). Muon needs weight decay and per-parameter update scaling —
  don't use it raw.

This module exposes `build_optimizer` which picks the right mix based on
each parameter's tensor shape:
  - 2D weight matrices (Linear.weight, conv kernel flattened) -> Muon
  - everything else (bias, LayerNorm, embeddings, scalars, conv weights
    kept 4D) -> AdamW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to orthogonalize G.

    Returns a matrix X with the same shape as G whose singular values are
    pushed towards 1. Used as the update direction for Muon.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) if G.dtype == torch.float32 else G
    # normalize spectral norm <= 1
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Minimal Muon optimizer for 2D weight matrices.

    Per-parameter scaling: update is multiplied by sqrt(max(rows, cols)) so
    a single LR works across widths (Kimi Moonlight recipe).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                # Only 2D weight matrices get the orthogonalized update;
                # 1D / 4D tensors fall back to SGD-momentum in the same step.
                if p.ndim == 2 and min(p.shape) >= 4:
                    u = _newton_schulz(g, steps=ns_steps)
                    # Per-parameter update scaling: match RMS of Adam update.
                    scale = max(1.0, (p.size(0) / max(1, p.size(1))) ** 0.5)
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-lr * scale)
                else:
                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(g, alpha=-lr)
        return loss


def _is_matrix_for_muon(p: torch.Tensor) -> bool:
    """Muon is best on 2D weight matrices. Skip 1D (bias/LN) and conv 4D."""
    if p.ndim != 2:
        return False
    # Avoid tiny 1xN or Nx1 — orthogonalization degenerates.
    return min(p.shape) >= 4


@dataclass
class HybridConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    adamw_lr: float = 3e-4
    adamw_weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)


def build_hybrid(params: Iterable[torch.Tensor], cfg: HybridConfig) -> torch.optim.Optimizer:
    """Route 2D matrices -> Muon, everything else -> AdamW (single optimizer)."""
    params = list(params)
    muon_params = [p for p in params if p.requires_grad and _is_matrix_for_muon(p)]
    adamw_params = [p for p in params if p.requires_grad and not _is_matrix_for_muon(p)]

    # Wrap as a single optimizer with two param groups; use Muon class as the
    # driver since it has the custom step, but fall back to AdamW for the
    # non-matrix group by monkey-dispatching inside step. Simpler: return a
    # CombinedOptimizer that owns both.
    opts: list[torch.optim.Optimizer] = []
    if muon_params:
        opts.append(
            Muon(
                muon_params,
                lr=cfg.muon_lr,
                momentum=cfg.muon_momentum,
                weight_decay=cfg.muon_weight_decay,
            )
        )
    if adamw_params:
        opts.append(
            torch.optim.AdamW(
                adamw_params,
                lr=cfg.adamw_lr,
                weight_decay=cfg.adamw_weight_decay,
                betas=cfg.betas,
            )
        )
    if len(opts) == 1:
        return opts[0]
    return CombinedOptimizer(opts)


class CombinedOptimizer:
    """Duck-typed optimizer wrapping several sub-optimizers.

    Behaves like a single torch.optim.Optimizer from the training loop's
    perspective: zero_grad / step / state_dict / load_state_dict /
    param_groups all delegate.
    """

    def __init__(self, opts: list[torch.optim.Optimizer]):
        if not opts:
            raise ValueError("CombinedOptimizer needs at least one sub-optimizer")
        self.opts = opts

    # -- torch.optim.Optimizer duck type ------------------------------
    @property
    def param_groups(self):
        groups = []
        for o in self.opts:
            groups.extend(o.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for o in self.opts:
            l = o.step(closure)
            if l is not None:
                loss = l
        return loss

    def state_dict(self) -> dict:
        return {
            "kind": "combined",
            "n": len(self.opts),
            "states": [o.state_dict() for o in self.opts],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "combined" or state.get("n") != len(self.opts):
            raise ValueError("CombinedOptimizer state mismatch")
        for o, s in zip(self.opts, state["states"]):
            o.load_state_dict(s)

    def add_param_group(self, group: dict) -> None:
        # route by shape: if any 2D matrix, give to Muon (first opt), else AdamW.
        params = list(group.get("params", []))
        target = self.opts[0] if any(_is_matrix_for_muon(p) for p in params) else self.opts[-1]
        target.add_param_group(group)


def build_optimizer(cfg, params: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    """Factory driven by TrainConfig."""
    params = [p for p in params if p.requires_grad]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "muon":
        return Muon(
            params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "muon_adamw":
        return build_hybrid(
            params,
            HybridConfig(
                muon_lr=cfg.muon_lr,
                muon_momentum=cfg.muon_momentum,
                muon_weight_decay=cfg.weight_decay,
                adamw_lr=cfg.lr,
                adamw_weight_decay=cfg.weight_decay,
            ),
        )
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def is_trainable_module(m: nn.Module) -> bool:
    return any(p.requires_grad for p in m.parameters(recurse=False))
