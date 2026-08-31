"""WSD learning-rate schedule: warmup → stable → decay.

Stable phase covers 85% of steps so any in-flight checkpoint can be
"harvested" by appending a short decay tail — the property that makes
training interruptible without losing the run.

Implementation notes:
- Pure-python step counter + LambdaLR-style multiplier; state_dict()
  round-trips the step counter so resume reproduces the exact LR curve.
- Decay supports "linear" or "cosine"; both end at ~0 by step == total.
- Compatible with Muon / AdamW hybrid optimizers: multiplies every
  param-group's base_lr, so per-group scaling set by the optimizer is
  preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import torch

DecayKind = Literal["linear", "cosine"]


@dataclass
class WSDConfig:
    total_steps: int
    warmup_frac: float = 0.05
    stable_frac: float = 0.85
    decay_frac: float = 0.10
    decay: DecayKind = "linear"
    min_lr_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        s = self.warmup_frac + self.stable_frac + self.decay_frac
        if abs(s - 1.0) > 1e-3:
            raise ValueError(f"warmup+stable+decay must equal 1.0, got {s}")


class WSDScheduler:
    """Per-step LR multiplier driven by a WSDConfig.

    Calls `step()` after each optimizer.step(). Persists as a plain dict
    so it survives torch.save without pickling the optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: WSDConfig):
        self.optimizer = optimizer
        self.cfg = cfg
        self.step_count = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_lrs = list(self.base_lrs)
        self._apply(0.0)

    # -- core curve -----------------------------------------------------
    def _multiplier(self, step: int) -> float:
        c = self.cfg
        warm = max(1, int(c.total_steps * c.warmup_frac))
        decay_n = max(1, int(c.total_steps * c.decay_frac))
        stable_end = c.total_steps - decay_n

        if step < warm:
            return (step + 1) / warm
        if step < stable_end:
            return 1.0
        # decay phase
        t = (step - stable_end) / decay_n
        t = min(1.0, max(0.0, t))
        if c.decay == "cosine":
            m = 0.5 * (1.0 + math.cos(math.pi * t))
        else:  # linear
            m = 1.0 - t
        # never go below min_lr_ratio * base
        return max(c.min_lr_ratio, m)

    def _apply(self, mult: float) -> None:
        for base, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base * mult
        self.last_lrs = [g["lr"] for g in self.optimizer.param_groups]

    # -- public API -----------------------------------------------------
    def step(self) -> None:
        # apply multiplier for the step we're about to take, then advance
        self._apply(self._multiplier(self.step_count))
        self.step_count += 1

    def get_last_lr(self) -> list[float]:
        return list(self.last_lrs)

    def state_dict(self) -> dict:
        return {
            "step": self.step_count,
            "base_lrs": list(self.base_lrs),
            "cfg": {
                "total_steps": self.cfg.total_steps,
                "warmup_frac": self.cfg.warmup_frac,
                "stable_frac": self.cfg.stable_frac,
                "decay_frac": self.cfg.decay_frac,
                "decay": self.cfg.decay,
                "min_lr_ratio": self.cfg.min_lr_ratio,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.step_count = int(state["step"])
        base = state.get("base_lrs")
        if base and len(base) == len(self.optimizer.param_groups):
            self.base_lrs = list(base)
        # Reapply the multiplier for the next step so resume is exact.
        self._apply(self._multiplier(self.step_count))


def build_wsd(optimizer: torch.optim.Optimizer, total_steps: int,
              warmup_frac: float = 0.05, decay_frac: float = 0.10,
              decay: DecayKind = "linear") -> WSDScheduler:
    cfg = WSDConfig(
        total_steps=total_steps,
        warmup_frac=warmup_frac,
        stable_frac=1.0 - warmup_frac - decay_frac,
        decay_frac=decay_frac,
        decay=decay,
    )
    return WSDScheduler(optimizer, cfg)
