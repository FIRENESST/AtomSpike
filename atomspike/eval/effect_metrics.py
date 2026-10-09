"""Held-out diagnostics for observed, not interventional, action outcomes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atomspike.data.effects import EFFECT_EVENTS


class EffectBrierMeter:
    """Compare equal-capacity action-aware/action-blind heads with a prior."""

    def __init__(self, active: NDArray[np.bool_], prior: NDArray[np.floating]):
        self.active = np.asarray(active, dtype=bool).reshape(-1)
        self.prior = np.asarray(prior, dtype=np.float64).reshape(-1)
        if self.active.size != len(EFFECT_EVENTS) or self.prior.size != len(EFFECT_EVENTS):
            raise ValueError("effect metric dimensions must match the event ontology")
        if not np.isfinite(self.prior).all() or np.any((self.prior < 0) | (self.prior > 1)):
            raise ValueError("effect prior must contain finite probabilities")
        self.windows = 0
        self.positive = np.zeros(len(EFFECT_EVENTS), dtype=np.float64)
        self.conditioned_sq = np.zeros(len(EFFECT_EVENTS), dtype=np.float64)
        self.visual_sq = np.zeros(len(EFFECT_EVENTS), dtype=np.float64)
        self.prior_sq = np.zeros(len(EFFECT_EVENTS), dtype=np.float64)

    def update(self, target: NDArray, conditioned: NDArray, visual: NDArray) -> None:
        y = np.asarray(target, dtype=np.float64)
        p = np.asarray(conditioned, dtype=np.float64)
        q = np.asarray(visual, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != len(EFFECT_EVENTS) or p.shape != y.shape or q.shape != y.shape:
            raise ValueError("effect metric batches must share [B,n_events] shape")
        if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("effect metrics cannot contain NaN/Inf")
        if np.any((y < 0) | (y > 1)) or np.any((p < 0) | (p > 1)) or np.any((q < 0) | (q > 1)):
            raise ValueError("effect targets and probabilities must be in [0,1]")
        self.windows += int(y.shape[0])
        self.positive += y.sum(axis=0)
        self.conditioned_sq += ((p - y) ** 2).sum(axis=0)
        self.visual_sq += ((q - y) ** 2).sum(axis=0)
        self.prior_sq += ((self.prior[None, :] - y) ** 2).sum(axis=0)

    def report(self) -> dict:
        active = np.flatnonzero(self.active)
        total = int(self.windows * len(active))
        if total == 0:
            return {"n": 0, "conditioned": 0.0, "visual_only": 0.0, "prior": 0.0, "by_event": {}}
        by_event = {}
        for i in active:
            by_event[EFFECT_EVENTS[i]] = {
                "n": self.windows,
                "positive": int(self.positive[i]),
                "negative": int(self.windows - self.positive[i]),
                "conditioned": float(self.conditioned_sq[i] / self.windows),
                "visual_only": float(self.visual_sq[i] / self.windows),
                "prior": float(self.prior_sq[i] / self.windows),
            }
        return {
            "n": total,
            "conditioned": float(self.conditioned_sq[active].sum() / total),
            "visual_only": float(self.visual_sq[active].sum() / total),
            "prior": float(self.prior_sq[active].sum() / total),
            "by_event": by_event,
        }
