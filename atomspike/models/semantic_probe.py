"""Small screen-only classifier for the crosshair target.

No game state, inventory ID, sidecar label, or goal enters this head. Its
output is a supervised visual probe, not a proof of broader causal reasoning.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.data.visual_concepts import CONCEPT_KINDS


class CrosshairConceptProbe(nn.Module):
    def __init__(self, d_model: int, hidden: int = 48):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(int(d_model), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), len(CONCEPT_KINDS)),
        )

    def forward(self, tokens: Tensor, hw: tuple[int, int] | None) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("visual tokens must have shape [B,N,D]")
        n = int(tokens.size(1))
        if hw is None or int(hw[0]) * int(hw[1]) != n:
            raise ValueError("crosshair probe requires a spatial token layout")
        gh, gw = int(hw[0]), int(hw[1])
        grid = tokens.transpose(1, 2).reshape(tokens.size(0), tokens.size(2), gh, gw)
        # At the crosshair, a small neighborhood is less brittle than a
        # single downsampled token; it still cannot see state/teacher labels.
        center = grid[:, :, max(0, gh // 2 - 1) : min(gh, gh // 2 + 2), max(0, gw // 2 - 1) : min(gw, gw // 2 + 2)]
        return self.head(F.adaptive_avg_pool2d(center, (1, 1)).flatten(1))
