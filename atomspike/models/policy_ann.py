"""ANN policy head over fused context."""

from __future__ import annotations

from torch import Tensor, nn

from atomspike.config import PolicyConfig
from atomspike.models.actions import ActionSpec
from atomspike.models.activations import Act
from atomspike.models.decoder import ActionDecoder


class ANNPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, spec: ActionSpec):
        super().__init__()
        self.pre = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            Act("gelu"),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.decoder = ActionDecoder(cfg.d_model, spec, mode=cfg.action_decode)
        self.value_head = nn.Linear(cfg.d_model, 1)

    def forward(self, context: Tensor, tokens: Tensor | None = None) -> dict[str, object]:
        h = context + self.pre(context)
        logits = self.decoder(h, tokens)
        value = self.value_head(h).squeeze(-1)
        return {"logits": logits, "value": value, "spikes": None}
