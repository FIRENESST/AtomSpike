"""ANN policy head over fused context."""

from __future__ import annotations

import numpy as np
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
        self.trace_enabled = False
        self.last_trace: dict | None = None

    def forward(
        self,
        context: Tensor,
        tokens: Tensor | None = None,
        *,
        sample_actions: bool = False,
        action_temperature: float = 1.0,
        deterministic_action: bool = False,
    ) -> dict[str, object]:
        if sample_actions and tokens is not None:
            raise ValueError("sample_actions and teacher tokens are mutually exclusive")
        h = context + self.pre(context)
        sampled_tokens = None
        if sample_actions:
            logits, sampled_tokens = self.decoder.sample(
                h,
                temperature=action_temperature,
                deterministic=deterministic_action,
            )
        else:
            logits = self.decoder(h, tokens)
        value = self.value_head(h).squeeze(-1)
        if self.trace_enabled and (not self.training):
            act = h.detach().float()[0].reshape(-1).cpu().numpy()
            scale = float(np.max(np.abs(act))) + 1e-6
            mem = np.clip(np.abs(act) / scale, 0.0, 1.0).astype(np.float32)
            spk = (act > 0.15 * scale).astype(np.float32)
            self.last_trace = {
                "kind": "ann",
                "in_spike": spk,
                "in_mem": mem,
                "layers": [],
                "sparsity": float(spk.mean()),
                "v_th": 1.0,
            }
        return {
            "logits": logits,
            "value": value,
            "spikes": None,
            "sampled_tokens": sampled_tokens,
        }
