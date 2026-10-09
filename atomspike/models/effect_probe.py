"""Small observational action-outcome probe, not a causal planner.

The visual concept distribution is detached: outcome prediction cannot make
the supervised concept names drift into arbitrary task features. Counterfactual
claims require separately collected interventions and are intentionally absent.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.data.effects import EFFECT_ACTIONS, EFFECT_EVENTS
from atomspike.data.visual_concepts import CONCEPT_KINDS
from atomspike.models.actions import ActionSpec, KEY_HOLD, KEY_PRESS


def action_macro_indices(tokens: Tensor, spec: ActionSpec) -> Tensor:
    """Torch twin of the offline diagnostic macro rule."""
    if tokens.ndim != 2 or tokens.size(1) != spec.n_slots:
        raise ValueError("action tokens must have shape [B,n_slots]")
    active = (tokens == KEY_PRESS) | (tokens == KEY_HOLD)
    macro = torch.full((tokens.size(0),), EFFECT_ACTIONS.index("idle"), device=tokens.device, dtype=torch.long)
    look = (tokens[:, spec.n_keys] != spec.mouse_center) | (tokens[:, spec.n_keys + 1] != spec.mouse_center)
    macro = torch.where(look, EFFECT_ACTIONS.index("look"), macro)
    jump = [i for i, name in enumerate(spec.key_names) if name == "SPACE"]
    if jump:
        macro = torch.where(active[:, jump[0]], EFFECT_ACTIONS.index("jump"), macro)
    move = [i for i, name in enumerate(spec.key_names) if name in {"W", "A", "S", "D"}]
    if move:
        macro = torch.where(active[:, move].any(dim=1), EFFECT_ACTIONS.index("move"), macro)
    use = [i for i, name in enumerate(spec.key_names) if name == "E"]
    if spec.n_mouse_buttons > 1:
        use.append(spec.n_keys + 3)
    if use:
        macro = torch.where(active[:, use].any(dim=1), EFFECT_ACTIONS.index("use"), macro)
    if spec.n_mouse_buttons > 0:
        macro = torch.where(active[:, spec.n_keys + 2], EFFECT_ACTIONS.index("attack"), macro)
    return macro


class ConditionalEffectProbe(nn.Module):
    """Screen-only outcome diagnostic; privileged telemetry supplies labels only."""

    def __init__(self, visual_dim: int, hidden: int = 64):
        super().__init__()
        input_dim = len(CONCEPT_KINDS) + int(visual_dim) + len(EFFECT_ACTIONS)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(EFFECT_EVENTS)),
        )

    def forward(
        self,
        concept_logits: Tensor,
        visual_features: Tensor,
        action_tokens: Tensor | None,
        spec: ActionSpec,
    ) -> Tensor:
        if visual_features.ndim != 2 or visual_features.size(0) != concept_logits.size(0):
            raise ValueError("visual features must have shape [B,D] matching concepts")
        concept = F.softmax(concept_logits.float().detach(), dim=-1)
        macro = (
            visual_features.new_zeros(visual_features.size(0), len(EFFECT_ACTIONS))
            if action_tokens is None else F.one_hot(action_macro_indices(action_tokens, spec), len(EFFECT_ACTIONS)).float()
        )
        # Keep this diagnostic independent of sidecar state, inventory IDs and
        # the upstream policy. The encoder can be trained by its own losses.
        features = torch.cat((concept, visual_features.float().detach(), macro), dim=-1)
        return self.head(features)
