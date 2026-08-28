"""Transformer reasoner over visual tokens + game-state tokens (5Hz path)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import ReasonerConfig


class SemanticReasoner(nn.Module):
    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_state_tokens = cfg.n_state_tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=int(cfg.d_model * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.blocks = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.game_state_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model * cfg.n_state_tokens),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, visual_tokens: Tensor, game_state: Tensor) -> Tensor:
        """
        visual_tokens: [B, N, D]
        game_state: [B, S]
        returns context: [B, D]
        """
        b = visual_tokens.size(0)
        state_tokens = self.state_proj(game_state).view(b, self.n_state_tokens, self.d_model)
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, visual_tokens, state_tokens], dim=1)
        seq = self.blocks(seq)
        return self.norm(seq[:, 0])
