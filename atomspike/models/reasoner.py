"""Transformer reasoner over visual tokens + game-state tokens (5Hz path)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import ReasonerConfig
from atomspike.models.activations import Act
from atomspike.models.attention import MultiHeadSelfAttention


class ReasonerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = Act("gelu")
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.fc2(self.act(self.fc1(self.norm2(x)))))
        return x


class SemanticReasoner(nn.Module):
    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_state_tokens = cfg.n_state_tokens
        self.blocks = nn.ModuleList(
            ReasonerBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.game_state_dim, cfg.d_model),
            Act("gelu"),
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
        for block in self.blocks:
            seq = block(seq)
        return self.norm(seq[:, 0])
