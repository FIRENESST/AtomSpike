"""Transformer reasoner over visual tokens + game-state tokens (10Hz path)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.config import ReasonerConfig
from atomspike.models.activations import Act
from atomspike.models.attention import MultiHeadSelfAttention
from atomspike.models.drop import DropPath


class ReasonerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float = 0.0,
        *,
        identity_init: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = Act("gelu")
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        if identity_init:
            nn.init.zeros_(self.attn.out.weight)
            if self.attn.out.bias is not None:
                nn.init.zeros_(self.attn.out.bias)
            nn.init.zeros_(self.fc2.weight)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
        self.identity_init = bool(identity_init)
        self._skip_eval: bool | None = None

    def is_eval_identity(self) -> bool:
        if self.training or not self.identity_init:
            self._skip_eval = None
            return False
        if self._skip_eval is None:
            z_fc = bool((self.fc2.weight.detach() == 0).all().item())
            z_attn = bool((self.attn.out.weight.detach() == 0).all().item())
            self._skip_eval = z_fc and z_attn
        return self._skip_eval

    def forward(self, x: Tensor) -> Tensor:
        if self.is_eval_identity():
            return x
        x = x + self.drop_path(self.drop(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.drop(self.fc2(self.act(self.fc1(self.norm2(x))))))
        return x


class SemanticReasoner(nn.Module):
    def __init__(self, cfg: ReasonerConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_state_tokens = cfg.n_state_tokens
        self.token_drop = float(getattr(cfg, "token_drop", 0.0) or 0.0)
        n_layers = max(1, int(cfg.n_layers))
        drop_path = float(getattr(cfg, "drop_path", 0.0) or 0.0)
        self.blocks = nn.ModuleList(
            ReasonerBlock(
                cfg.d_model,
                cfg.n_heads,
                cfg.mlp_ratio,
                cfg.dropout,
                drop_path * (i / max(1, n_layers - 1)),
                identity_init=i >= 2,
            )
            for i in range(n_layers)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.game_state_dim, cfg.d_model),
            Act("gelu"),
            nn.Linear(cfg.d_model, cfg.d_model * cfg.n_state_tokens),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.norm = nn.LayerNorm(cfg.d_model)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _drop_visual(self, visual: Tensor) -> Tensor:
        if (not self.training) or self.token_drop <= 0.0:
            return visual
        keep = 1.0 - self.token_drop
        mask = visual.new_empty(visual.size(0), visual.size(1), 1).bernoulli_(keep)
        return visual * mask / keep

    def forward(
        self,
        visual_tokens: Tensor,
        game_state: Tensor,
        plan_tokens: Tensor | None = None,
    ) -> Tensor:
        """
        visual_tokens: [B, N, D]
        game_state: [B, S]
        plan_tokens: optional [B, P, D] self-semantic task tokens
        returns context: [B, D]
        """
        b = visual_tokens.size(0)
        visual_tokens = self._drop_visual(visual_tokens)
        state_tokens = self.state_proj(game_state).view(b, self.n_state_tokens, self.d_model)
        cls = self.cls.expand(b, -1, -1)
        parts = [cls, visual_tokens, state_tokens]
        if plan_tokens is not None:
            parts.append(plan_tokens)
        seq = torch.cat(parts, dim=1)
        for block in self.blocks:
            seq = block(seq)
        return self.norm(seq[:, 0])
