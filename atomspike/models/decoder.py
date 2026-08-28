"""8-slot action decoder: parallel heads, GRU AR (default), or tiny Transformer AR.

GRU AR is the better 30Hz default: 8 tokens need sequential dependence
(multi-key combos) but not a full decoder stack.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from atomspike.models.actions import ActionSpec


class ActionDecoder(nn.Module):
    def __init__(self, d_model: int, spec: ActionSpec, mode: str = "gru_ar"):
        super().__init__()
        self.spec = spec
        self.mode = mode
        self.slot_embed = nn.Embedding(spec.n_slots, d_model)
        self.heads = nn.ModuleList(nn.Linear(d_model, v) for v in spec.slot_vocabs)
        self.token_embeds = nn.ModuleList(nn.Embedding(v, d_model) for v in spec.slot_vocabs)
        self.rnn: nn.GRUCell | None = None
        self.decoder: nn.TransformerDecoder | None = None
        if mode == "gru_ar":
            self.rnn = nn.GRUCell(d_model, d_model)
        elif mode == "transformer_ar":
            layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        elif mode != "parallel":
            raise ValueError(f"unknown action_decode mode: {mode}")

    def forward(self, context: Tensor, tokens: Tensor | None = None) -> list[Tensor]:
        """
        context: [B, D]
        tokens: [B, 8] teacher tokens (training) or None
        returns list of 8 logits tensors [B, vocab_i]
        """
        if self.mode == "parallel":
            return [head(context) for head in self.heads]
        if self.mode == "gru_ar":
            return self._gru_ar(context, tokens)
        return self._tf_ar(context, tokens)

    def _gru_ar(self, context: Tensor, tokens: Tensor | None) -> list[Tensor]:
        assert self.rnn is not None
        b = context.size(0)
        h = context
        logits_out: list[Tensor] = []
        for i in range(self.spec.n_slots):
            inp = context + self.slot_embed.weight[i].unsqueeze(0).expand(b, -1)
            h = self.rnn(inp, h)
            logits_out.append(self.heads[i](h))
            if tokens is not None:
                tok = tokens[:, i].clamp(0, self.spec.slot_vocabs[i] - 1)
            else:
                tok = logits_out[-1].argmax(dim=-1)
            h = h + self.token_embeds[i](tok)
        return logits_out

    def _tf_ar(self, context: Tensor, tokens: Tensor | None) -> list[Tensor]:
        assert self.decoder is not None
        b = context.size(0)
        memory = context.unsqueeze(1)
        learned = self.slot_embed.weight.unsqueeze(0).expand(b, -1, -1)
        if tokens is None:
            decoded: list[Tensor] = []
            so_far = learned[:, :1]
            for i in range(self.spec.n_slots):
                out = self.decoder(so_far, memory)[:, -1]
                decoded.append(self.heads[i](out))
                pred = decoded[-1].argmax(-1)
                nxt = learned[:, i : i + 1] + self.token_embeds[i](pred).unsqueeze(1)
                so_far = nxt if i == 0 else torch.cat([so_far, nxt], dim=1)
            return decoded
        tok_emb = [
            self.token_embeds[i](tokens[:, i].clamp(0, self.spec.slot_vocabs[i] - 1))
            for i in range(self.spec.n_slots)
        ]
        te = torch.stack(tok_emb, dim=1) + learned
        zeros = torch.zeros_like(te[:, :1])
        tgt = torch.cat([zeros, te[:, :-1]], dim=1)
        n = self.spec.n_slots
        causal = torch.triu(torch.ones(n, n, device=context.device), diagonal=1).bool()
        out = self.decoder(tgt, memory, tgt_mask=causal)
        return [self.heads[i](out[:, i]) for i in range(n)]
