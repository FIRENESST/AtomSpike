"""Cheap inventory / backpack sense: item IDs + a tiny LIF residual.

Sidecar sends CRC32(item path) plus a few name strings. This module maps
those onto a small catalog (named vanilla items + hash buckets), embeds
hotbar / pack / menu / hand / hover / carry, and runs a 2-step LIF. A
frozen use-table (harvest / weapon / food / place / LMB vs RMB) is mixed
in through a zero-init Linear so old checkpoints stay put until it learns.

The 19-d game_state vector is unchanged. Cursor / hovered_slot / gui_open
already live there; item identity is this side channel.

Live path: 10Hz with perception, and every 30Hz policy tick while a GUI is
open (the LIF is tiny; CNN/reasoner stay cached).
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from atomspike.config import InventoryConfig
from atomspike.models.lif import LIFCell

INV_PLAYER = 36
INV_MENU = 27
INV_SLOTS = INV_PLAYER + INV_MENU

# Common early-game / tool / food ids (vanilla path, no namespace).
NAMED_ITEMS: tuple[str, ...] = (
    "",
    "air",
    "stone",
    "granite",
    "diorite",
    "andesite",
    "dirt",
    "grass_block",
    "cobblestone",
    "oak_log",
    "oak_planks",
    "oak_sapling",
    "oak_leaves",
    "sand",
    "gravel",
    "flint",
    "coal",
    "charcoal",
    "raw_iron",
    "iron_ingot",
    "raw_gold",
    "gold_ingot",
    "diamond",
    "stick",
    "torch",
    "crafting_table",
    "furnace",
    "chest",
    "oak_door",
    "oak_boat",
    "wheat_seeds",
    "wheat",
    "bread",
    "apple",
    "golden_apple",
    "porkchop",
    "cooked_porkchop",
    "beef",
    "cooked_beef",
    "chicken",
    "cooked_chicken",
    "cod",
    "cooked_cod",
    "salmon",
    "cooked_salmon",
    "egg",
    "sugar",
    "bucket",
    "water_bucket",
    "lava_bucket",
    "milk_bucket",
    "wooden_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "golden_pickaxe",
    "diamond_pickaxe",
    "wooden_axe",
    "stone_axe",
    "iron_axe",
    "wooden_shovel",
    "stone_shovel",
    "iron_shovel",
    "wooden_sword",
    "stone_sword",
    "iron_sword",
    "diamond_sword",
    "wooden_hoe",
    "stone_hoe",
    "bow",
    "arrow",
    "shield",
    "fishing_rod",
    "bone",
    "string",
    "feather",
    "leather",
    "gunpowder",
    "redstone",
    "glowstone_dust",
    "obsidian",
    "crying_obsidian",
    "ender_pearl",
    "blaze_rod",
    "nether_wart",
    "ancient_debris",
    "netherite_ingot",
    "oak_sign",
    "glass",
    "glass_bottle",
    "potion",
    "book",
    "paper",
    "map",
    "compass",
    "clock",
    "shears",
    "flint_and_steel",
)

_NAMED_INDEX = {name: i for i, name in enumerate(NAMED_ITEMS) if name}
N_NAMED = len(NAMED_ITEMS)
_CODE_TO_NAMED: dict[int, int] = {
    zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF: i
    for i, name in enumerate(NAMED_ITEMS)
    if name
}


def normalize_item_path(name: str | None) -> str:
    s = str(name or "").strip().lower()
    if s.startswith("minecraft:"):
        s = s[10:]
    if s in {"", "air", "empty", "none"}:
        return ""
    return s


def item_code_from_path(name: str | None) -> int:
    key = normalize_item_path(name)
    if not key:
        return 0
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def item_id_from_code(code: int | None, n_buckets: int = 48) -> int:
    """Map a sidecar CRC32 (or 0) onto the catalog + hash buckets."""
    c = int(code or 0) & 0xFFFFFFFF
    if c == 0:
        return 0
    named = _CODE_TO_NAMED.get(c)
    if named is not None:
        return int(named)
    n_buckets = max(1, int(n_buckets))
    return N_NAMED + (c % n_buckets)


def item_id_from_name(name: str | None, n_buckets: int = 48) -> int:
    key = normalize_item_path(name)
    if not key:
        return 0
    if key in _NAMED_INDEX:
        return int(_NAMED_INDEX[key])
    return item_id_from_code(item_code_from_path(key), n_buckets=n_buckets)


def item_id_from_any(value: Any, n_buckets: int = 48) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return item_id_from_name(value, n_buckets=n_buckets)
    if isinstance(value, (bytes, bytearray)):
        return item_id_from_name(value.decode("utf-8", errors="ignore"), n_buckets=n_buckets)
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return 0
        return item_id_from_code(int(value), n_buckets=n_buckets)
    except (TypeError, ValueError):
        return 0


def item_label_from_id(item_id: int, n_buckets: int = 48) -> str:
    tid = int(item_id)
    if 0 <= tid < N_NAMED:
        name = NAMED_ITEMS[tid]
        return name if name else "-"
    extra = tid - N_NAMED
    n_buckets = max(1, int(n_buckets))
    if extra >= 0:
        return f"#{extra % n_buckets}"
    return "-"


def n_item_ids(n_buckets: int = 48) -> int:
    return N_NAMED + max(1, int(n_buckets))


# harvest, melee, ranged, edible, placeable, station, lmb-use, rmb-use
USE_DIM = 8
_TOOL_TIER = {
    "wooden": 0.25,
    "stone": 0.5,
    "iron": 0.75,
    "golden": 0.35,
    "diamond": 1.0,
    "netherite": 1.0,
}
_FOOD = {
    "apple",
    "golden_apple",
    "bread",
    "wheat",
    "porkchop",
    "cooked_porkchop",
    "beef",
    "cooked_beef",
    "chicken",
    "cooked_chicken",
    "cod",
    "cooked_cod",
    "salmon",
    "cooked_salmon",
    "egg",
    "milk_bucket",
    "potion",
}
_PLACE = {
    "torch",
    "crafting_table",
    "furnace",
    "chest",
    "oak_door",
    "oak_boat",
    "oak_sapling",
    "oak_sign",
    "oak_log",
    "oak_planks",
    "cobblestone",
    "stone",
    "dirt",
    "grass_block",
    "sand",
    "gravel",
    "glass",
    "obsidian",
    "crying_obsidian",
}
_STATION = {"crafting_table", "furnace", "chest"}
_RMB = {
    "bucket",
    "water_bucket",
    "lava_bucket",
    "milk_bucket",
    "bow",
    "shield",
    "fishing_rod",
    "ender_pearl",
    "flint_and_steel",
    "shears",
    "potion",
    "oak_boat",
    "oak_door",
    "oak_sign",
    "compass",
    "clock",
    "map",
}


def item_use_row(name: str | None) -> list[float]:
    """Fixed prior: what a vanilla item is for / which mouse button uses it."""
    key = normalize_item_path(name)
    v = [0.0] * USE_DIM
    if not key:
        return v
    for prefix, tier in _TOOL_TIER.items():
        stem = prefix + "_"
        if key.startswith(stem):
            rest = key[len(stem) :]
            v[0] = float(tier)
            if rest == "sword":
                v[1] = 1.0
                v[6] = 1.0
            elif rest == "pickaxe":
                v[6] = 1.0
            elif rest in {"axe", "shovel", "hoe"}:
                v[6] = 0.85
            return v
    if key == "bow" or key == "arrow":
        v[2] = 1.0
        v[6] = 1.0 if key == "bow" else 0.0
        v[7] = 1.0 if key == "bow" else 0.0
        return v
    if key in _FOOD:
        v[3] = 1.0
        v[7] = 1.0
        return v
    if key in _STATION:
        v[4] = 1.0
        v[5] = 1.0
        v[7] = 1.0
        return v
    if key in _PLACE:
        v[4] = 1.0
        v[7] = 1.0
        return v
    if key in _RMB:
        v[7] = 1.0
        return v
    if key in {"stick", "coal", "charcoal", "iron_ingot", "gold_ingot", "diamond", "string", "leather"}:
        return v
    return v


def item_use_table(n_buckets: int = 48) -> Tensor:
    n = n_item_ids(n_buckets)
    tbl = torch.zeros(n, USE_DIM)
    for i, name in enumerate(NAMED_ITEMS):
        tbl[i] = torch.tensor(item_use_row(name), dtype=torch.float32)
    return tbl


def inventory_pack_from_signal(signal: dict[str, Any] | None, n_buckets: int = 48) -> dict[str, Any]:
    """CPU pack for HDF5 / live tensors. Empty when sidecar has no item fields."""
    raw = signal or {}
    inv = np.zeros((INV_SLOTS,), dtype=np.int64)
    codes = raw.get("inv")
    if isinstance(codes, (list, tuple)):
        n_player = min(INV_PLAYER, len(codes))
        for i in range(n_player):
            inv[i] = item_id_from_any(codes[i], n_buckets=n_buckets)
        if len(codes) > INV_PLAYER:
            extra = min(INV_MENU, len(codes) - INV_PLAYER)
            for i in range(extra):
                inv[INV_PLAYER + i] = item_id_from_any(codes[INV_PLAYER + i], n_buckets=n_buckets)
    menu = raw.get("menu")
    if isinstance(menu, (list, tuple)):
        n_menu = min(INV_MENU, len(menu))
        for i in range(n_menu):
            inv[INV_PLAYER + i] = item_id_from_any(menu[i], n_buckets=n_buckets)
    return {
        "inv_ids": inv,
        "hand_id": int(item_id_from_name(str(raw.get("hand_item") or ""), n_buckets=n_buckets)),
        "hover_id": int(item_id_from_name(str(raw.get("hover_item") or ""), n_buckets=n_buckets)),
        "carry_id": int(item_id_from_name(str(raw.get("carry_item") or ""), n_buckets=n_buckets)),
    }


def inventory_tensors_from_signal(
    signal: dict[str, Any] | None,
    *,
    device: torch.device,
    batch: int = 1,
    n_buckets: int = 48,
) -> dict[str, Tensor]:
    pack = inventory_pack_from_signal(signal, n_buckets=n_buckets)
    b = max(1, int(batch))
    inv = torch.from_numpy(pack["inv_ids"]).to(device=device, dtype=torch.long).unsqueeze(0).expand(b, -1).contiguous()
    def _id(v: int) -> Tensor:
        return torch.full((b,), int(v), device=device, dtype=torch.long)

    return {
        "inv_ids": inv,
        "hand_id": _id(pack["hand_id"]),
        "hover_id": _id(pack["hover_id"]),
        "carry_id": _id(pack["carry_id"]),
    }


def inventory_forward_kwargs(batch: dict, device: torch.device) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    if "inv_ids" in batch:
        out["inv_ids"] = batch["inv_ids"].to(device, non_blocking=True)
    for key in ("hand_id", "hover_id", "carry_id"):
        if key not in batch:
            continue
        t = batch[key].to(device, non_blocking=True)
        out[key] = t.reshape(t.shape[0], -1)[:, 0] if t.ndim > 1 else t
    return out


def _gui_index(state_dim: int) -> int | None:
    from atomspike.capture.mc_state import MC_STATE_FIELDS

    if "gui_open" not in MC_STATE_FIELDS:
        return None
    idx = MC_STATE_FIELDS.index("gui_open")
    return idx if idx < int(state_dim) else None


def _default_slot_xy(n_slots: int) -> Tensor:
    """Rough vanilla survival inventory / chest layout in window units."""
    n = max(1, int(n_slots))
    xy = torch.zeros(n, 2)
    for i in range(n):
        if i < 9:
            col, row = i, 0
            y = 0.84
        elif i < INV_PLAYER:
            col, row = (i - 9) % 9, (i - 9) // 9
            y = 0.70 - row * 0.08
        else:
            col, row = (i - INV_PLAYER) % 9, (i - INV_PLAYER) // 9
            y = 0.40 - row * 0.08
        xy[i, 0] = 0.22 + col * (0.56 / 8.0)
        xy[i, 1] = y
    return xy


class InventorySense(nn.Module):
    """Item-id LIF residual. ~a few thousand params; zero bias until trained."""

    def __init__(self, cfg: InventoryConfig, d_model: int, state_dim: int):
        super().__init__()
        self.n_slots = max(INV_PLAYER, int(cfg.n_slots or INV_SLOTS))
        self.n_buckets = max(1, int(cfg.n_buckets))
        self.n_ids = n_item_ids(self.n_buckets)
        self.item_dim = max(8, int(cfg.item_dim))
        self.hidden = max(8, int(cfg.hidden))
        self.lif_steps = max(1, int(cfg.lif_steps))
        self.state_dim = int(state_dim)
        self.gui_index = _gui_index(self.state_dim)
        self.cursor_index = None
        self.hover_slot_index = None
        from atomspike.capture.mc_state import MC_STATE_FIELDS

        if "cursor_x" in MC_STATE_FIELDS:
            cx = MC_STATE_FIELDS.index("cursor_x")
            self.cursor_index = cx if cx + 1 < self.state_dim else None
        if "hovered_slot" in MC_STATE_FIELDS:
            hs = MC_STATE_FIELDS.index("hovered_slot")
            self.hover_slot_index = hs if hs < self.state_dim else None
        self.item_emb = nn.Embedding(self.n_ids, self.item_dim, padding_idx=0)
        self.register_buffer("use_tbl", item_use_table(self.n_buckets), persistent=True)
        self.use_mix = nn.Linear(USE_DIM, self.item_dim, bias=False)
        nn.init.zeros_(self.use_mix.weight)
        self.use_scale = nn.Parameter(torch.zeros(1))
        # hotbar + pack + menu + hand + hover + carry + gui + cursor_xy + hovered_slot
        feat = self.item_dim * 6 + 4
        self.proj = nn.Linear(feat, self.hidden)
        self.lif = LIFCell(
            tau=float(cfg.lif_tau),
            v_th=float(cfg.lif_v_th),
            thresh_shape=(self.hidden,),
        )
        self.bias_out = nn.Linear(self.hidden, int(d_model), bias=False)
        nn.init.zeros_(self.bias_out.weight)
        self.pointer_scale = nn.Parameter(torch.zeros(1))
        self.slot_xy = nn.Parameter(_default_slot_xy(self.n_slots))

    def _pool(self, emb: Tensor, mask: Tensor) -> Tensor:
        w = mask.to(dtype=emb.dtype).unsqueeze(-1)
        num = (emb * w).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1.0)
        return num / den

    def _ids(self, value: Tensor | None, batch: int, device, slots: int | None = None) -> Tensor:
        if value is None:
            shape = (batch, slots) if slots is not None else (batch,)
            return torch.zeros(shape, device=device, dtype=torch.long)
        t = value.to(device=device, dtype=torch.long)
        if slots is None:
            return t.reshape(batch, -1)[:, 0].clamp(0, self.n_ids - 1)
        t = t.reshape(batch, -1)
        if t.size(1) < slots:
            t = F.pad(t, (0, slots - t.size(1)))
        elif t.size(1) > slots:
            t = t[:, :slots]
        return t.clamp(0, self.n_ids - 1)

    def _embed(self, ids: Tensor, dtype) -> Tensor:
        use = self.use_tbl[ids.clamp(0, self.use_tbl.size(0) - 1)].to(device=ids.device, dtype=dtype)
        return self.item_emb(ids) + self.use_scale * self.use_mix(use)

    def _point(
        self,
        emb: Tensor,
        cursor: Tensor,
        hovered: Tensor,
        gui: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Soft slot under the cursor. Meaning is learned via pointer CE + mouse CE."""
        xy = self.slot_xy
        if xy.size(0) != emb.size(1):
            xy = xy[: emb.size(1)] if xy.size(0) > emb.size(1) else F.pad(xy, (0, 0, 0, emb.size(1) - xy.size(0)))
        dist = (cursor.unsqueeze(1) - xy.unsqueeze(0)).pow(2).sum(dim=-1)
        hover_idx = (hovered.reshape(-1) * 90.0 - 1.0).round().long()
        valid = (hover_idx >= 0) & (hover_idx < emb.size(1))
        boost = dist.new_zeros(dist.shape)
        safe = hover_idx.clamp(0, max(0, emb.size(1) - 1)).view(-1, 1)
        boost.scatter_(1, safe, 4.0 * valid.to(dtype=dist.dtype).view(-1, 1))
        logits = -dist * 12.0 + boost
        attn = logits.softmax(dim=-1) * gui.clamp(0.0, 1.0)
        pointed = (attn.unsqueeze(-1) * emb).sum(dim=1)
        return pointed, logits, hover_idx

    def forward(
        self,
        inv_ids: Tensor | None,
        hand_id: Tensor | None,
        hover_id: Tensor | None,
        carry_id: Tensor | None,
        state: Tensor,
        mem: Tensor | None = None,
    ) -> dict[str, Tensor]:
        b = int(state.size(0))
        device = state.device
        ids = self._ids(inv_ids, b, device, self.n_slots)
        occ = (ids > 0).to(dtype=state.dtype)
        emb = self._embed(ids, state.dtype)
        n_hot = min(9, self.n_slots)
        n_pack = min(INV_PLAYER, self.n_slots)
        hotbar = self._pool(emb[:, :n_hot], occ[:, :n_hot])
        pack = self._pool(emb[:, n_hot:n_pack], occ[:, n_hot:n_pack]) if n_pack > n_hot else hotbar.new_zeros(b, self.item_dim)
        menu = self._pool(emb[:, n_pack:], occ[:, n_pack:]) if self.n_slots > n_pack else hotbar.new_zeros(b, self.item_dim)
        hand = self._embed(self._ids(hand_id, b, device), state.dtype)
        hover = self._embed(self._ids(hover_id, b, device), state.dtype)
        carry = self._embed(self._ids(carry_id, b, device), state.dtype)
        gui = state.new_zeros(b, 1)
        cursor = state.new_zeros(b, 2)
        hovered = state.new_zeros(b, 1)
        if self.gui_index is not None:
            gui = state[:, self.gui_index : self.gui_index + 1]
        if self.cursor_index is not None:
            cursor = state[:, self.cursor_index : self.cursor_index + 2]
        if self.hover_slot_index is not None:
            hovered = state[:, self.hover_slot_index : self.hover_slot_index + 1]
        pointed, pointer_logits, pointer_idx = self._point(emb, cursor, hovered, gui)
        hover = hover + self.pointer_scale * pointed
        feat = torch.cat([hotbar, pack, menu, hand, hover, carry, gui, cursor, hovered], dim=-1)
        h = F.gelu(self.proj(feat))
        spike_acc = torch.zeros_like(h)
        mem_t = mem
        for _ in range(self.lif_steps):
            spk, mem_t = self.lif(h, mem_t)
            spike_acc = spike_acc + spk
        rate = spike_acc / float(self.lif_steps)
        bias = self.bias_out(rate * h)
        n_items = occ[:, :n_pack].sum(dim=1)
        return {
            "bias": bias,
            "mem": mem_t,
            "spike": rate,
            "n_items": n_items,
            "gui": gui.reshape(b),
            "pointer_logits": pointer_logits,
            "pointer_idx": pointer_idx,
        }
