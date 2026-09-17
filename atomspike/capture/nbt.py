"""Minimal big-endian NBT reader for Minecraft playerdata / level.dat."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

TAG_END = 0
TAG_BYTE = 1
TAG_SHORT = 2
TAG_INT = 3
TAG_LONG = 4
TAG_FLOAT = 5
TAG_DOUBLE = 6
TAG_BYTE_ARRAY = 7
TAG_STRING = 8
TAG_LIST = 9
TAG_COMPOUND = 10
TAG_INT_ARRAY = 11
TAG_LONG_ARRAY = 12


class _R:
    def __init__(self, data: bytes):
        self.b = data
        self.i = 0

    def u8(self) -> int:
        v = self.b[self.i]
        self.i += 1
        return v

    def unpack(self, fmt: str):
        n = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, self.b, self.i)
        self.i += n
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        n = self.unpack(">H")
        s = self.b[self.i : self.i + n]
        self.i += n
        return s.decode("utf-8", errors="replace")

    def payload(self, tag: int) -> Any:
        if tag == TAG_BYTE:
            return self.unpack(">b")
        if tag == TAG_SHORT:
            return self.unpack(">h")
        if tag == TAG_INT:
            return self.unpack(">i")
        if tag == TAG_LONG:
            return self.unpack(">q")
        if tag == TAG_FLOAT:
            return self.unpack(">f")
        if tag == TAG_DOUBLE:
            return self.unpack(">d")
        if tag == TAG_BYTE_ARRAY:
            n = self.unpack(">i")
            v = self.b[self.i : self.i + n]
            self.i += n
            return v
        if tag == TAG_STRING:
            return self.string()
        if tag == TAG_LIST:
            inner = self.u8()
            n = self.unpack(">i")
            return [self.payload(inner) for _ in range(max(0, n))]
        if tag == TAG_COMPOUND:
            out: dict[str, Any] = {}
            while True:
                t = self.u8()
                if t == TAG_END:
                    break
                name = self.string()
                out[name] = self.payload(t)
            return out
        if tag == TAG_INT_ARRAY:
            n = self.unpack(">i")
            return [self.unpack(">i") for _ in range(max(0, n))]
        if tag == TAG_LONG_ARRAY:
            n = self.unpack(">i")
            return [self.unpack(">q") for _ in range(max(0, n))]
        raise ValueError(f"unknown nbt tag {tag}")


def load_nbt(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    r = _R(raw)
    tag = r.u8()
    if tag != TAG_COMPOUND:
        raise ValueError("root nbt is not a compound")
    _name = r.string()
    return r.payload(TAG_COMPOUND)


def extract_player(compound: dict[str, Any]) -> dict[str, Any] | None:
    if "Health" in compound or "Pos" in compound:
        return compound
    data = compound.get("Data")
    if isinstance(data, dict) and isinstance(data.get("Player"), dict):
        return data["Player"]
    return None


def uuid_from_int_array(arr: list[int]) -> str:
    parts = [int(x) & 0xFFFFFFFF for x in arr[:4]]
    while len(parts) < 4:
        parts.append(0)
    n = (parts[0] << 96) | (parts[1] << 64) | (parts[2] << 32) | parts[3]
    s = f"{n:032x}"
    return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}"


def singleplayer_uuid(level_root: dict[str, Any]) -> str | None:
    data = level_root.get("Data") if isinstance(level_root, dict) else None
    if not isinstance(data, dict):
        return None
    raw = data.get("singleplayer_uuid")
    if isinstance(raw, list) and raw:
        return uuid_from_int_array([int(x) for x in raw])
    if isinstance(raw, str) and raw:
        return raw
    return None


def player_to_fields(player: dict[str, Any]) -> dict[str, float | int | str | bool]:
    pos = player.get("Pos") or [0.0, 0.0, 0.0]
    rot = player.get("Rotation") or [0.0, 0.0]
    mot = player.get("Motion") or [0.0, 0.0, 0.0]
    hp = float(player.get("Health", 0.0) or 0.0)
    hunger = int(player.get("foodLevel", 0) or 0)
    sat = float(player.get("foodSaturationLevel", 0.0) or 0.0)
    air = int(player.get("Air", 300) or 0)
    xp_p = float(player.get("XpP", 0.0) or 0.0)
    xp_l = int(player.get("XpLevel", 0) or 0)
    slot = int(player.get("SelectedItemSlot", 0) or 0)
    on_ground = bool(player.get("OnGround", 0))
    inv = player.get("Inventory") or []
    held = ""
    count = 0
    if isinstance(inv, list):
        for it in inv:
            if isinstance(it, dict) and int(it.get("Slot", -1)) == slot:
                held = str(it.get("id", ""))
                count = int(it.get("count", it.get("Count", 0)) or 0)
                break
    eq = player.get("equipment")
    if isinstance(eq, dict) and not held:
        hand = eq.get("mainhand") or eq.get("main_hand") or {}
        if isinstance(hand, dict):
            held = str(hand.get("id", "") or "")
            count = int(hand.get("count", hand.get("Count", 0)) or 0)
    abilities = player.get("abilities") if isinstance(player.get("abilities"), dict) else {}
    flying = bool(abilities.get("flying", 0))
    pose = str(player.get("Pose") or "STANDING").upper()
    return {
        "hp": hp,
        "max_hp": 20.0,
        "hunger": hunger,
        "saturation": sat,
        "air": air,
        "xp_progress": xp_p,
        "xp_level": xp_l,
        "x": float(pos[0]) if len(pos) > 0 else 0.0,
        "y": float(pos[1]) if len(pos) > 1 else 0.0,
        "z": float(pos[2]) if len(pos) > 2 else 0.0,
        "yaw": float(rot[0]) if len(rot) > 0 else 0.0,
        "pitch": float(rot[1]) if len(rot) > 1 else 0.0,
        "vx": float(mot[0]) if len(mot) > 0 else 0.0,
        "vy": float(mot[1]) if len(mot) > 1 else 0.0,
        "vz": float(mot[2]) if len(mot) > 2 else 0.0,
        "on_ground": on_ground,
        "hotbar_slot": slot,
        "selected_item": held,
        "selected_count": count,
        "flying": flying,
        "pose": pose,
        "sneaking": pose == "CROUCHING",
        "sprinting": pose == "SPRINTING",
        "in_water": pose in {"SWIMMING", "SPIN_ATTACK"},
    }
