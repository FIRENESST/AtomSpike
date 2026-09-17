"""Minecraft key-state fusion: HUD pixels + NBT saves + local UDP/file sidecar."""

from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from atomspike.capture.nbt import extract_player, load_nbt, player_to_fields, singleplayer_uuid

MC_FACT_FIELDS: tuple[str, ...] = ("x", "y", "z", "yaw", "pitch", "hp", "hunger", "hotbar_slot")
MC_STATE_FIELDS: tuple[str, ...] = (
    "hp",
    "hunger",
    "air",
    "xp_progress",
    "yaw_sin",
    "yaw_cos",
    "pitch_norm",
    "hotbar",
    "on_ground",
    "sneaking",
    "sprinting",
    "in_water",
    "speed",
    "y_norm",
    "selected_count",
    "gui_open",
    "cursor_x",
    "cursor_y",
    "hovered_slot",
)


def default_saves_dir() -> Path:
    dirs = discover_saves_dirs()
    return dirs[0] if dirs else Path(os.environ.get("APPDATA", str(Path.home()))) / ".minecraft" / "saves"


def discover_saves_dirs() -> list[Path]:
    """Find vanilla + launcher-isolated saves (version folders, custom gameDir)."""
    found: list[Path] = []

    def add(p: Path) -> None:
        try:
            p = p.resolve()
        except OSError:
            p = p
        if p.is_dir() and p not in found:
            found.append(p)

    appdata = os.environ.get("APPDATA", "")
    if appdata:
        add(Path(appdata) / ".minecraft" / "saves")
    home = Path.home()
    add(home / "AppData" / "Roaming" / ".minecraft" / "saves")
    add(home / ".minecraft" / "saves")

    for game in _minecraft_roots_from_process() + _minecraft_roots_from_disk():
        add(game / "saves")
        versions = game / "versions"
        if versions.is_dir():
            for ver in versions.iterdir():
                add(ver / "saves")

    # newest player files first
    def newest(d: Path) -> float:
        best = 0.0
        try:
            best = d.stat().st_mtime
            for world in d.iterdir():
                for dat in _player_dat_files(world):
                    best = max(best, dat.stat().st_mtime)
        except OSError:
            pass
        return best

    found.sort(key=newest, reverse=True)
    return found


def _minecraft_roots_from_process() -> list[Path]:
    roots: list[Path] = []
    for cmd in _javaw_command_lines():
        m = re.search(r"--gameDir\s+\"([^\"]+)\"|--gameDir\s+(\S+)", cmd)
        if m:
            raw = m.group(1) or m.group(2)
            p = Path(raw)
            roots.append(p)
            if p.name.lower() != ".minecraft" and (p / ".minecraft").is_dir():
                roots.append(p / ".minecraft")
        for hit in re.findall(r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*\.minecraft", cmd):
            roots.append(Path(hit))
    out: list[Path] = []
    for r in roots:
        if r.is_dir() and r not in out:
            out.append(r)
    return out


def _minecraft_roots_from_disk() -> list[Path]:
    roots: list[Path] = []
    for base in (
        Path(r"D:\Minecraft"),
        Path(r"C:\Minecraft"),
        Path.home() / "Minecraft",
        Path(r"D:\Games"),
    ):
        if not base.is_dir():
            continue
        try:
            for p in base.rglob(".minecraft"):
                if p.is_dir():
                    roots.append(p)
                    if len(roots) >= 12:
                        return roots
        except OSError:
            continue
    return roots


def _javaw_command_lines() -> list[str]:
    try:
        r = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name = 'javaw.exe'\" | ForEach-Object CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            errors="replace",
        )
        return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def _player_dat_files(world: Path) -> list[Path]:
    files: list[Path] = []
    for folder in (world / "players" / "data", world / "playerdata"):
        if folder.is_dir():
            files.extend(folder.glob("*.dat"))
    return files


def probe_hud(frame: NDArray[np.uint8]) -> dict[str, float]:
    """Cheap vanilla-HUD guess from a RGB frame. Overridden by sidecar/NBT when present."""
    h, w = frame.shape[:2]
    out = {k: 0.0 for k in ("hp", "hunger", "xp_progress", "hotbar", "gui_open")}
    if h < 32 or w < 32:
        return out
    band = frame[int(h * 0.80) :, :, :].astype(np.int16)
    left = band[:, : max(1, w // 3)]
    right = band[:, 2 * w // 3 :]
    red = (left[:, :, 0] > 150) & (left[:, :, 1] < 70) & (left[:, :, 2] < 70)
    brown = (right[:, :, 0] > 90) & (right[:, :, 1] > 40) & (right[:, :, 1] < 130) & (right[:, :, 2] < 60)
    out["hp"] = float(np.clip(red.mean() * 22.0, 0.0, 1.0))
    out["hunger"] = float(np.clip(brown.mean() * 28.0, 0.0, 1.0))
    xp = frame[int(h * 0.76) : int(h * 0.82), int(w * 0.22) : int(w * 0.78)]
    if xp.size:
        green = (xp[:, :, 1] > 140) & (xp[:, :, 0] < 180) & (xp[:, :, 2] < 80)
        out["xp_progress"] = float(np.clip(green.mean() * 8.0, 0.0, 1.0))
    hot = frame[int(h * 0.88) :, int(w * 0.20) : int(w * 0.80)]
    if hot.size:
        cols = np.array_split(hot, 9, axis=1)
        bright = [float(c.mean()) for c in cols if c.size]
        if bright:
            out["hotbar"] = float(int(np.argmax(bright))) / 8.0
    center = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.size:
        # inventory GUI is a relatively flat gray-brown sheet
        std = float(center.std())
        mean = float(center.mean())
        out["gui_open"] = 1.0 if std < 28.0 and 40.0 < mean < 110.0 else 0.0
    return out


def fields_to_vector(src: dict[str, Any], dim: int = 16) -> NDArray[np.float32]:
    hp = float(src.get("hp", 0.0))
    max_hp = float(src.get("max_hp", 20.0) or 20.0)
    hunger = float(src.get("hunger", 0.0))
    air = float(src.get("air", 300.0))
    yaw = float(src.get("yaw", 0.0))
    pitch = float(src.get("pitch", 0.0))
    yaw_rad = math.radians(yaw)
    vx = float(src.get("vx", 0.0))
    vz = float(src.get("vz", 0.0))
    gui_open = bool(src.get("gui_open"))
    held = float(src.get("selected_count", 0.0))
    if gui_open and src.get("carried_count") is not None:
        held = float(src.get("carried_count") or 0.0)
    cx = float(src.get("mouse_nx", src.get("cursor_x", 0.5)))
    cy = float(src.get("mouse_ny", src.get("cursor_y", 0.5)))
    hovered = float(src.get("hovered_slot", -1.0))
    vec = np.zeros(dim, dtype=np.float32)
    packed = [
        np.clip(hp / max(1.0, max_hp), 0.0, 1.5),
        np.clip(hunger / 20.0, 0.0, 1.5),
        np.clip(air / 300.0, 0.0, 1.5),
        float(np.clip(src.get("xp_progress", 0.0), 0.0, 1.0)),
        math.sin(yaw_rad),
        math.cos(yaw_rad),
        float(np.clip(pitch / 90.0, -1.0, 1.0)),
        float(np.clip(float(src.get("hotbar_slot", src.get("hotbar", 0.0))) / 8.0, 0.0, 1.0)),
        1.0 if src.get("on_ground") else 0.0,
        1.0 if src.get("sneaking") else 0.0,
        1.0 if src.get("sprinting") else 0.0,
        1.0 if src.get("in_water") else 0.0,
        float(np.clip(math.sqrt(vx * vx + vz * vz) / 4.35, 0.0, 2.0)),
        float(np.clip(float(src.get("y", 64.0)) / 256.0, 0.0, 2.0)),
        float(np.clip(held / 64.0, 0.0, 1.0)),
        1.0 if gui_open else 0.0,
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip((hovered + 1.0) / 90.0, 0.0, 1.0)),
    ]
    n = min(dim, len(packed))
    vec[:n] = np.asarray(packed[:n], dtype=np.float32)
    return vec


def _sidecar_has_gui(merged: dict[str, Any]) -> bool:
    """True only when sidecar/NBT actually reported an open GUI."""
    return "gui_open" in merged and bool(merged.get("gui_open"))


class MinecraftStateHub:
    def __init__(
        self,
        *,
        saves_dir: str | Path | None = None,
        sidecar_path: str | Path | None = None,
        udp_host: str = "127.0.0.1",
        udp_port: int = 21337,
        poll_saves_hz: float = 2.0,
        hud_probe_enabled: bool = True,
        state_dim: int = 16,
    ):
        self.saves_dir = Path(saves_dir) if saves_dir else default_saves_dir()
        self.saves_dirs = [self.saves_dir] if saves_dir else discover_saves_dirs() or [self.saves_dir]
        self.sidecar_path = Path(sidecar_path) if sidecar_path else None
        self.poll_dt = 1.0 / max(0.2, poll_saves_hz)
        self.hud_probe_enabled = hud_probe_enabled
        self.state_dim = state_dim
        self._lock = threading.Lock()
        self._nbt: dict[str, Any] = {}
        self._sidecar: dict[str, Any] = {}
        self._sidecar_t = 0.0
        self._alive = True
        self._sock: socket.socket | None = None
        self.last_raw: dict[str, Any] = {}
        self.last_world: str = ""
        self._nbt_thread = threading.Thread(target=self._poll_saves, daemon=True, name="mc-nbt")
        self._nbt_thread.start()
        self._udp_port = int(udp_port)
        if self._udp_port > 0:
            self._udp_thread = threading.Thread(
                target=self._udp_loop, args=(udp_host, self._udp_port), daemon=True, name="mc-udp"
            )
            self._udp_thread.start()
        else:
            self._udp_thread = None

    def snapshot(
        self,
        frame: NDArray[np.uint8] | None,
        keys_down: dict[str, bool] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> NDArray[np.float32]:
        merged: dict[str, Any] = {}
        if self.hud_probe_enabled and frame is not None:
            hud = probe_hud(frame)
            hud.pop("gui_open", None)
            merged.update(hud)
            if "hotbar" in hud:
                merged["hotbar_slot"] = hud["hotbar"] * 8.0
        with self._lock:
            merged.update(self._nbt)
            if time.time() - self._sidecar_t < 1.0:
                merged.update(self._sidecar)
        self._read_sidecar_file(merged)
        if extra:
            for k, v in extra.items():
                merged.setdefault(k, v)
        kd = keys_down or {}
        merged.setdefault("sneaking", bool(kd.get("SHIFT")))
        merged.setdefault("sprinting", bool(kd.get("CTRL")))
        # HUD gray-center is a guess and must not switch look encoding to GUI cursor.
        if not _sidecar_has_gui(merged):
            merged["gui_open"] = False
        self.last_raw = merged
        return fields_to_vector(merged, dim=self.state_dim)

    def last_facts(self) -> NDArray[np.float32]:
        """Raw-ish numbers for HDF5 (x y z yaw pitch hp hunger hotbar)."""
        m = self.last_raw
        hot = m.get("hotbar_slot", m.get("hotbar", 0.0))
        return np.asarray(
            [
                float(m.get("x", 0.0)),
                float(m.get("y", 64.0)),
                float(m.get("z", 0.0)),
                float(m.get("yaw", 0.0)),
                float(m.get("pitch", 0.0)),
                float(m.get("hp", 0.0)),
                float(m.get("hunger", 0.0)),
                float(hot),
            ],
            dtype=np.float32,
        )

    def source_name(self) -> str:
        with self._lock:
            if time.time() - self._sidecar_t < 1.0:
                return "sidecar"
            if self._nbt:
                world = str(self._nbt.get("_world") or self.last_world or "nbt")
                return f"nbt:{world}"
        if self.sidecar_path is not None and self.sidecar_path.is_file():
            return "file"
        return "hud"

    def latest_debug(self) -> dict[str, Any]:
        with self._lock:
            return {"nbt": dict(self._nbt), "sidecar_age_s": time.time() - self._sidecar_t}

    def close(self) -> None:
        self._alive = False
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
            self._sock = None

    def _read_sidecar_file(self, merged: dict[str, Any]) -> None:
        if self.sidecar_path is None or not self.sidecar_path.is_file():
            return
        try:
            age = time.time() - self.sidecar_path.stat().st_mtime
            if age > 2.0:
                return
            data = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
        except Exception:
            return

    def _poll_saves(self) -> None:
        while self._alive:
            try:
                fields = _read_latest_player_any(self.saves_dirs)
                if fields:
                    with self._lock:
                        self._nbt = fields
                        self.last_world = str(fields.get("_world", ""))
            except Exception:
                pass
            time.sleep(self.poll_dt)

    def _udp_loop(self, host: str, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        except OSError:
            sock.close()
            return
        sock.settimeout(0.25)
        self._sock = sock
        while self._alive:
            try:
                payload, addr = sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                break
            if addr[0] not in {"127.0.0.1", "::1"}:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                with self._lock:
                    self._sidecar = data
                    self._sidecar_t = time.time()
        try:
            sock.close()
        except OSError:
            pass
        if self._sock is sock:
            self._sock = None


def _read_latest_player_any(saves_dirs: list[Path]) -> dict[str, Any] | None:
    best: tuple[float, Path, Path] | None = None
    for saves_dir in saves_dirs:
        if not saves_dir.is_dir():
            continue
        try:
            worlds = [p for p in saves_dir.iterdir() if p.is_dir()]
        except OSError:
            continue
        for world in worlds:
            preferred = _preferred_player_file(world)
            files = _player_dat_files(world)
            if not files:
                level = world / "level.dat"
                if level.is_file():
                    files = [level]
            for path in files:
                try:
                    mt = path.stat().st_mtime
                except OSError:
                    continue
                score = mt + (0.5 if preferred is not None and path == preferred else 0.0)
                if best is None or score > best[0]:
                    best = (score, world, path)
    if best is None:
        return None
    _score, world, path = best
    try:
        root = load_nbt(path)
        player = extract_player(root)
        if player is None and isinstance(root, dict) and ("Health" in root or "Pos" in root):
            player = root
        if not player:
            return None
        fields = player_to_fields(player)
        fields["_world"] = world.name
        fields["_path"] = str(path)
        return fields
    except Exception:
        return None


def _preferred_player_file(world: Path) -> Path | None:
    level = world / "level.dat"
    if not level.is_file():
        return None
    try:
        uid = singleplayer_uuid(load_nbt(level))
    except Exception:
        return None
    if not uid:
        return None
    names = [uid, uid.replace("-", "")]
    for folder in (world / "players" / "data", world / "playerdata"):
        for name in names:
            p = folder / f"{name}.dat"
            if p.is_file():
                return p
    return None


def _read_latest_player(saves_dir: Path) -> dict[str, Any] | None:
    return _read_latest_player_any([saves_dir])
