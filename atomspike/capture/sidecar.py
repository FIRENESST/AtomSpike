"""Localhost sidecar protocol for Minecraft live state.

The collector listens on UDP 127.0.0.1:21337 (JSON object, IPv4 localhost only).
Packets older than 1s are ignored. This is not a cheat client: it only publishes
the local player's own stats for research capture.
"""

from __future__ import annotations

import json
import socket
from typing import Any

SIDECAR_HOST = "127.0.0.1"
SIDECAR_PORT = 21337

SIDECAR_FIELDS = (
    "hp",
    "max_hp",
    "hunger",
    "air",
    "xp_progress",
    "x",
    "y",
    "z",
    "yaw",
    "pitch",
    "vx",
    "vy",
    "vz",
    "hotbar_slot",
    "selected_count",
    "on_ground",
    "sneaking",
    "sprinting",
    "in_water",
    "gui_open",
    "mouse_nx",
    "mouse_ny",
    "hovered_slot",
    "carried_count",
    "mouse_grabbed",
    "gui",
)


def sidecar_payload(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k in SIDECAR_FIELDS or k in {"hp", "x", "y", "z"}}


def send_sidecar(payload: dict[str, Any], host: str = SIDECAR_HOST, port: int = SIDECAR_PORT) -> int:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(data, (host, int(port)))
    finally:
        sock.close()
    return len(data)
