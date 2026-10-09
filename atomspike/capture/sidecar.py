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
    "hand_item",
    "hover_item",
    "carry_item",
    "inv",
    "menu",
    "mouse_grabbed",
    "gui",
    "adv_done",
    "adv_next_frac",
    "adv_next",
    "sky_light",
    "can_see_sky",
    "look_miss",
    "look_block_id",
    "look_entity_id",
    "dead",
    "paused",
    "inv_open",
    "blocks_seen",
    "broke",
    "placed",
    # This is emitted by the Fabric client from getSingleplayerServer().  It is
    # deliberately required by the OS-injection gate; absence is not evidence
    # that a world is safe to automate.
    "singleplayer",
    "session_nonce",
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


SIDECAR_CMD_PORT = 21338


def send_sidecar_cmd(
    cmd: str,
    host: str = SIDECAR_HOST,
    port: int = SIDECAR_CMD_PORT,
    *,
    nonce: str | None = None,
) -> int:
    """Send an authenticated-session command to the local sidecar."""
    payload = {"cmd": str(cmd)}
    if nonce is not None:
        payload["nonce"] = str(nonce)
    return send_sidecar(payload, host=host, port=port)
