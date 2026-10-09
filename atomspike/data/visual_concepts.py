"""Auditable, coarse labels for the block/entity under the crosshair.

These labels come from a local Minecraft sidecar during *collection only*.
The model never receives them as an inference input. -1 means unobserved;
an absent/stale sidecar must never be silently relabelled as open air.
"""

from __future__ import annotations

CONCEPT_KINDS: tuple[str, ...] = (
    "air", "log", "stone", "ore", "soil", "plant", "water", "lava",
    "wood", "other_block", "entity",
)
UNOBSERVED = -1


def concept_from_signal(signal: dict | None, *, sidecar_fresh: bool = True) -> int:
    """Map a fresh ray hit to a coarse class, retaining unknown as -1."""
    if not sidecar_fresh or not isinstance(signal, dict) or any(
        bool(signal.get(flag)) for flag in ("gui_open", "paused", "inv_open")
    ):
        return UNOBSERVED
    entity = signal.get("look_entity_id")
    if isinstance(entity, str) and entity:
        return CONCEPT_KINDS.index("entity")
    block = signal.get("look_block_id")
    if isinstance(block, str) and block:
        name = block.split(":", 1)[-1].lower()
        if name in {"water", "bubble_column"}:
            kind = "water"
        elif name == "lava":
            kind = "lava"
        elif name.endswith(("_log", "_stem", "_hyphae")) or name in {"bamboo_block", "stripped_bamboo_block"}:
            kind = "log"
        elif name.endswith("_ore") or name in {"ancient_debris", "nether_quartz_ore"}:
            kind = "ore"
        elif name.endswith(("_planks", "_wood", "_slab", "_stairs", "_fence", "_door")):
            kind = "wood"
        elif name in {"stone", "cobblestone", "deepslate", "cobbled_deepslate", "granite", "diorite", "andesite", "tuff", "netherrack", "basalt", "bedrock", "obsidian"} or "stone" in name:
            kind = "stone"
        elif name in {"dirt", "coarse_dirt", "grass_block", "podzol", "mycelium", "sand", "red_sand", "gravel", "clay", "mud", "snow_block"}:
            kind = "soil"
        elif name.endswith(("_leaves", "_sapling", "_flower", "_grass", "_crop")) or name in {"short_grass", "tall_grass", "vine", "wheat", "sugar_cane", "cactus", "moss_block"}:
            kind = "plant"
        else:
            kind = "other_block"
        return CONCEPT_KINDS.index(kind)
    if signal.get("look_miss") is True:
        return CONCEPT_KINDS.index("air")
    return UNOBSERVED
