from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from atomspike.config import AtomSpikeConfig, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
PACKAGED_CONFIGS = ROOT / "atomspike" / "_configs"


@pytest.mark.parametrize("name", ["default.yaml", "smoke.yaml", "minecraft.yaml"])
def test_source_and_packaged_profiles_are_equivalent(name: str) -> None:
    source = load_config(ROOT / "configs" / name)
    packaged = load_config(PACKAGED_CONFIGS / name)
    assert asdict(source) == asdict(packaged)


def test_default_profile_is_coherent() -> None:
    cfg = AtomSpikeConfig().aligned().validate()
    assert cfg.env.kind == "minecraft"
    assert cfg.action.n_slots == 13
    assert cfg.reasoner.game_state_dim == cfg.minecraft.state_dim
    assert isinstance(cfg.action.key_names, tuple)


def test_fp16_is_valid_with_dynamic_loss_scaling() -> None:
    cfg = AtomSpikeConfig().aligned()
    cfg.train.precision = "fp16"
    assert cfg.validate().train.precision == "fp16"


def test_yaml_lists_are_normalized_to_tuples(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "encoder:\n  stem_channels: [4, 8, 16]\naction:\n  key_names: [w, a, s, d, space, shift, ctrl, e]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.encoder.stem_channels == (4, 8, 16)
    assert cfg.action.key_names == ("W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E")


def test_unknown_key_reports_full_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learnng_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"train\.learnng_rate"):
        load_config(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda cfg: setattr(cfg.dual_rate, "perception_hz", 60.0), "cannot exceed"),
        (lambda cfg: setattr(cfg.action, "mouse_bins", 10), "odd integer"),
        (lambda cfg: setattr(cfg.minecraft, "sidecar_host", "0.0.0.0"), "loopback"),
        (lambda cfg: setattr(cfg.train, "val_frac", 1.0), r"\[0, 1\)"),
    ],
)
def test_invalid_config_fails_early(mutate, message: str) -> None:
    cfg = AtomSpikeConfig().aligned()
    mutate(cfg)
    with pytest.raises(ConfigError, match=message):
        cfg.validate()
