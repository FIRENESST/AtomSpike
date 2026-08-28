"""Dataclass configs with optional YAML overlay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


ActionDecode = Literal["parallel", "gru_ar", "transformer_ar"]
PolicyKind = Literal["ann", "spike"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass
class CaptureConfig:
    fps: int = 30
    monitor: int = 1
    region: tuple[int, int, int, int] | None = None
    jpeg_quality: int = 90


@dataclass
class EncoderConfig:
    in_channels: int = 3
    stem_channels: tuple[int, int, int] = (16, 32, 64)
    d_model: int = 128
    frame_size: int = 84


@dataclass
class ReasonerConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    game_state_dim: int = 8
    n_state_tokens: int = 4


@dataclass
class TemporalAdapterConfig:
    """Cheap 30Hz residual so dual-rate perception does not starve the policy."""

    enabled: bool = True
    hidden: int = 32
    d_model: int = 128


@dataclass
class PolicyConfig:
    kind: PolicyKind = "ann"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    action_decode: ActionDecode = "gru_ar"
    temperature: float = 1.0
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    spike_time_steps: int = 4


@dataclass
class ActionConfig:
    n_keys: int = 4
    key_names: tuple[str, ...] = ("W", "A", "S", "D")
    n_key_states: int = 4  # idle / press / hold / release
    mouse_bins: int = 21  # quantized dx/dy in [-10, 10]
    n_mouse_buttons: int = 2
    n_slots: int = 8


@dataclass
class DualRateConfig:
    perception_hz: float = 5.0
    policy_hz: float = 30.0

    @property
    def policy_period_s(self) -> float:
        return 1.0 / self.policy_hz

    @property
    def perception_every_n(self) -> int:
        return max(1, int(round(self.policy_hz / self.perception_hz)))


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    num_workers: int = 0
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    precision: Precision = "fp32"
    kl_coef: float = 0.1
    advantage_temp: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    log_wandb: bool = False


@dataclass
class EnvConfig:
    kind: str = "synthetic"
    max_steps: int = 128
    frame_size: int = 84
    seed: int = 0
    vizdoom_scenario: str = "basic"


@dataclass
class ConvertConfig:
    method: Literal["pmsm", "spiked_attention"] = "pmsm"
    time_steps: int = 1
    threshold_percentile: float = 99.0


@dataclass
class AtomSpikeConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    adapter: TemporalAdapterConfig = field(default_factory=TemporalAdapterConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    dual_rate: DualRateConfig = field(default_factory=DualRateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    convert: ConvertConfig = field(default_factory=ConvertConfig)

    def aligned(self) -> "AtomSpikeConfig":
        """Keep submodule widths consistent after YAML overlays."""
        d = self.encoder.d_model
        self.reasoner.d_model = d
        self.adapter.d_model = d
        self.policy.d_model = d
        self.env.frame_size = self.encoder.frame_size
        return self


def _merge(dc: Any, data: dict[str, Any]) -> None:
    for f in fields(dc):
        if f.name not in data:
            continue
        cur = getattr(dc, f.name)
        val = data[f.name]
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val)
        else:
            setattr(dc, f.name, val)


def load_config(path: str | Path | None = None) -> AtomSpikeConfig:
    cfg = AtomSpikeConfig()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config {path} must be a mapping")
        _merge(cfg, raw)
    return cfg.aligned()


def dump_config(cfg: AtomSpikeConfig) -> dict[str, Any]:
    return asdict(cfg)
