"""Dataclass configs with optional YAML overlay."""

from __future__ import annotations

import ipaddress
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

ActionDecode = Literal["parallel", "gru_ar", "transformer_ar"]
PolicyKind = Literal["ann", "spike"]
Precision = Literal["fp32", "fp16", "bf16"]


class ConfigError(ValueError):
    """A user-facing configuration error with the failing field path."""


@dataclass
class CaptureConfig:
    fps: int = 30
    monitor: int = 1
    region: tuple[int, int, int, int] | None = None
    jpeg_quality: int = 90
    window_title: str = "Minecraft"
    store_width: int = 854
    store_height: int = 480
    client_only: bool = True


@dataclass
class MinecraftConfig:
    """Human 30fps collector for local / research Minecraft sessions."""

    window_title: str = "Minecraft"
    sidecar_host: str = "127.0.0.1"
    sidecar_udp_port: int = 21337
    sidecar_cmd_port: int = 21338
    sidecar_path: str = ""
    saves_dir: str = ""
    poll_saves_hz: float = 2.0
    hud_probe: bool = True
    record_input: bool = True
    stop_key: str = "F9"
    mark_key: str = "F8"
    state_dim: int = 19


@dataclass
class EncoderConfig:
    in_channels: int = 3
    stem_channels: tuple[int, int, int] = (16, 32, 64)
    d_model: int = 128
    frame_size: int = 84
    max_token_h: int = 12
    max_token_w: int = 20
    n_blocks_per_stage: int = 2


@dataclass
class ReasonerConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    drop_path: float = 0.1
    token_drop: float = 0.12
    game_state_dim: int = 19
    n_state_tokens: int = 4


@dataclass
class TemporalAdapterConfig:
    """Cheap 30Hz residual so dual-rate perception does not starve the policy."""

    enabled: bool = True
    hidden: int = 32
    d_model: int = 128
    down_h: int = 96
    down_w: int = 160


@dataclass
class IteratorConfig:
    """10Hz spatial-temporal-task iterator. Bias is zero until it learns."""

    enabled: bool = True
    hidden: int = 64
    spatial_hw: int = 4
    spatial_h: int = 4
    spatial_w: int = 6
    spatial_dim: int = 16
    task_dim: int = 32
    n_task_buckets: int = 16
    change_aux: float = 0.04
    # Live 10Hz ticks keep the GRU. Training rows stay one step from rest.
    persistent_inference: bool = True


@dataclass
class SceneConfig:
    """10Hz curl of the spatial LIF field as a scene fingerprint. Bias starts at zero."""

    enabled: bool = True
    spatial_hw: int = 4
    spatial_h: int = 4
    spatial_w: int = 6
    spatial_dim: int = 16
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class GroupConfig:
    """Activity coalitions + 10Hz spike group attention + specialist birth."""

    enabled: bool = True
    n_groups: int = 8
    n_heads: int = 2
    specialist_hidden: int = 8
    ema: float = 0.9
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    birth_energy: float = 0.25
    persistent_inference: bool = True


@dataclass
class ScreenEventConfig:
    """SECA: top-K screen spike events + coincidence self-attention."""

    enabled: bool = True
    k_events: int = 12
    max_spatial: int = 64
    n_heads: int = 2
    onset: float = 1.0
    # Innate look-center tag when a block breaks / is placed. Fraction of
    # local RMS so the pixels still carry which-block identity.
    block_amp: float = 0.08
    block_onset: float = 0.2
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class DetectConfig:
    """YOLO-style 10Hz spike presence grid. Residual starts at zero."""

    enabled: bool = True
    grid_h: int = 6
    grid_w: int = 10
    n_classes: int = 8
    n_keep: int = 8
    hidden: int = 32
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class EventSeqConfig:
    """Global event tape + planner slots; order head learns temporal ranking."""

    enabled: bool = True
    n_slots: int = 8
    n_events: int = 24
    n_heads: int = 2
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class InventoryConfig:
    """Item-id LIF residual for inventory / backpack. Bias starts at zero."""

    enabled: bool = True
    n_slots: int = 63
    item_dim: int = 16
    hidden: int = 32
    n_buckets: int = 48
    lif_steps: int = 2
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class ThoughtConfig:
    """Internal spike deliberation at 10Hz. Readout starts at zero."""

    enabled: bool = True
    n_cells: int = 8
    steps: int = 3
    hidden: int = 16
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    persistent_inference: bool = True


@dataclass
class PolicyConfig:
    kind: PolicyKind = "spike"
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    drop_path: float = 0.08
    action_decode: ActionDecode = "gru_ar"
    temperature: float = 1.0
    lif_tau: float = 0.5
    lif_v_th: float = 1.0
    # One LIF step per tick. Live inference keeps the membrane; training rows
    # still start from rest so the learned weights are unchanged.
    spike_time_steps: int = 1
    persistent_inference: bool = True


@dataclass
class PlannerConfig:
    """Invented task slots, inferred importance, self-arranged successor."""

    enabled: bool = True
    latent_dim: int = 64
    hidden: int = 128
    n_phases: int = 16
    n_plan_tokens: int = 2
    future_steps: int = 15
    future_near: int = 5
    future_far: int = 45
    session_gap_s: float = 2.0
    kl_coef: float = 0.01
    phase_coef: float = 0.0
    next_coef: float = 0.4
    future_coef: float = 0.2
    time_coef: float = 0.12
    mono_coef: float = 0.0
    div_coef: float = 0.05
    arrange: float = 0.7
    creative: bool = True
    explore: float = 0.08
    event_pool: int = 8


@dataclass
class ActionConfig:
    n_keys: int = 8
    key_names: tuple[str, ...] = ("W", "A", "S", "D", "SPACE", "SHIFT", "CTRL", "E")
    n_key_states: int = 4  # idle / press / hold / release
    mouse_bins: int = 41  # look / GUI cursor quantization
    n_mouse_buttons: int = 2
    n_slots: int = 8
    mouse_delta_max: float = 240.0
    mouse_delta_scale: float = 16.0
    wheel_bins: int = 11  # odd >= 3 enables a scroll slot; 1 = off
    wheel_max: float = 4.0


@dataclass
class SelfLearnConfig:
    """Opt-in online adaptation while ``predict-mc`` is running.

    Without input injection this is supervised human-demonstration
    adaptation.  With guarded singleplayer injection it becomes a small
    actor-critic update on sampled model actions.
    """

    # Mutating a deployed model remains explicit even though both modes are
    # implemented and covered by verification.
    enabled: bool = False
    lr: float = 1.0e-4
    kl_coef: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    reward_clip: float = 2.0
    credit_steps: int = 4
    credit_gamma: float = 0.9
    buffer: int = 32
    budget_ms: float = 5.0
    max_new_events: int = 4
    max_new_layers: int = 2
    layer_hidden: int = 16
    birth_surprise: float = 0.45
    birth_cooldown_s: float = 8.0
    save_every_s: float = 20.0
    goal_reward: float = 1.0
    hp_penalty: float = 0.2
    sky_penalty: float = 0.06
    sky_grace_s: float = 0.8
    sky_pitch: float = -32.0
    ground_penalty: float = 0.06
    ground_grace_s: float = 5.0
    ground_pitch: float = 40.0
    look_bonus: float = 0.02
    look_bins: int = 8
    look_memory_s: float = 6.0
    stall_s: float = 600.0
    stall_penalty: float = 2.0
    inv_penalty: float = 0.04
    pause_penalty: float = 0.3
    view_bonus: float = 0.05
    jump_penalty: float = 0.05
    circle_penalty: float = 0.06
    circle_s: float = 3.5
    idle_penalty: float = 0.08
    idle_s: float = 30.0
    idle_radius: float = 1.75


ClockMode = Literal["sim", "realtime"]


@dataclass
class DualRateConfig:
    perception_hz: float = 10.0
    policy_hz: float = 30.0
    mode: ClockMode = "sim"
    pace: bool = False

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
    epochs: int = 32
    num_workers: int = 0
    grad_clip: float = 1.0
    grad_accum: int = 1  # micro-batches to accumulate before optimizer.step
    seed: int = 0
    device: str = "auto"
    precision: Precision = "bf16"
    compile: bool = False  # torch.compile the agent before training
    compile_mode: str = "default"
    # WSD schedule (fractions of total optimizer steps)
    lr_schedule: Literal["constant", "wsd"] = "wsd"
    wsd_warmup_frac: float = 0.05
    wsd_decay_frac: float = 0.10
    wsd_decay: Literal["linear", "cosine"] = "linear"
    # Optimizer
    optimizer: Literal["adamw", "muon", "muon_adamw"] = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    # Checkpointing
    ckpt_keep_k: int = 3
    ckpt_verify: bool = True
    kl_coef: float = 0.1
    advantage_temp: float = 1.0
    rl_gamma: float = 0.99
    lora_r: int = 8
    lora_alpha: int = 16
    log_wandb: bool = False
    val_frac: float = 0.1
    gui_sample_weight: float = 5.0
    motion_sample_weight: float = 5.0
    look_move_ce_boost: float = 4.0
    gui_mouse_ce_boost: float = 5.0
    slot_balance: bool = False
    label_smoothing: float = 0.05
    semantic_loss_weight: float = 0.3  # active only for labelled crosshair rays
    effect_loss_weight: float = 0.2  # observational outcomes; never a causal claim
    aug_brightness: float = 0.18
    aug_contrast: float = 0.18
    aug_saturation: float = 0.12
    aug_noise: float = 0.02
    aug_erase: float = 0.12
    aug_erase_scale: float = 0.08
    # Keep full-resolution HDF5 frames out of RAM once the raw frame tensor is
    # larger than this threshold; labels stay resident for sampling and splits.
    max_resident_frames_mib: int = 1024


@dataclass
class EnvConfig:
    kind: str = "minecraft"
    max_steps: int = 128
    frame_size: int = 84
    seed: int = 0
    vizdoom_scenario: str = "basic"


@dataclass
class ConvertConfig:
    method: Literal["pmsm", "spiked_attention"] = "pmsm"
    time_steps: int = 1
    threshold_percentile: float = 99.0
    calibration_samples: int = 256


@dataclass
class AtomSpikeConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    adapter: TemporalAdapterConfig = field(default_factory=TemporalAdapterConfig)
    iterator: IteratorConfig = field(default_factory=IteratorConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    sea: ScreenEventConfig = field(default_factory=ScreenEventConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    event_seq: EventSeqConfig = field(default_factory=EventSeqConfig)
    thought: ThoughtConfig = field(default_factory=ThoughtConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    dual_rate: DualRateConfig = field(default_factory=DualRateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    minecraft: MinecraftConfig = field(default_factory=MinecraftConfig)
    convert: ConvertConfig = field(default_factory=ConvertConfig)
    learn: SelfLearnConfig = field(default_factory=SelfLearnConfig)

    def aligned(self) -> AtomSpikeConfig:
        """Keep submodule widths consistent after YAML overlays."""
        self.encoder.stem_channels = tuple(int(v) for v in self.encoder.stem_channels)
        self.action.key_names = tuple(str(v).upper() for v in self.action.key_names)
        d = self.encoder.d_model
        self.reasoner.d_model = d
        self.adapter.d_model = d
        self.policy.d_model = d
        self.env.frame_size = self.encoder.frame_size
        mc_kinds = {"minecraft", "mc", "minerl"}
        if self.env.kind.lower() in mc_kinds:
            self.reasoner.game_state_dim = max(self.reasoner.game_state_dim, self.minecraft.state_dim)
            self.capture.window_title = self.minecraft.window_title or self.capture.window_title
            if int(self.action.wheel_bins) < 3:
                self.action.wheel_bins = 11
        wheel_slot = 1 if int(self.action.wheel_bins) >= 3 else 0
        self.action.n_slots = self.action.n_keys + 2 + self.action.n_mouse_buttons + wheel_slot
        if self.policy.n_heads < 1 or d % self.policy.n_heads != 0:
            for h in (8, 4, 2, 1):
                if d % h == 0:
                    self.policy.n_heads = h
                    break
        if self.reasoner.n_heads < 1 or d % self.reasoner.n_heads != 0:
            self.reasoner.n_heads = self.policy.n_heads
        return self

    def validate(self) -> AtomSpikeConfig:
        """Reject invalid settings early, before model or capture startup."""

        def positive(path: str, value: int | float) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"{path} must be positive, got {value!r}")

        def probability(path: str, value: float, *, inclusive_one: bool = True) -> None:
            number = float(value)
            upper_ok = number <= 1.0 if inclusive_one else number < 1.0
            if number < 0.0 or not upper_ok:
                bound = "[0, 1]" if inclusive_one else "[0, 1)"
                raise ConfigError(f"{path} must be in {bound}, got {value!r}")

        for path, value in (
            ("capture.fps", self.capture.fps),
            ("capture.store_width", self.capture.store_width),
            ("capture.store_height", self.capture.store_height),
            ("minecraft.poll_saves_hz", self.minecraft.poll_saves_hz),
            ("minecraft.state_dim", self.minecraft.state_dim),
            ("encoder.frame_size", self.encoder.frame_size),
            ("encoder.d_model", self.encoder.d_model),
            ("encoder.n_blocks_per_stage", self.encoder.n_blocks_per_stage),
            ("reasoner.n_layers", self.reasoner.n_layers),
            ("reasoner.n_heads", self.reasoner.n_heads),
            ("reasoner.game_state_dim", self.reasoner.game_state_dim),
            ("policy.n_layers", self.policy.n_layers),
            ("policy.n_heads", self.policy.n_heads),
            ("policy.temperature", self.policy.temperature),
            ("policy.spike_time_steps", self.policy.spike_time_steps),
            ("action.n_keys", self.action.n_keys),
            ("action.n_key_states", self.action.n_key_states),
            ("action.mouse_bins", self.action.mouse_bins),
            ("action.mouse_delta_max", self.action.mouse_delta_max),
            ("action.mouse_delta_scale", self.action.mouse_delta_scale),
            ("dual_rate.perception_hz", self.dual_rate.perception_hz),
            ("dual_rate.policy_hz", self.dual_rate.policy_hz),
            ("train.batch_size", self.train.batch_size),
            ("train.lr", self.train.lr),
            ("train.epochs", self.train.epochs),
            ("train.grad_accum", self.train.grad_accum),
            ("env.max_steps", self.env.max_steps),
            ("env.frame_size", self.env.frame_size),
            ("convert.time_steps", self.convert.time_steps),
            ("convert.calibration_samples", self.convert.calibration_samples),
            ("train.max_resident_frames_mib", self.train.max_resident_frames_mib),
            ("learn.lr", self.learn.lr),
            ("learn.buffer", self.learn.buffer),
            ("learn.reward_clip", self.learn.reward_clip),
            ("learn.credit_steps", self.learn.credit_steps),
        ):
            positive(path, value)

        if self.capture.region is not None and (
            len(self.capture.region) != 4 or self.capture.region[2] <= 0 or self.capture.region[3] <= 0
        ):
            raise ConfigError("capture.region must be (left, top, width, height) with positive width/height")
        for path, value in (
            ("minecraft.sidecar_udp_port", self.minecraft.sidecar_udp_port),
            ("minecraft.sidecar_cmd_port", self.minecraft.sidecar_cmd_port),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
                raise ConfigError(f"{path} must be an integer in [1, 65535], got {value!r}")
        try:
            if not ipaddress.ip_address(self.minecraft.sidecar_host).is_loopback:
                raise ConfigError("minecraft.sidecar_host must be a loopback address")
        except ValueError as exc:
            raise ConfigError("minecraft.sidecar_host must be a loopback IP address") from exc

        if len(self.encoder.stem_channels) != 3 or any(int(v) <= 0 for v in self.encoder.stem_channels):
            raise ConfigError("encoder.stem_channels must contain exactly three positive integers")
        if self.encoder.d_model % self.reasoner.n_heads:
            raise ConfigError("encoder.d_model must be divisible by reasoner.n_heads")
        if self.encoder.d_model % self.policy.n_heads:
            raise ConfigError("encoder.d_model must be divisible by policy.n_heads")
        if len(self.action.key_names) != self.action.n_keys:
            raise ConfigError("action.key_names length must equal action.n_keys")
        if len(set(self.action.key_names)) != len(self.action.key_names):
            raise ConfigError("action.key_names must be unique")
        if self.action.mouse_bins < 3 or self.action.mouse_bins % 2 == 0:
            raise ConfigError("action.mouse_bins must be an odd integer >= 3")
        if self.action.wheel_bins != 1 and (self.action.wheel_bins < 3 or self.action.wheel_bins % 2 == 0):
            raise ConfigError("action.wheel_bins must be 1 (disabled) or an odd integer >= 3")
        if self.policy.spike_time_steps != 1:
            raise ConfigError(
                "policy.spike_time_steps must be 1: multi-step training does not match the deployed 30Hz policy"
            )
        if not math.isfinite(float(self.train.semantic_loss_weight)) or self.train.semantic_loss_weight < 0:
            raise ConfigError("train.semantic_loss_weight must be finite and non-negative")
        if not math.isfinite(float(self.train.effect_loss_weight)) or self.train.effect_loss_weight < 0:
            raise ConfigError("train.effect_loss_weight must be finite and non-negative")
        if self.dual_rate.perception_hz > self.dual_rate.policy_hz:
            raise ConfigError("dual_rate.perception_hz cannot exceed dual_rate.policy_hz")
        for path, value, inclusive in (
            ("reasoner.dropout", self.reasoner.dropout, False),
            ("reasoner.drop_path", self.reasoner.drop_path, False),
            ("reasoner.token_drop", self.reasoner.token_drop, False),
            ("policy.dropout", self.policy.dropout, False),
            ("policy.drop_path", self.policy.drop_path, False),
            ("train.val_frac", self.train.val_frac, False),
            ("train.label_smoothing", self.train.label_smoothing, False),
            ("train.wsd_warmup_frac", self.train.wsd_warmup_frac, True),
            ("train.wsd_decay_frac", self.train.wsd_decay_frac, True),
            ("learn.credit_gamma", self.learn.credit_gamma, True),
        ):
            probability(path, value, inclusive_one=inclusive)
        if self.learn.kl_coef < 0.0:
            raise ConfigError("learn.kl_coef must be non-negative")
        if self.learn.value_coef < 0.0:
            raise ConfigError("learn.value_coef must be non-negative")
        if self.learn.entropy_coef < 0.0:
            raise ConfigError("learn.entropy_coef must be non-negative")
        if self.train.wsd_warmup_frac + self.train.wsd_decay_frac > 1.0:
            raise ConfigError("train WSD warmup and decay fractions must sum to <= 1")
        if not 0.0 < self.train.rl_gamma <= 1.0:
            raise ConfigError("train.rl_gamma must be in (0, 1]")
        if not 0.0 <= self.convert.threshold_percentile < 100.0:
            raise ConfigError(
                "convert.threshold_percentile must be in [0, 100); 100 would calibrate every binary layer to never fire"
            )
        if self.convert.time_steps != 1:
            raise ConfigError("convert.time_steps must be 1: multi-step ANN→SNN conversion is not implemented")
        return self


def _merge(dc: Any, data: dict[str, Any], path: str = "") -> None:
    known = {f.name: f for f in fields(dc)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        prefix = f"{path}." if path else ""
        options = ", ".join(sorted(known))
        raise ConfigError(f"unknown config key {prefix}{unknown[0]}; expected one of: {options}")
    for f in fields(dc):
        if f.name not in data:
            continue
        cur = getattr(dc, f.name)
        val = data[f.name]
        field_path = f"{path}.{f.name}" if path else f.name
        if is_dataclass(cur) and isinstance(val, dict):
            _merge(cur, val, field_path)
        elif is_dataclass(cur):
            raise ConfigError(f"{field_path} must be a mapping")
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(dc, f.name, tuple(val))
        else:
            setattr(dc, f.name, val)


def load_config(path: str | Path | None = None) -> AtomSpikeConfig:
    cfg = AtomSpikeConfig()
    if path is not None:
        config_path = Path(path)
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "PyYAML is required to read YAML configs; install the atomspike package dependencies"
            ) from exc
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"config {config_path} must be a mapping")
        _merge(cfg, raw)
    return cfg.aligned().validate()


def dump_config(cfg: AtomSpikeConfig) -> dict[str, Any]:
    return asdict(cfg)
