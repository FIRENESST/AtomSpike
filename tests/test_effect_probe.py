from __future__ import annotations

import numpy as np
import torch

from atomspike.config import AtomSpikeConfig
from atomspike.capture.mc_state import MC_STATE_FIELDS
from atomspike.data.effects import EFFECT_ACTIONS, EFFECT_EVENTS, action_macros, observed_effect_targets
from atomspike.data.io import H5DemoDataset
from atomspike.data.stream import StreamingDemoWriter
from atomspike.eval.effect_metrics import EffectBrierMeter
from atomspike.models.actions import ActionSpec, KEY_HOLD
from atomspike.models.effect_probe import ConditionalEffectProbe, action_macro_indices
from atomspike.train.bc import _total_loss


def _spec() -> ActionSpec:
    return ActionSpec.from_config(AtomSpikeConfig().aligned().action)


def test_macro_encoding_matches_train_and_inference() -> None:
    spec = _spec()
    actions = np.tile(np.asarray(spec.idle_tokens(), dtype=np.int64), (5, 1))
    actions[0, spec.n_keys + 2] = KEY_HOLD
    actions[1, spec.n_keys + 3] = KEY_HOLD
    actions[2, spec.key_names.index("W")] = KEY_HOLD
    actions[3, spec.key_names.index("SPACE")] = KEY_HOLD
    actions[4, spec.n_keys] = spec.mouse_center + 1
    expected = ["attack", "use", "move", "jump", "look"]
    numpy_macro = action_macros(actions, spec)
    torch_macro = action_macro_indices(torch.from_numpy(actions), spec).numpy()
    assert np.array_equal(numpy_macro, torch_macro)
    assert [EFFECT_ACTIONS[i] for i in numpy_macro] == expected


def test_effect_windows_never_cross_episode_or_stale_teacher() -> None:
    spec = _spec()
    n = 8
    state = np.zeros((n, 19), dtype=np.float32)
    state[:, 0] = 1.0
    state[2, 0] = 0.8
    state[2, 14] = 1.0 / 64.0
    actions = np.tile(np.asarray(spec.idle_tokens(), dtype=np.int64), (n, 1))
    actions[:, spec.n_keys + 2] = KEY_HOLD
    aux = np.zeros((n, 8), dtype=np.float32)
    aux[2, 0] = 0.5
    task = np.zeros(n, dtype=np.int64)
    pulse = np.zeros((n, 2), dtype=np.float32)
    pulse[2, 0] = 1.0
    look = np.zeros(n, dtype=np.int64)
    sessions = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    y, valid = observed_effect_targets(state, actions, aux, task, pulse, look, sessions, spec, horizon=2)
    assert valid[0] == 1
    assert y[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert valid[2] == 0  # t+2 crosses the episode cut
    recording = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    _, cross_recording = observed_effect_targets(
        state, actions, aux, task, pulse, look, np.zeros(n, dtype=np.int32), spec,
        horizon=2, recording_id=recording,
    )
    assert cross_recording[0] == 0  # explicit recording boundary, even with continuous timestamps
    look[2] = -1
    _, stale = observed_effect_targets(state, actions, aux, task, pulse, look, sessions, spec, horizon=2)
    assert stale[0] == 0
    actions[1, spec.n_keys + 2] = 0
    _, switched = observed_effect_targets(state, actions, aux, task, pulse, np.zeros(n, dtype=np.int64), sessions, spec, horizon=2)
    assert switched[0] == 0  # only one of two ticks keeps the primary macro


def test_effect_probe_detaches_concept_and_respects_valid_mask() -> None:
    spec = _spec()
    cfg = AtomSpikeConfig()
    cfg.planner.enabled = False
    concept = torch.randn(2, 11, requires_grad=True)
    visual = torch.randn(2, 128, requires_grad=True)
    action = torch.tensor([spec.idle_tokens(), spec.idle_tokens()])
    probe = ConditionalEffectProbe(128)
    effect_logits = probe(concept, visual, action, spec)
    assert effect_logits.shape == (2, len(EFFECT_EVENTS))
    effect_logits.sum().backward(retain_graph=True)
    assert concept.grad is None
    assert visual.grad is None

    effect_logits.retain_grad()
    out = {
        "logits": [torch.zeros(2, 2, requires_grad=True)],
        "observed_effect_logits": effect_logits,
    }
    tokens = torch.zeros(2, 1, dtype=torch.long)
    common = (out, tokens, torch.ones(2, 1), torch.ones(1), None, cfg, {})
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    active = torch.tensor([True, False, False, False])
    baseline = _total_loss(*common, effect_target=target, effect_valid=torch.zeros(2, 1), effect_active=active)
    trained = _total_loss(*common, effect_target=target, effect_valid=torch.tensor([[1], [0]]), effect_active=active)
    assert trained > baseline


def test_action_blind_head_is_trained_but_excluded_from_selection_loss() -> None:
    cfg = AtomSpikeConfig()
    cfg.planner.enabled = False
    conditioned = torch.zeros(2, 4, requires_grad=True)
    visual = torch.zeros(2, 4, requires_grad=True)
    out = {
        "logits": [torch.zeros(2, 2, requires_grad=True)],
        "observed_effect_logits": conditioned,
        "visual_only_effect_logits": visual,
    }
    args = (out, torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1), torch.ones(1), None, cfg, {})
    kw = {
        "effect_target": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "effect_valid": torch.ones(2, 1),
        "effect_active": torch.tensor([True, False, False, False]),
    }
    _total_loss(*args, **kw).backward(retain_graph=True)
    assert conditioned.grad is not None
    assert visual.grad is None
    _total_loss(*args, **kw, train_visual_baseline=True).backward()
    assert visual.grad is not None


def test_effect_brier_reports_action_blind_and_event_support() -> None:
    meter = EffectBrierMeter(
        np.array([True, False, False, False]),
        np.array([0.5, 0.0, 0.0, 0.0]),
    )
    target = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    conditioned = np.array([[0.9, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]])
    visual = np.array([[0.5, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    meter.update(target, conditioned, visual)
    report = meter.report()
    assert report["n"] == 2
    assert np.isclose(report["conditioned"], 0.01)
    assert np.isclose(report["visual_only"], 0.25)
    assert np.isclose(report["prior"], 0.25)
    assert report["by_event"]["goal_progress"]["positive"] == 1
    assert report["by_event"]["goal_progress"]["negative"] == 1
    assert list(report["by_event"]) == ["goal_progress"]


def test_h5_dataset_exposes_bounded_effect_targets(tmp_path) -> None:
    spec = _spec()
    path = tmp_path / "effects.h5"
    writer = StreamingDemoWriter(
        path, (4, 5), 19, spec.n_slots,
        key_names=spec.key_names, state_fields=MC_STATE_FIELDS,
    )
    action = np.asarray(spec.idle_tokens(), dtype=np.int64)
    action[spec.n_keys + 2] = KEY_HOLD
    for i in range(8):
        state = np.zeros(19, dtype=np.float32)
        state[0] = 1.0 if i != 2 else 0.8
        aux = np.zeros(8, dtype=np.float32)
        aux[0] = 0.5 if i == 2 else 0.0
        writer.add(
            np.zeros((4, 5, 3), dtype=np.uint8), state, action, 0.0,
            i == 3, i * 33_000_000 + 1,
            task_aux=aux, look_concept=0,
        )
    writer.close()
    ds = H5DemoDataset(path, spec=spec)
    ds.attach_timeline(4, 2, future_far=2)
    assert ds[0]["effect_valid"].tolist() == [1]
    assert ds[0]["effect_target"][EFFECT_EVENTS.index("goal_progress")] == 1.0
    assert ds[2]["effect_valid"].tolist() == [0]
    ds.close()
