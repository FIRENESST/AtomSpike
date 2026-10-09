from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from torch import nn

from atomspike.capture.mc_predict import inject_gate, run_minecraft_predict
from atomspike.capture.mc_collector import run_minecraft_collector
from atomspike.capture.mc_state import fields_to_vector, sanitize_sidecar_payload
from atomspike.config import AtomSpikeConfig
from atomspike.convert.pmsm import enable_pmsm
from atomspike.data.io import H5DemoDataset
from atomspike.data.rabc import require_informative_rewards
from atomspike.data.sessions import recording_ids, stride_split_segments
from atomspike.data.stream import StreamingDemoWriter
from atomspike.models.activations import Act
from atomspike.train.common import grad_accum_divisor


def _tiny_h5(path, *, committed: np.ndarray | None = None) -> None:
    n = 3
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=np.arange(n * 4 * 5 * 3, dtype=np.uint8).reshape(n, 4, 5, 3))
        f.create_dataset("game_state", data=np.zeros((n, 2), dtype=np.float32))
        f.create_dataset("actions", data=np.zeros((n, 3), dtype=np.int64))
        f.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
        f.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.uint8))
        if committed is not None:
            f.create_dataset("committed", data=committed.astype(np.uint8))


def test_large_frame_storage_is_lazy_and_committed_tail_is_hidden(tmp_path) -> None:
    path = tmp_path / "demo.h5"
    _tiny_h5(path, committed=np.array([1, 1, 0], dtype=np.uint8))
    ds = H5DemoDataset(path, max_resident_frames_mib=0.00001)
    assert ds.frame_storage == "lazy"
    assert len(ds) == 2
    assert ds.frames.shape[0] == 2
    assert ds[1]["frame"].shape == (3, 4, 5)
    ds.close()


def test_session_split_is_recording_disjoint() -> None:
    sessions = np.array([0] * 4 + [1] * 5 + [2] * 6, dtype=np.int32)
    train, val = stride_split_segments(len(sessions), 0.2, sessions)
    assert train.size and val.size
    assert set(sessions[train]).isdisjoint(set(sessions[val]))
    assert set(sessions[val]) == {2}
    one_train, one_val = stride_split_segments(5, 0.2, np.zeros(5, dtype=np.int32))
    assert one_train.size == 5 and one_val.size == 0


def test_episode_marks_do_not_create_validation_recordings() -> None:
    n = 8
    times = np.arange(n, dtype=np.int64) * 33_000_000
    dones = np.zeros(n, dtype=np.uint8)
    dones[3] = 1
    ids = recording_ids(n, times, dones=dones)
    assert np.unique(ids).tolist() == [0]
    explicit = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    ids = recording_ids(n, times, dones=dones, explicit_ids=explicit)
    assert ids.tolist() == [0] * 4 + [1] * 4


def test_prev_frame_never_crosses_a_recording_boundary(tmp_path) -> None:
    path = tmp_path / "recordings.h5"
    _tiny_h5(path)
    with h5py.File(path, "a") as f:
        f.create_dataset("recording_id", data=np.array([0, 1, 1], dtype=np.int32))
    ds = H5DemoDataset(path)
    sample = ds[1]
    assert np.array_equal(sample["prev_frame"], sample["frame"])
    ds.close()


def test_injection_gate_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr("atomspike.capture.mc_predict._hwnd_is_foreground", lambda _hwnd: True)
    monkeypatch.setattr("atomspike.capture.mc_predict.is_window_usable", lambda _hwnd: True)
    assert inject_gate({"singleplayer": True}, hwnd=0)["reason"] == "no_window"
    assert inject_gate({"singleplayer": True}, hwnd=1, sidecar_age_s=2.0)["reason"] == "stale_sidecar"
    assert inject_gate({}, hwnd=1, sidecar_age_s=0.0)["reason"] == "unverified_world"
    assert (
        inject_gate(
            {"singleplayer": True, "session_nonce": "wrong"},
            hwnd=1,
            sidecar_age_s=0.0,
            expected_nonce="expected",
        )["reason"]
        == "unauthenticated_sidecar"
    )
    ok = inject_gate({"singleplayer": True, "mouse_grabbed": True}, hwnd=1, sidecar_age_s=0.1)
    assert ok["ok"] and ok["buttons"]


def test_injection_gate_rejects_hidden_or_minimized_window(monkeypatch) -> None:
    monkeypatch.setattr("atomspike.capture.mc_predict._hwnd_is_foreground", lambda _hwnd: True)
    monkeypatch.setattr("atomspike.capture.mc_predict.is_window_usable", lambda _hwnd: False)
    gate = inject_gate({"singleplayer": True}, hwnd=1, sidecar_age_s=0.1)
    assert gate["reason"] == "unusable_window"


def test_injection_rejects_a_region_saved_in_the_config(monkeypatch) -> None:
    cfg = AtomSpikeConfig()
    cfg.capture.region = (0, 0, 100, 100)
    monkeypatch.setattr("atomspike.capture.mc_predict.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="named Minecraft window"):
        run_minecraft_predict(cfg, None, inject=True, seconds=0.1)


def test_dry_run_refuses_to_modify_an_existing_recording(tmp_path) -> None:
    path = tmp_path / "recording.h5"
    path.write_bytes(b"keep this recording")
    with pytest.raises(RuntimeError, match="refuses to modify"):
        run_minecraft_collector(AtomSpikeConfig(), path, seconds=0.1, dry_run=True)
    assert path.read_bytes() == b"keep this recording"


def test_append_refuses_to_relabel_an_old_key_slot(tmp_path) -> None:
    path = tmp_path / "recording.h5"
    writer = StreamingDemoWriter(path, (4, 5), 2, 2, key_names=("W", "F"))
    writer.close()
    with pytest.raises(ValueError, match="key_names differ"):
        StreamingDemoWriter(path, (4, 5), 2, 2, key_names=("W", "E"), append=True)


def test_each_streaming_append_gets_a_distinct_recording_id(tmp_path) -> None:
    path = tmp_path / "recording.h5"
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    state = np.zeros(2, dtype=np.float32)
    action = np.zeros(2, dtype=np.int64)
    writer = StreamingDemoWriter(path, (4, 5), 2, 2, key_names=("W", "E"))
    writer.add(frame, state, action, 0.0, False, 1)
    writer.close()
    writer = StreamingDemoWriter(path, (4, 5), 2, 2, key_names=("W", "E"), append=True)
    writer.add(frame, state, action, 0.0, False, 2)
    writer.close()
    with h5py.File(path, "r") as f:
        assert np.asarray(f["recording_id"], dtype=np.int64).tolist() == [0, 1]


def test_sidecar_rejects_nonfinite_and_vector_stays_finite() -> None:
    assert sanitize_sidecar_payload({"hp": None}) is None
    assert sanitize_sidecar_payload({"hp": float("nan")}) is None
    assert sanitize_sidecar_payload({"hp": 20.0, "singleplayer": True}) == {
        "hp": 20.0,
        "singleplayer": True,
    }
    assert sanitize_sidecar_payload({"session_nonce": "ab" * 16}) == {
        "session_nonce": "ab" * 16,
    }
    vec = fields_to_vector({"hp": None, "yaw": float("nan"), "mouse_nx": "bad"}, dim=19)
    assert np.isfinite(vec).all()


def test_return_aware_training_rejects_empty_rewards() -> None:
    ds = SimpleNamespace(rewards=np.zeros(4, dtype=np.float32), dones=np.array([0, 0, 0, 1], dtype=np.uint8))
    with pytest.raises(ValueError, match="requires finite"):
        require_informative_rewards(ds, purpose="offline RL")


def test_pmsm_refuses_a_global_threshold_without_layer_calibration() -> None:
    model = nn.Sequential(Act())
    with pytest.raises(TypeError):
        enable_pmsm(model, threshold=0.05)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="per-layer calibration"):
        enable_pmsm(model, calibration={})


def test_final_gradient_accumulation_group_uses_its_own_size() -> None:
    assert [grad_accum_divisor(i, 10, 4) for i in range(10)] == [4] * 8 + [2, 2]
