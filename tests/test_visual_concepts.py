from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from atomspike.capture.mc_state import sanitize_sidecar_payload
from atomspike.config import AtomSpikeConfig
from atomspike.data.io import H5DemoDataset
from atomspike.data.stream import StreamingDemoWriter
from atomspike.data.augment import augment_frame_pair
from atomspike.data.visual_concepts import CONCEPT_KINDS, UNOBSERVED, concept_from_signal
from atomspike.models.semantic_probe import CrosshairConceptProbe
from atomspike.train.bc import _total_loss


def test_crosshair_labels_require_a_fresh_sidecar() -> None:
    log = {"look_block_id": "minecraft:oak_log", "look_miss": False}
    assert concept_from_signal(log) == CONCEPT_KINDS.index("log")
    assert concept_from_signal(log, sidecar_fresh=False) == UNOBSERVED
    assert concept_from_signal({"look_miss": True}) == CONCEPT_KINDS.index("air")
    assert concept_from_signal({"look_miss": False}) == UNOBSERVED
    assert concept_from_signal({"look_entity_id": "minecraft:zombie"}) == CONCEPT_KINDS.index("entity")
    assert concept_from_signal({**log, "gui_open": True}) == UNOBSERVED
    clean = sanitize_sidecar_payload(log)
    assert clean is not None and clean["look_block_id"] == "minecraft:oak_log"


def test_legacy_stream_rows_remain_unobserved_after_append(tmp_path) -> None:
    path = tmp_path / "demo.h5"
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    state = np.zeros(2, dtype=np.float32)
    action = np.zeros(2, dtype=np.int64)
    writer = StreamingDemoWriter(path, (4, 5), 2, 2)
    writer.add(frame, state, action, 0.0, False, 1)
    writer.close()
    with h5py.File(path, "a") as f:
        del f["look_concept"]  # emulate a recording from before this schema
    writer = StreamingDemoWriter(path, (4, 5), 2, 2, append=True)
    writer.add(frame, state, action, 0.0, False, 2, look_concept=CONCEPT_KINDS.index("log"))
    writer.close()
    ds = H5DemoDataset(path)
    assert ds.look_concept.tolist() == [-1, CONCEPT_KINDS.index("log")]
    assert ds[0]["look_concept"].tolist() == [-1]
    with pytest.raises(ValueError, match="look_concept"):
        writer = StreamingDemoWriter(tmp_path / "invalid.h5", (4, 5), 2, 2)
        try:
            writer.add(frame, state, action, 0.0, False, 1, look_concept=999)
        finally:
            writer.close()


def test_screen_probe_uses_only_crosshair_tokens() -> None:
    probe = CrosshairConceptProbe(8)
    tokens = torch.randn(2, 25, 8)
    changed = tokens.clone()
    changed[:, 0] += 1000.0  # outside the 3x3 crosshair neighborhood
    assert torch.allclose(probe(tokens, (5, 5)), probe(changed, (5, 5)))
    assert probe(tokens, (5, 5)).shape == (2, len(CONCEPT_KINDS))
    with pytest.raises(ValueError, match="spatial"):
        probe(tokens, (4, 5))


def test_erase_does_not_destroy_a_labelled_crosshair() -> None:
    frame = torch.ones(2, 3, 16, 16)
    cfg = SimpleNamespace(
        aug_brightness=0.0, aug_contrast=0.0, aug_saturation=0.0,
        aug_noise=0.0, aug_erase=1.0, aug_erase_scale=0.5,
    )
    changed, _ = augment_frame_pair(
        frame, frame, cfg, protect_crosshair=torch.tensor([[True], [False]])
    )
    assert torch.equal(changed[0], frame[0])
    assert not torch.equal(changed[1], frame[1])


def test_semantic_loss_ignores_missing_labels_and_trains_when_present() -> None:
    cfg = AtomSpikeConfig()
    cfg.planner.enabled = False
    visual_logits = torch.zeros(1, len(CONCEPT_KINDS), requires_grad=True)
    out = {"logits": [torch.zeros(1, 2, requires_grad=True)], "look_concept_logits": visual_logits}
    args = (out, torch.tensor([[0]]), torch.ones(1, 1), torch.ones(1), None, cfg, {})
    baseline = _total_loss(*args, look_target=torch.tensor([[-1]]))
    labelled = _total_loss(*args, look_target=torch.tensor([[CONCEPT_KINDS.index("log")]]))
    assert labelled > baseline
    labelled.backward()
    assert visual_logits.grad is not None and visual_logits.grad.abs().sum() > 0
