from __future__ import annotations

from pathlib import Path

from atomspike.train.common import _archive_current, _rotate_keep_k, _sha256


def test_checkpoint_rotation_archives_only_numbered_generations(tmp_path: Path) -> None:
    current = tmp_path / "bc.pt"
    current.write_bytes(b"one")
    current.with_suffix(".pt.sha256").write_text(
        f"{_sha256(current)}  bc.pt\n", encoding="utf-8"
    )
    plastic = tmp_path / "bc.plastic.pt"
    plastic.write_bytes(b"plastic")

    first = _archive_current(current)
    assert first is not None and first.name == "bc.step-000001.pt"
    current.write_bytes(b"two")
    second = _archive_current(current)
    assert second is not None and second.name == "bc.step-000002.pt"
    current.write_bytes(b"three")
    third = _archive_current(current)
    assert third is not None and third.name == "bc.step-000003.pt"

    _rotate_keep_k(current, keep_k=3)

    assert not first.exists()
    assert second.exists() and third.exists()
    assert plastic.read_bytes() == b"plastic"
