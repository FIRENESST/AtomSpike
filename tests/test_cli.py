from __future__ import annotations

import json

import pytest

from atomspike.cli import build_parser, main


def test_region_is_parsed_once() -> None:
    args = build_parser().parse_args(["collect-mc", "--region", "10,20,800,450", "--dry-run"])
    assert args.region == (10, 20, 800, 450)


@pytest.mark.parametrize(
    "argv",
    [
        ["collect-mc", "--region", "0,0,-1,20"],
        ["predict-mc", "--ckpt", "model.pt", "--learn", "--no-learn"],
        ["predict-mc", "--ckpt", "model.pt", "--viz", "--no-viz"],
        ["sidecar-send", "--host", "192.0.2.1"],
        ["collect", "--out", "demo.h5", "--episodes", "0"],
    ],
)
def test_invalid_cli_values_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_doctor_json_is_available_without_importing_torch(capsys) -> None:
    code = main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code in {0, 1}
    assert report["python"]["ok"]
    assert "torch" in report["dependencies"]
    assert set(report["configs"]) == {"default.yaml", "smoke.yaml", "minecraft.yaml"}
