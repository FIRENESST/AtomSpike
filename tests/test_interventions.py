from __future__ import annotations

import json

import pytest

from atomspike.cli import main
from atomspike.data.effects import EFFECT_EVENTS
from atomspike.eval.interventions import (
    RESULT_SCHEMA, evaluate_pair_results, make_pair_plan, write_pair_plan,
)


def _outcomes(**changes: int) -> dict[str, int]:
    values = {name: 0 for name in EFFECT_EVENTS}
    values.update(changes)
    return values


def _write_results(path, plan, *, treatment=None, control=None) -> None:
    treatment = _outcomes(goal_progress=1) if treatment is None else treatment
    control = _outcomes(damage=1) if control is None else control
    rows = []
    for pair in plan:
        for arm, outcome in (("treatment", treatment), ("control", control)):
            rows.append({
                "schema": RESULT_SCHEMA,
                "experiment_id": pair["experiment_id"],
                "pair_id": pair["pair_id"],
                "arm": arm,
                "order": int(arm != pair["first_arm"]),
                "executed_action": pair[f"{arm}_action"],
                "observed_ticks": pair["horizon_ticks"],
                "telemetry_valid": True,
                "start_state_sha256": pair["snapshot_sha256"],
                "outcomes": outcome,
            })
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _plan(tmp_path, monkeypatch, n=2):
    assignments = iter(i % 2 for i in range(n))
    monkeypatch.setattr("atomspike.eval.interventions.secrets.randbelow", lambda _n: next(assignments))
    snapshots = []
    for i in range(n):
        path = tmp_path / f"world-{i}.zip"
        path.write_bytes(f"snapshot {i}".encode())
        snapshots.append(path)
    rows = make_pair_plan(snapshots, treatment="attack", control="idle", goal="collect_log", horizon_ticks=45)
    plan_path = tmp_path / "plan.jsonl"
    write_pair_plan(plan_path, rows)
    return plan_path, rows


def test_paired_intervention_report_is_descriptive_and_audited(tmp_path, monkeypatch) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)
    assert [p["first_arm"] for p in plan] == ["treatment", "control"]
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path, plan)
    report = evaluate_pair_results(plan_path, results_path)
    assert report["n_pairs"] == 2
    assert report["minimum_pairs_met"] is False
    assert report["causal_claim_verified"] is False
    assert report["events"]["goal_progress"]["paired_risk_difference"] == 1.0
    assert report["events"]["goal_progress"]["exact_sign_p_exploratory"] == 0.5
    assert report["events"]["damage"]["paired_risk_difference"] == -1.0
    assert report["events"]["block_change"]["exact_sign_p_exploratory"] == 1.0
    with pytest.raises(ValueError, match="overwrite"):
        write_pair_plan(plan_path, plan)


@pytest.mark.parametrize("damage", ["missing_arm", "wrong_order", "wrong_start", "bad_label", "duplicate_arm"])
def test_invalid_paired_results_are_rejected(tmp_path, monkeypatch, damage: str) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path, plan)
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    if damage == "missing_arm":
        rows.pop()
    elif damage == "wrong_order":
        rows[0]["order"] = 1 - rows[0]["order"]
    elif damage == "wrong_start":
        rows[1]["start_state_sha256"] = "0" * 64
    elif damage == "bad_label":
        rows[0]["outcomes"]["goal_progress"] = 0.5
    else:
        rows.append(dict(rows[0]))
    results_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_pair_results(plan_path, results_path)


def test_plan_refuses_duplicate_snapshots_and_json_tricks(tmp_path, monkeypatch) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)
    snapshot = tmp_path / "world-0.zip"
    with pytest.raises(ValueError, match="duplicate file content"):
        make_pair_plan([snapshot, snapshot], treatment="attack", control="idle", goal="x", horizon_ticks=1)
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path, plan)
    line = results_path.read_text(encoding="utf-8").splitlines()[0]
    results_path.write_text(line[:-1] + ',"order":0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluate_pair_results(plan_path, results_path)


def test_truncated_intervention_plan_is_rejected(tmp_path, monkeypatch) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path, plan)
    plan_path.write_text(json.dumps(plan[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete or truncated"):
        evaluate_pair_results(plan_path, results_path)


def test_intervention_cli_rejects_incomplete_results(tmp_path, monkeypatch, capsys) -> None:
    plan_path, plan = _plan(tmp_path, monkeypatch)
    results_path = tmp_path / "results.jsonl"
    _write_results(results_path, plan)
    assert main(["eval-interventions", "--plan", str(plan_path), "--results", str(results_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["n_pairs"] == 2
    results_path.write_text("{}\n", encoding="utf-8")
    assert main(["eval-interventions", "--plan", str(plan_path), "--results", str(results_path)]) == 2


def test_plan_interventions_cli_creates_manifest_without_overwrite(tmp_path, capsys) -> None:
    snapshot = tmp_path / "world.zip"
    snapshot.write_bytes(b"independent world snapshot")
    plan_path = tmp_path / "plan.jsonl"
    args = [
        "plan-interventions", "--snapshots", str(snapshot),
        "--treatment", "attack", "--control", "idle",
        "--goal", "collect_log", "--out", str(plan_path),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["pairs"] == 1
    assert json.loads(plan_path.read_text(encoding="utf-8"))["planned_pairs"] == 1
    assert main(args) == 2
    assert "overwrite" in capsys.readouterr().err
