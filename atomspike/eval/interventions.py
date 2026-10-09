"""Auditable paired-intervention plans and offline outcome comparisons.

This module never controls Minecraft. A plan records randomized arm order for
distinct world snapshots; the user must actually restore each snapshot before
both arms. JSON results are self-reported, so this evaluator cannot certify
that the intervention protocol was followed or establish causality by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import uuid
from pathlib import Path
from typing import Any

from atomspike.data.effects import EFFECT_ACTIONS, EFFECT_EVENTS

PLAN_SCHEMA = "scb_pair_plan_v1"
RESULT_SCHEMA = "scb_trial_result_v1"


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line, object_pairs_hook=_object_no_duplicates, parse_constant=_reject_constant)
            except ValueError as exc:
                raise ValueError(f"{source}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{source}:{line_no}: expected a JSON object")
            rows.append(item)
    if not rows:
        raise ValueError(f"{source}: no trial rows")
    return rows


def _hex_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _snapshot_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"world snapshot archive not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pair_plan(
    snapshots: list[str | Path], *, treatment: str, control: str, goal: str, horizon_ticks: int,
) -> list[dict]:
    """One randomized pair per distinct snapshot; no gameplay or injection."""
    if not snapshots:
        raise ValueError("at least one world snapshot archive is required")
    if treatment not in EFFECT_ACTIONS or control not in EFFECT_ACTIONS or treatment == control:
        raise ValueError(f"treatment/control must be different action macros: {EFFECT_ACTIONS}")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty declared task name")
    if isinstance(horizon_ticks, bool) or not isinstance(horizon_ticks, int) or horizon_ticks <= 0:
        raise ValueError("horizon_ticks must be a positive integer")
    digests = [_snapshot_sha256(p) for p in snapshots]
    if len(set(digests)) != len(digests):
        raise ValueError("each pair needs a distinct world snapshot; duplicate file content found")
    experiment_id = str(uuid.uuid4())
    return [
        {
            "schema": PLAN_SCHEMA,
            "experiment_id": experiment_id,
            "pair_id": str(uuid.uuid4()),
            "planned_pairs": len(snapshots),
            "snapshot_sha256": digest,
            "snapshot_file": str(Path(path).resolve()),
            "goal": goal.strip(),
            "treatment_action": treatment,
            "control_action": control,
            "horizon_ticks": horizon_ticks,
            "first_arm": "treatment" if secrets.randbelow(2) == 0 else "control",
        }
        for path, digest in zip(snapshots, digests)
    ]


def write_pair_plan(path: str | Path, rows: list[dict]) -> None:
    """Create a new manifest, refusing to overwrite an existing experiment."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite an existing intervention plan: {target}") from exc


def _require_plan(rows: list[dict]) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    snapshots: set[str] = set()
    reference: tuple[Any, ...] | None = None
    for row in rows:
        if row.get("schema") != PLAN_SCHEMA:
            raise ValueError("unsupported intervention plan schema")
        pair_id = row.get("pair_id")
        experiment_id = row.get("experiment_id")
        snapshot = _hex_digest(row.get("snapshot_sha256"), "snapshot_sha256")
        treatment = row.get("treatment_action")
        control = row.get("control_action")
        horizon = row.get("horizon_ticks")
        planned_pairs = row.get("planned_pairs")
        goal = row.get("goal")
        if not isinstance(pair_id, str) or not pair_id or not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("plan requires non-empty pair_id and experiment_id")
        if pair_id in pairs or snapshot in snapshots:
            raise ValueError("plan contains a duplicate pair_id or world snapshot")
        if treatment not in EFFECT_ACTIONS or control not in EFFECT_ACTIONS or treatment == control:
            raise ValueError("plan has invalid or identical treatment/control actions")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("plan horizon_ticks must be a positive integer")
        if isinstance(planned_pairs, bool) or not isinstance(planned_pairs, int) or planned_pairs <= 0:
            raise ValueError("plan planned_pairs must be a positive integer")
        if not isinstance(goal, str) or not goal.strip() or row.get("first_arm") not in {"treatment", "control"}:
            raise ValueError("plan has invalid goal or randomized arm order")
        config = (experiment_id, treatment, control, horizon, goal, planned_pairs)
        if reference is not None and config != reference:
            raise ValueError("all planned pairs must share experiment, actions, goal and horizon")
        reference = config
        pairs[pair_id] = row
        snapshots.add(snapshot)
    if len(pairs) != reference[-1]:
        raise ValueError("intervention plan is incomplete or truncated")
    return pairs


def _binary_outcomes(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(EFFECT_EVENTS):
        raise ValueError(f"outcomes must contain exactly {EFFECT_EVENTS}")
    if any(type(v) is not int or v not in {0, 1} for v in value.values()):
        raise ValueError("each outcome must be an integer 0 or 1")
    return value


def _exact_sign_p(n_positive: int, n_negative: int) -> float:
    discordant = n_positive + n_negative
    if discordant == 0:
        return 1.0
    tail = min(n_positive, n_negative)
    log_terms = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1) - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(tail + 1)
    ]
    top = max(log_terms)
    return min(1.0, 2.0 * math.exp(top) * sum(math.exp(v - top) for v in log_terms))


def evaluate_pair_results(plan_path: str | Path, results_path: str | Path, *, min_pairs: int = 20) -> dict:
    """Validate pair integrity, then report descriptive matched risk differences."""
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
        raise ValueError("min_pairs must be positive")
    pairs = _require_plan(_read_jsonl(plan_path))
    results: dict[str, dict[str, dict]] = {pair_id: {} for pair_id in pairs}
    for row in _read_jsonl(results_path):
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError("unsupported intervention result schema")
        pair_id = row.get("pair_id")
        if pair_id not in pairs:
            raise ValueError(f"result references an unplanned pair: {pair_id}")
        plan = pairs[pair_id]
        arm = row.get("arm")
        if arm not in {"treatment", "control"} or arm in results[pair_id]:
            raise ValueError(f"pair {pair_id} has an invalid or duplicate arm")
        expected_order = 0 if arm == plan["first_arm"] else 1
        expected_action = plan[f"{arm}_action"]
        if row.get("experiment_id") != plan["experiment_id"]:
            raise ValueError(f"pair {pair_id} experiment_id mismatch")
        if type(row.get("order")) is not int or row["order"] != expected_order:
            raise ValueError(f"pair {pair_id} did not follow the randomized arm order")
        if (
            row.get("executed_action") != expected_action
            or type(row.get("observed_ticks")) is not int
            or row["observed_ticks"] != plan["horizon_ticks"]
        ):
            raise ValueError(f"pair {pair_id} action or observation horizon mismatch")
        if row.get("telemetry_valid") is not True:
            raise ValueError(f"pair {pair_id} lacks an affirmed valid outcome observation")
        _hex_digest(row.get("start_state_sha256"), "start_state_sha256")
        _binary_outcomes(row.get("outcomes"))
        results[pair_id][arm] = row
    for pair_id, arms in results.items():
        if set(arms) != {"treatment", "control"}:
            raise ValueError(f"pair {pair_id} must have one treatment and one control result")
        if arms["treatment"]["start_state_sha256"] != arms["control"]["start_state_sha256"]:
            raise ValueError(f"pair {pair_id} did not report the same starting state")
    n = len(pairs)
    events = {}
    for name in EFFECT_EVENTS:
        n10 = sum(int(arms["treatment"]["outcomes"][name] == 1 and arms["control"]["outcomes"][name] == 0)
                  for arms in results.values())
        n01 = sum(int(arms["treatment"]["outcomes"][name] == 0 and arms["control"]["outcomes"][name] == 1)
                  for arms in results.values())
        n11 = sum(int(arms["treatment"]["outcomes"][name] == 1 and arms["control"]["outcomes"][name] == 1)
                  for arms in results.values())
        p = _exact_sign_p(n10, n01)
        events[name] = {
            "treatment_only": n10,
            "control_only": n01,
            "both": n11,
            "neither": n - n10 - n01 - n11,
            "paired_risk_difference": (n10 - n01) / n,
            "exact_sign_p_exploratory": p,
            "bonferroni_p_exploratory": min(1.0, len(EFFECT_EVENTS) * p),
        }
    first_treatment = sum(p["first_arm"] == "treatment" for p in pairs.values())
    return {
        "experiment_id": next(iter(pairs.values()))["experiment_id"],
        "n_pairs": n,
        "distinct_snapshots": n,
        "treatment_first": first_treatment,
        "control_first": n - first_treatment,
        "minimum_pairs_met": n >= min_pairs,
        "events": events,
        "causal_claim_verified": False,
        "audit_warning": (
            "Manifest randomizes order and results pass structural checks, but this tool cannot verify "
            "world restoration, action execution, telemetry truth, or independence. P-values are exploratory."
        ),
    }
