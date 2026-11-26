from __future__ import annotations

from typing import Any

from .common import diff_operational, digest, get_path, normalized

VALIDATOR_A_VERSION = "pcg_v3_m11.terminal_validator_a.direct_delta.v1"


def validate_terminal_state_a(
    pre_state: dict[str, Any],
    actual_post_state: dict[str, Any],
    expected_state_delta: list[dict[str, Any]],
    terminal_invariants: list[dict[str, Any]],
    forbidden_side_effect_spec: dict[str, Any],
) -> dict[str, Any]:
    actual_delta = diff_operational(pre_state, actual_post_state)
    expected_norm = normalized(expected_state_delta)
    actual_norm = normalized(actual_delta)
    missing = [row for row in expected_norm if row not in actual_norm]
    unexpected = [row for row in actual_norm if row not in expected_norm]
    invariant_rows = []
    for item in terminal_invariants:
        actual = get_path({"state": actual_post_state}, "state." + item["path"])
        invariant_rows.append(
            {
                "path": item["path"],
                "expected": item["equals"],
                "actual": actual,
                "passed": actual == item["equals"],
            }
        )
    forbidden = bool(unexpected) or _forbidden_object_changed(
        unexpected, forbidden_side_effect_spec
    )
    success = (
        not missing
        and not unexpected
        and all(row["passed"] for row in invariant_rows)
        and not forbidden
    )
    failure_codes = []
    if missing:
        failure_codes.append("REQUIRED_STATE_CHANGE_MISSING")
    if unexpected:
        failure_codes.append("UNEXPECTED_STATE_UPDATE")
    if forbidden:
        failure_codes.append("FORBIDDEN_SIDE_EFFECT")
    if not all(row["passed"] for row in invariant_rows):
        failure_codes.append("TERMINAL_INVARIANT_FAILED")
    return {
        "schema_version": "pcg_v3_m11.validator_a_result.v1",
        "validator_version": VALIDATOR_A_VERSION,
        "required_additions": [row for row in expected_norm if row["op"] == "add"],
        "required_removals": [row for row in expected_norm if row["op"] == "remove"],
        "required_updates": [row for row in expected_norm if row["op"] == "update"],
        "missing_required_changes": missing,
        "unexpected_changes": unexpected,
        "terminal_invariant_rows": invariant_rows,
        "forbidden_side_effect": forbidden,
        "terminal_state_success": success,
        "failure_codes": tuple(failure_codes),
        "result_hash": digest(
            {"delta": actual_delta, "invariants": invariant_rows, "success": success}
        ),
    }


def _forbidden_object_changed(
    unexpected: list[dict[str, Any]], spec: dict[str, Any]
) -> bool:
    forbidden_prefixes = tuple(spec.get("forbidden_path_prefixes", ()))
    return any(
        row["path"].startswith(prefix)
        for row in unexpected
        for prefix in forbidden_prefixes
    )
