from __future__ import annotations

from typing import Any

from .common import clone, digest, get_path, set_path, state_hash

VALIDATOR_B_VERSION = "pcg_v3_m11.terminal_validator_b.log_replay.v1"


def validate_terminal_state_b(
    pre_state: dict[str, Any],
    structured_action: dict[str, Any],
    transaction_log: list[dict[str, Any]],
    action_semantics: dict[str, Any],
    terminal_invariants: list[dict[str, Any]],
    recorded_post_state: dict[str, Any],
) -> dict[str, Any]:
    replay = clone(pre_state)
    errors: list[str] = []
    prev_hash = state_hash(pre_state)
    seen_sequences: set[int] = set()
    for expected_sequence, entry in enumerate(transaction_log, start=1):
        if (
            entry.get("sequence") != expected_sequence
            or entry.get("sequence") in seen_sequences
        ):
            errors.append("TRANSACTION_LOG_INCONSISTENT")
        seen_sequences.add(int(entry.get("sequence", -1)))
        if entry.get("transaction_id") != structured_action.get("transaction_id"):
            errors.append("TRANSACTION_LOG_INCONSISTENT")
        if entry.get("prev_hash") != prev_hash:
            errors.append("TRANSACTION_LOG_INCONSISTENT")
        copied = {key: value for key, value in entry.items() if key != "entry_hash"}
        if digest(copied) != entry.get("entry_hash"):
            errors.append("TRANSACTION_LOG_INCONSISTENT")
        prev_hash = str(entry.get("entry_hash"))
        try:
            _apply_log_entry(replay, entry)
        except Exception:
            errors.append("TRANSACTION_LOG_INCOMPLETE")
    invariant_rows = []
    for item in terminal_invariants:
        actual = get_path({"state": replay}, "state." + item["path"])
        invariant_rows.append(
            {
                "path": item["path"],
                "expected": item["equals"],
                "actual": actual,
                "passed": actual == item["equals"],
            }
        )
    replay_match = replay == recorded_post_state
    if not replay_match:
        errors.append("REPLAY_STATE_MISMATCH")
    if not all(row["passed"] for row in invariant_rows):
        errors.append("TERMINAL_INVARIANT_FAILED")
    if not transaction_log:
        errors.append("TRANSACTION_LOG_INCOMPLETE")
    success = not errors and _postcondition_ok(
        structured_action, action_semantics, replay
    )
    if not success and "TERMINAL_STATE_INCORRECT" not in errors and not errors:
        errors.append("TERMINAL_STATE_INCORRECT")
    return {
        "schema_version": "pcg_v3_m11.validator_b_result.v1",
        "validator_version": VALIDATOR_B_VERSION,
        "reconstructed_post_state_hash": state_hash(replay),
        "recorded_post_state_hash": state_hash(recorded_post_state),
        "replay_state_matches_recorded_post_state": replay_match,
        "terminal_invariant_rows": invariant_rows,
        "terminal_state_success": success,
        "failure_codes": tuple(sorted(set(errors))),
        "result_hash": digest(
            {"replay_hash": state_hash(replay), "errors": errors, "success": success}
        ),
    }


def _apply_log_entry(state: dict[str, Any], entry: dict[str, Any]) -> None:
    op = entry["op"]
    path = entry["path"]
    if op == "set":
        if get_path(state, path) != entry["old"]:
            raise ValueError("old_value_mismatch")
        set_path(state, path, entry["new"])
    elif op == "append":
        if get_path(state, path) != entry["old"]:
            raise ValueError("old_value_mismatch")
        set_path(state, path, entry["new"])
    elif op in {"noop", "failed_attempt"}:
        return
    else:
        raise ValueError("unsupported_log_op")


def _postcondition_ok(
    action: dict[str, Any], semantics: dict[str, Any], state: dict[str, Any]
) -> bool:
    del semantics
    action_type = action["action_type"]
    objects = state["mutable_operational_state"]["objects"]
    if action_type == "apply_current_bundle":
        return all(objects[target]["applied"] for target in action["primary_target"])
    if action_type == "select_current_interface":
        return objects[action.get("optional_interface") or action["primary_target"]][
            "selected"
        ]
    if action_type == "migrate_to_allowed_path":
        return objects[action["primary_target"]]["path_id"] == action["optional_path"]
    if action_type == "replace_current_object":
        return (
            objects[action["optional_replacement_target"]]["current"]
            and not objects[action["primary_target"]]["current"]
        )
    if action_type == "allow_or_deny_operation":
        return bool(state.get("action_history"))
    return False
