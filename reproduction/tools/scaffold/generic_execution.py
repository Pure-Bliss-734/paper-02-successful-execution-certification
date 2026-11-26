from __future__ import annotations
from typing import Any
from pcg_v3_m11.executor import execute_action
from pcg_v3_m27r.execution import action_envelope


def strict_action_contract_match(observed: Any, expected: dict[str, Any]) -> bool:
    """Reproduce the frozen evaluator's action comparison exactly."""
    if not isinstance(observed, dict):
        return False
    action_type = expected.get("action_type")
    required = {"action_type", "primary_target"}
    if action_type == "select_current_interface":
        required.add("interface")
    elif action_type == "migrate_to_allowed_path":
        required.add("path")
    elif action_type == "replace_current_object":
        required.add("replacement_target")
    elif action_type == "allow_or_deny_operation":
        required.update({"actor_or_role", "decision"})
    for key in required:
        observed_value = observed.get(key)
        expected_value = expected.get(key)
        if (
            key == "primary_target"
            and isinstance(expected_value, list)
            and isinstance(observed_value, str)
        ):
            observed_value = [observed_value]
        if observed_value != expected_value:
            return False
    return isinstance(observed.get("schema_version"), str)


def observed_envelope(
    observed: Any, pre_state: dict[str, Any], task_id: str
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """Map the frozen model action object to the already frozen M11 envelope.

    The sole normalization mirrors the frozen strict scorer: an
    apply-current-bundle scalar target is represented as a one-item bundle.
    No missing action parameter is filled from scorer evidence.
    """
    if not isinstance(observed, dict):
        return None, [], "OBSERVED_ACTION_NOT_OBJECT"
    if not isinstance(observed.get("action_type"), str):
        return None, [], "OBSERVED_ACTION_TYPE_MISSING"
    if "primary_target" not in observed:
        return None, [], "OBSERVED_PRIMARY_TARGET_MISSING"
    adapted = dict(observed)
    events: list[str] = []
    if adapted["action_type"] == "apply_current_bundle" and isinstance(
        adapted.get("primary_target"), str
    ):
        adapted["primary_target"] = [adapted["primary_target"]]
        events.append("SCALAR_BUNDLE_TARGET_WRAPPED")
    try:
        return action_envelope(adapted, pre_state, task_id), events, None
    except (KeyError, TypeError, ValueError) as exc:
        return None, events, f"ACTION_ENVELOPE_ERROR:{type(exc).__name__}:{exc}"


def delta_invariants(expected_delta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Operational invariants independent of gold transaction identifiers."""
    return [
        {
            "path": "mutable_operational_state." + row["path"],
            "equals": row["new"],
        }
        for row in expected_delta
        if row["op"] in {"add", "update"}
    ]


def safe_execute(
    envelope: dict[str, Any] | None, pre_state: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    if envelope is None:
        return None, "NO_EXECUTABLE_ENVELOPE"
    try:
        return execute_action(pre_state, envelope, {}, {}), None
    except Exception as exc:  # Malformed model actions are evidence, not audit crashes.
        return None, f"EXECUTOR_EXCEPTION:{type(exc).__name__}:{exc}"
