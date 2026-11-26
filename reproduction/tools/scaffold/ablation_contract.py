from __future__ import annotations
from copy import deepcopy
from typing import Any
from pcg_v3.hashing import sha256_json
from pcg_v3_m11.schemas import ACTION_SCHEMA_VERSION as EXECUTOR_ACTION_SCHEMA_VERSION


ACTION_SCHEMA_VERSION = "pcg_v3.supplement.full_action_contract.v1"

ACTION_TYPES = (
    "apply_current_bundle",
    "select_current_interface",
    "migrate_to_allowed_path",
    "replace_current_object",
    "allow_or_deny_operation",
)

COMMON_REQUIRED = (
    "action_type",
    "primary_target",
    "declared_dependencies",
    "schema_version",
)

ACTION_REQUIRED: dict[str, tuple[str, ...]] = {
    "apply_current_bundle": COMMON_REQUIRED,
    "select_current_interface": COMMON_REQUIRED + ("interface",),
    "migrate_to_allowed_path": COMMON_REQUIRED + ("path",),
    "replace_current_object": COMMON_REQUIRED + ("replacement_target",),
    "allow_or_deny_operation": COMMON_REQUIRED
    + ("actor_or_role", "operation", "decision"),
}


def _string() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}


def _base_properties(action_type: str) -> dict[str, Any]:
    target: dict[str, Any]
    if action_type == "apply_current_bundle":
        target = {
            "type": "array",
            "items": _string(),
            "minItems": 1,
            "uniqueItems": True,
            "description": "Sorted identifiers of the current valid bundle components to apply.",
        }
    else:
        target = {
            **_string(),
            "description": {
                "select_current_interface": "Identifier of the unique currently valid interface; it must equal interface.",
                "migrate_to_allowed_path": "Identifier of the current object or configuration that must be migrated.",
                "replace_current_object": "Source identifier of the active replacement relationship, not its replacement target.",
                "allow_or_deny_operation": "Identifier of the object on which the operation is evaluated.",
            }[action_type],
        }
    return {
        "action_type": {"type": "string", "const": action_type},
        "primary_target": target,
        "declared_dependencies": {
            "type": "array",
            "items": _string(),
            "uniqueItems": True,
            "description": (
                "Sorted target identifiers of active dependency_relationships whose source is a primary_target. "
                "Use [] when there is no such edge. Do not include replacement targets, paths, permissions, "
                "context links, or the primary target itself unless an explicit active dependency edge requires it."
            ),
        },
        "schema_version": {"type": "string", "const": ACTION_SCHEMA_VERSION},
    }


def _action_schema(action_type: str) -> dict[str, Any]:
    properties = _base_properties(action_type)
    if action_type == "select_current_interface":
        properties["interface"] = {
            **_string(),
            "description": "The unique current valid interface identifier; equal to primary_target.",
        }
    elif action_type == "migrate_to_allowed_path":
        properties["path"] = {
            **_string(),
            "description": "The unique currently allowed destination path identifier.",
        }
    elif action_type == "replace_current_object":
        properties["replacement_target"] = {
            **_string(),
            "description": "Target identifier of the active replacement relationship whose source is primary_target.",
        }
    elif action_type == "allow_or_deny_operation":
        properties.update(
            {
                "actor_or_role": {
                    **_string(),
                    "description": "Actor or role named by the applicable current permission record.",
                },
                "operation": {"type": "string", "const": "execute"},
                "decision": {
                    "type": "string",
                    "enum": ["allow", "deny"],
                    "description": "allow only when the applicable current authorization permits execute; otherwise deny.",
                },
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(ACTION_REQUIRED[action_type]),
        "properties": properties,
    }


FULL_ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": ACTION_SCHEMA_VERSION,
    "title": "PCG Dynamic-State full action-specific model contract",
    "description": (
        "One exact semantic action. Transaction identifiers and executor-envelope "
        "fields are harness-owned and therefore absent."
    ),
    "oneOf": [_action_schema(action_type) for action_type in ACTION_TYPES],
    "x-action-types": list(ACTION_TYPES),
    "x-no-defaults": True,
    "x-no-semantic-repair": True,
}


def validate_full_action(action: Any) -> tuple[bool, list[str]]:
    """Validate without filling defaults or coercing values."""

    errors: list[str] = []
    if not isinstance(action, dict):
        return False, ["ACTION_NOT_OBJECT"]
    action_type = action.get("action_type")
    if action_type not in ACTION_REQUIRED:
        return False, ["ACTION_TYPE_UNSUPPORTED"]
    required = set(ACTION_REQUIRED[action_type])
    keys = set(action)
    for field in sorted(required - keys):
        errors.append(f"MISSING_REQUIRED_FIELD:{field}")
    for field in sorted(keys - required):
        errors.append(f"FORBIDDEN_FIELD:{field}")
    if action.get("schema_version") != ACTION_SCHEMA_VERSION:
        errors.append("SCHEMA_VERSION_MISMATCH")
    primary = action.get("primary_target")
    if action_type == "apply_current_bundle":
        if (
            not isinstance(primary, list)
            or not primary
            or not all(isinstance(item, str) and item for item in primary)
        ):
            errors.append("PRIMARY_TARGET_LIST_INVALID")
        elif len(set(primary)) != len(primary):
            errors.append("PRIMARY_TARGET_DUPLICATE")
    elif not isinstance(primary, str) or not primary:
        errors.append("PRIMARY_TARGET_INVALID")
    dependencies = action.get("declared_dependencies")
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) and item for item in dependencies
    ):
        errors.append("DECLARED_DEPENDENCIES_INVALID")
    elif len(set(dependencies)) != len(dependencies):
        errors.append("DECLARED_DEPENDENCIES_DUPLICATE")
    elif dependencies != sorted(dependencies):
        errors.append("DECLARED_DEPENDENCIES_NOT_CANONICAL")
    for field in required - {
        "action_type",
        "primary_target",
        "declared_dependencies",
        "schema_version",
        "decision",
        "operation",
    }:
        if not isinstance(action.get(field), str) or not action.get(field):
            errors.append(f"FIELD_INVALID:{field}")
    if action_type == "allow_or_deny_operation":
        if action.get("operation") != "execute":
            errors.append("OPERATION_INVALID")
        if action.get("decision") not in {"allow", "deny"}:
            errors.append("DECISION_INVALID")
    return not errors, errors


def action_equal(observed: Any, expected: dict[str, Any]) -> bool:
    valid, _ = validate_full_action(observed)
    return bool(valid and observed == expected)


def bridge_to_executor_envelope(
    action: dict[str, Any], *, task_id: str, row_id: str
) -> dict[str, Any]:
    """Convert a valid semantic action into the immutable M11 envelope."""

    valid, errors = validate_full_action(action)
    if not valid:
        raise ValueError("invalid full action: " + ",".join(errors))
    action_type = action["action_type"]
    return {
        "action_type": action_type,
        "primary_target": deepcopy(action["primary_target"]),
        "optional_secondary_target": None,
        "optional_path": action.get("path"),
        "optional_interface": action.get("interface"),
        "optional_replacement_target": action.get("replacement_target"),
        "optional_actor_or_role": action.get("actor_or_role"),
        "optional_decision": action.get("decision"),
        "declared_dependencies": deepcopy(action["declared_dependencies"]),
        "action_parameters": {"operation": action.get("operation")},
        "transaction_id": f"TX-SUP-{sha256_json([task_id, row_id, action])[:20]}",
        "action_schema_version": EXECUTOR_ACTION_SCHEMA_VERSION,
    }
