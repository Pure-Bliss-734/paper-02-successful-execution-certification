from __future__ import annotations
import json
from types import SimpleNamespace
from typing import Any
from pcg_v3.hashing import sha256_json

RunnerConfig = SimpleNamespace
FormalTaskPackage = SimpleNamespace
CONDITION_CONTRACTS = {
    name: {"state_representation": "natural" if name == "N" else "structured"}
    for name in ("N", "S", "B", "C")
}


VISIBLE_CHECKER_VERSION = "pcg_v3_m30.visible_category_checker.v1"

ALLOWED_CHECKER_CATEGORIES = (
    "STATE_SCHEMA_INVALID",
    "INVALIDATED_STATE_PRESENT",
    "CONFLICT_PRESENT",
    "CRITICAL_RELATION_MISSING",
    "REQUIRED_STATE_MISSING",
    "PATH_OR_PERMISSION_UNSATISFIED",
    "ACTION_NOT_UNIQUE",
    "NO_DETECTED_ERROR",
)

BLIND_REVIEW_FEEDBACK = "BLIND_REVIEW_NO_TASK_SPECIFIC_DIAGNOSTIC"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _base_request(
    *, messages: list[dict[str, str]], config: RunnerConfig
) -> dict[str, Any]:
    return {
        "model": config.model,
        "messages": messages,
        "thinking": {"type": config.thinking},
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
        "tools": [],
    }


def _state_contract(representation: str) -> str:
    if representation == "natural":
        return (
            "carried_state must be a string using only these line forms: "
            "ENTITY <JSON object>, RELATION <JSON object>, and EXCLUSION <JSON object>. "
            "Each ENTITY JSON object must contain exactly entity_id, entity_type, and "
            "state; all status/value fields must be nested inside the state object. "
            'For example: ENTITY {"entity_id":"E1","entity_type":"interface",'
            '"state":{"active":true,"allowed":true,"blocked":false,'
            '"current":true,"protected":false,"revoked":false,'
            '"stale":false,"value":"v1"}}. Do not flatten state fields '
            "next to entity_id or entity_type. Each RELATION JSON object must contain "
            "exactly relation_id, source_id, target_id, relation_type, and direction. "
            "Each EXCLUSION JSON object must contain exactly entity_id and reason. "
            "Do not put relation_id, source_id, target_id, or excluded_event_id in an "
            "EXCLUSION row. "
            "Include every current entity and relation needed by the task, including "
            "inactive or invalidated entities when they are represented by an exclusion. "
            "Carry only currently valid relations; do not copy invalidated, superseded, "
            "or historical relation rows. "
            "Do not omit an exclusion record or current-state entity merely because "
            "it is not selected for the next action. "
            "For carried-state rows, do not copy surface_label, initial_state, distractor, "
            "critical, excluded_event_id, or other presentation metadata. "
            "Do not place the answer in carried_state."
        )
    if representation == "structured":
        return (
            "carried_state must be a JSON object with exactly entities, relations, "
            "and exclusions arrays. Entity rows contain entity_id, entity_type, and "
            "state. The state field must be a JSON object, never a scalar string. "
            "Each state object must contain exactly active, allowed, blocked, current, "
            "protected, revoked, stale, and value. The seven status fields are Boolean "
            'and value is a string. Example entity: {"entity_id":"E1",'
            '"entity_type":"interface","state":{"active":true,'
            '"allowed":true,"blocked":false,"current":true,'
            '"protected":false,"revoked":false,"stale":false,'
            '"value":"v1"}}. Relation rows contain relation_id, source_id, target_id, '
            "relation_type, and direction. Exclusion rows contain entity_id and reason. "
            "Include every current entity and relation needed by the task, including "
            "inactive or invalidated entities when they are represented by an exclusion. "
            "Carry only currently valid relations; do not copy invalidated, superseded, "
            "or historical relation rows. "
            "Do not omit an exclusion record or current-state entity merely because "
            "it is not selected for the next action. "
            "Do not copy surface_label, initial_state, distractor, critical, "
            "excluded_event_id, or other presentation metadata. "
            "Do not place the answer in carried_state."
        )
    raise ValueError("unsupported state representation")


def _action_semantics_contract(action_schema: dict[str, Any]) -> str:
    """State generic parameter roles without supplying task-specific values."""
    properties = action_schema.get("properties", {})
    action_types = properties.get("action_type", {}).get("enum", [])
    action_type = action_types[0] if len(action_types) == 1 else None
    contracts = {
        "apply_current_bundle": (
            "For apply_current_bundle, primary_target is an array containing every "
            "currently valid bundle component; exclude revoked, stale, blocked, "
            "protected, and superseded components."
        ),
        "select_current_interface": (
            "For select_current_interface, primary_target and interface must identify "
            "the same uniquely current valid interface."
        ),
        "migrate_to_allowed_path": (
            "For migrate_to_allowed_path, primary_target identifies the object being "
            "migrated and path identifies its currently allowed destination."
        ),
        "replace_current_object": (
            "For replace_current_object, primary_target identifies the current object "
            "being replaced and replacement_target identifies the valid target selected "
            "by the complete chronological replacement or rollback history. Do not "
            "select an earlier superseded target."
        ),
        "allow_or_deny_operation": (
            "For allow_or_deny_operation, primary_target identifies the permission "
            "target, actor_or_role identifies the actor, and decision is the result "
            "of the current permission and policy state."
        ),
    }
    return contracts.get(
        action_type, "Use the action schema's declared parameter roles exactly."
    )


def _answer_semantics_contract(answer_schema: dict[str, Any]) -> str:
    """Make the frozen answer-field predicates explicit without naming objects."""
    properties = answer_schema.get("properties", {})
    clauses: list[str] = []
    if "current_entity_ids" in properties:
        clauses.append(
            "current_entity_ids means only entity rows with current=true and active=true; "
            "do not include a path merely because it is allowed"
        )
    if "current_interface_ids" in properties:
        clauses.append(
            "current_interface_ids means current=true, active=true rows whose entity_type is interface"
        )
    if "allowed_path_ids" in properties:
        clauses.append(
            "allowed_path_ids means path rows with allowed=true and active=true"
        )
    if "allowed_permission_ids" in properties:
        clauses.append(
            "allowed_permission_ids means only permission targets with a currently valid "
            "permission_grant relation; if no such relation exists, return an empty array "
            "and never reuse a path ID"
        )
    if not clauses:
        return "Use the answer schema's declared field semantics exactly."
    return "For the answer fields, use these predicates: " + "; ".join(clauses) + "."


def build_initial_request(
    condition: str, task: FormalTaskPackage, config: RunnerConfig
) -> dict[str, Any]:
    if condition == "H":
        instruction = (
            "Use the complete visible history below. Return one JSON object with "
            "exactly two fields: answer and action. answer must follow ANSWER_SCHEMA; "
            "action must follow ACTION_SCHEMA. Do not add commentary."
        )
    elif condition in {"N", "S", "B", "C"}:
        representation = CONDITION_CONTRACTS[condition]["state_representation"]
        instruction = (
            "Use the complete visible history below. Return one JSON object with "
            "exactly two fields: answer and carried_state. answer must follow "
            f"ANSWER_SCHEMA. {_state_contract(representation)} Return JSON only."
        )
    else:
        raise ValueError("initial request is unavailable for this condition")
    user = (
        f"{instruction}\n\nVISIBLE_TASK:\n{task.model_visible_surface}\n\n"
        f"ANSWER_QUESTION:\n{task.answer_question}\n\n"
        f"ANSWER_FIELD_SEMANTICS:\n{_answer_semantics_contract(task.answer_schema)}\n\n"
        f"ACTION_QUESTION:\n{task.action_question}\n\n"
        f"ACTION_PARAMETER_ROLES:\n{_action_semantics_contract(task.action_schema)}\n\n"
        f"ANSWER_SCHEMA:\n{_json(task.answer_schema)}\n\n"
        f"ACTION_SCHEMA:\n{_json(task.action_schema)}"
    )
    return _base_request(
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the supplied evidence exactly. Return valid JSON. "
                    "Do not use tools, retrieval, or unstated facts."
                ),
            },
            {"role": "user", "content": user},
        ],
        config=config,
    )


def build_review_request(
    *,
    condition: str,
    task: FormalTaskPackage,
    candidate_state: Any,
    feedback_category: str,
    config: RunnerConfig,
) -> dict[str, Any]:
    if condition not in {"B", "C"}:
        raise ValueError("review stage is limited to B and C")
    expected = BLIND_REVIEW_FEEDBACK if condition == "B" else feedback_category
    if condition == "C" and expected not in ALLOWED_CHECKER_CATEGORIES:
        raise ValueError("checker feedback category is not frozen")
    shell = (
        "Review the candidate state once. Preserve the frozen first-stage answer; "
        "it is intentionally absent from this request and must not be reconstructed. "
        "Return one JSON object with exactly one field, carried_state. "
        f"{_state_contract('structured')} Return JSON only."
    )
    user = (
        f"{shell}\n\nCANDIDATE_STATE:\n{_json(candidate_state)}\n\n"
        f"REVIEW_FEEDBACK_CATEGORY:\n{expected}\n\n"
        f"ACTION_QUESTION:\n{task.action_question}\n\n"
        f"ACTION_PARAMETER_ROLES:\n{_action_semantics_contract(task.action_schema)}"
    )
    return _base_request(
        messages=[
            {
                "role": "system",
                "content": (
                    "The feedback contains no correct identifier, path, permission, "
                    "or action. Return valid JSON and perform at most one repair."
                ),
            },
            {"role": "user", "content": user},
        ],
        config=config,
    )


def build_state_only_action_request(
    *,
    task: FormalTaskPackage,
    carried_state: Any,
    config: RunnerConfig,
) -> dict[str, Any]:
    user = (
        "Use only CARRIED_STATE to answer ACTION_QUESTION. The original history and "
        "first-stage answer are unavailable. Return exactly the ACTION_SCHEMA JSON "
        "object with no wrapper or commentary.\n\n"
        f"CARRIED_STATE:\n{_json(carried_state)}\n\n"
        f"ACTION_QUESTION:\n{task.action_question}\n\n"
        f"ACTION_PARAMETER_ROLES:\n{_action_semantics_contract(task.action_schema)}\n\n"
        f"ACTION_SCHEMA:\n{_json(task.action_schema)}"
    )
    return _base_request(
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON derived only from the supplied state.",
            },
            {"role": "user", "content": user},
        ],
        config=config,
    )


def validate_json_contract(
    value: Any, schema: dict[str, Any], path: str = "$"
) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path}:expected_object"]
        required = schema.get("required", [])
        if not isinstance(required, list):
            errors.append(f"{path}:schema_required_not_list")
            required = []
        for field in required:
            if field not in value:
                errors.append(f"{path}.{field}:required_field_missing")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}:schema_properties_not_object")
            properties = {}
        if schema.get("additionalProperties") is False:
            for field in sorted(set(value) - set(properties)):
                errors.append(f"{path}.{field}:additional_property_forbidden")
        for field, child in value.items():
            if field in properties:
                errors.extend(
                    validate_json_contract(child, properties[field], f"{path}.{field}")
                )
    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path}:expected_array"]
        item_schema = schema.get("items", {})
        for index, child in enumerate(value):
            errors.extend(
                validate_json_contract(child, item_schema, f"{path}[{index}]")
            )
    elif expected_type == "string" and not isinstance(value, str):
        errors.append(f"{path}:expected_string")
    elif expected_type == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        errors.append(f"{path}:expected_integer")
    elif expected_type == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        errors.append(f"{path}:expected_number")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}:expected_boolean")
    elif expected_type == "null" and value is not None:
        errors.append(f"{path}:expected_null")
    elif isinstance(expected_type, list):
        alternatives = [
            validate_json_contract(value, {**schema, "type": kind}, path)
            for kind in expected_type
        ]
        if alternatives and all(option for option in alternatives):
            errors.append(f"{path}:no_allowed_type_matched")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}:enum_mismatch")
    return errors


def visible_checker_category(
    candidate_state: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    errors = candidate_state.get("errors", [])
    category = "NO_DETECTED_ERROR"
    state = candidate_state.get("state", {})
    expected = reference.get("state", {})
    if errors:
        category = "STATE_SCHEMA_INVALID"
    else:
        observed_entities = {
            row.get("entity_id"): row for row in state.get("entities", [])
        }
        expected_entities = {
            row.get("entity_id"): row for row in expected.get("entities", [])
        }
        invalidated = {
            entity_id
            for entity_id, row in expected_entities.items()
            if any(
                row.get("state", {}).get(flag) is True
                for flag in ("revoked", "stale", "blocked", "forbidden")
            )
        }
        unsafe_present = any(
            entity_id in invalidated
            and not any(
                row.get("state", {}).get(flag) is True
                for flag in ("revoked", "stale", "blocked", "forbidden")
            )
            for entity_id, row in observed_entities.items()
        )
        current_by_type: dict[str, set[str]] = {}
        for row in observed_entities.values():
            if row.get("state", {}).get("current") is True:
                key = str(row.get("entity_type"))
                current_by_type.setdefault(key, set()).add(str(row.get("entity_id")))
        expected_current_by_type: dict[str, set[str]] = {}
        for row in expected_entities.values():
            if row.get("state", {}).get("current") is True:
                key = str(row.get("entity_type"))
                expected_current_by_type.setdefault(key, set()).add(
                    str(row.get("entity_id"))
                )
        conflict = any(
            len(observed_ids) > len(expected_current_by_type.get(kind, set()))
            for kind, observed_ids in current_by_type.items()
        )
        expected_relations = {sha256_json(row) for row in expected.get("relations", [])}
        observed_relations = {sha256_json(row) for row in state.get("relations", [])}
        missing_entity = any(
            identifier not in observed_entities for identifier in expected_entities
        )
        path_permission_problem = any(
            row.get("entity_type") in {"path", "permission"}
            and observed_entities.get(identifier) != row
            for identifier, row in expected_entities.items()
        )
        if unsafe_present:
            category = "INVALIDATED_STATE_PRESENT"
        elif conflict:
            category = "CONFLICT_PRESENT"
        elif not expected_relations <= observed_relations:
            category = "CRITICAL_RELATION_MISSING"
        elif missing_entity:
            category = "REQUIRED_STATE_MISSING"
        elif path_permission_problem:
            category = "PATH_OR_PERMISSION_UNSATISFIED"
    result = {
        "checker_version": VISIBLE_CHECKER_VERSION,
        "category": category,
        "candidate_state_sha256": candidate_state.get("state_sha256"),
        "visible_reference_sha256": reference.get("reference_sha256"),
        "oracle_accessed": False,
        "identifier_disclosed": False,
        "path_disclosed": False,
        "permission_disclosed": False,
        "action_disclosed": False,
    }
    result["feedback_sha256"] = sha256_json(result)
    return result
