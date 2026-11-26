from __future__ import annotations
import json
from copy import deepcopy
from typing import Any
from .events import apply_events_a, derive_action_a


BOOL_FIELDS = (
    "active",
    "allowed",
    "blocked",
    "current",
    "protected",
    "revoked",
    "stale",
)


def state_text(state: dict[str, Any]) -> str:
    lines = []
    for entity in sorted(state.get("entities", []), key=lambda row: row["entity_id"]):
        lines.append(
            "ENTITY "
            + json.dumps(
                entity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    for relation in sorted(
        state.get("relations", []),
        key=lambda row: (
            str(row.get("relation_type")),
            str(row.get("source_id")),
            str(row.get("target_id")),
            str(row.get("direction")),
        ),
    ):
        lines.append(
            "RELATION "
            + json.dumps(
                relation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    for exclusion in sorted(
        state.get("exclusions", []),
        key=lambda row: (str(row.get("entity_id")), str(row.get("reason"))),
    ):
        lines.append(
            "EXCLUSION "
            + json.dumps(
                exclusion,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return "\n".join(lines)


def action_ids(action: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    primary = action.get("primary_target")
    if isinstance(primary, str):
        result.add(primary)
    elif isinstance(primary, list):
        result.update(item for item in primary if isinstance(item, str))
    for field in ("interface", "path", "replacement_target", "actor_or_role"):
        value = action.get(field)
        if isinstance(value, str):
            result.add(value)
    result.update(
        item
        for item in action.get("declared_dependencies", [])
        if isinstance(item, str)
    )
    return result


def symbolic_action(
    state: dict[str, Any], task: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        after = apply_events_a(state, task["future_events"])
        return derive_action_a(after, task["slot_contract"]["action_type"]), None
    except (KeyError, TypeError, ValueError) as error:
        return None, type(error).__name__


def delete_necessary(
    base: dict[str, Any], task: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = action_ids(expected)
    future_ids = {
        value
        for event in task["future_events"]
        for value in (event.get("subject_id"), event.get("target_id"))
        if isinstance(value, str)
    }
    candidates = sorted((required - future_ids) or required)
    for entity_id in candidates:
        candidate = deepcopy(base)
        before = len(candidate["entities"])
        candidate["entities"] = [
            row for row in candidate["entities"] if row["entity_id"] != entity_id
        ]
        if len(candidate["entities"]) == before:
            continue
        candidate["relations"] = [
            row
            for row in candidate["relations"]
            if entity_id not in {row.get("source_id"), row.get("target_id")}
        ]
        candidate["exclusions"] = [
            row for row in candidate["exclusions"] if row.get("entity_id") != entity_id
        ]
        action, error = symbolic_action(candidate, task)
        if action != expected:
            return candidate, {
                "intervention": "delete_required_entity",
                "entity_id": entity_id,
                "symbolic_action": action,
                "symbolic_error": error,
                "future_event_mentions_entity": entity_id in future_ids,
            }
    raise RuntimeError(f"no necessary-state deletion found: {task['task_id']}")


def wrong_state(
    base: dict[str, Any], task: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = action_ids(expected)
    for entity_id in sorted(required):
        for field in BOOL_FIELDS:
            candidate = deepcopy(base)
            row = next(
                (
                    item
                    for item in candidate["entities"]
                    if item["entity_id"] == entity_id
                ),
                None,
            )
            if row is None or not isinstance(row.get("state", {}).get(field), bool):
                continue
            old = row["state"][field]
            row["state"][field] = not old
            action, error = symbolic_action(candidate, task)
            if error is None and action != expected:
                return candidate, {
                    "intervention": "flip_required_boolean_fact",
                    "entity_id": entity_id,
                    "field": field,
                    "before": old,
                    "after": not old,
                    "symbolic_action": action,
                    "symbolic_error": None,
                }
    raise RuntimeError(f"no counterfactual wrong state found: {task['task_id']}")


def unrelated_state(
    base: dict[str, Any], task: dict[str, Any], expected: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    relevant = action_ids(expected)
    relevant.update(
        value
        for event in task["future_events"]
        for value in (event.get("subject_id"), event.get("target_id"))
        if isinstance(value, str)
    )
    for entity in sorted(base["entities"], key=lambda row: row["entity_id"]):
        entity_id = entity["entity_id"]
        if entity_id in relevant:
            continue
        candidate = deepcopy(base)
        row = next(
            item for item in candidate["entities"] if item["entity_id"] == entity_id
        )
        before = row["state"].get("value")
        row["state"]["value"] = "unrelated_control_perturbation"
        action, error = symbolic_action(candidate, task)
        if error is None and action == expected:
            return candidate, {
                "intervention": "change_irrelevant_value",
                "entity_id": entity_id,
                "field": "value",
                "before": before,
                "after": row["state"]["value"],
                "symbolic_action": action,
                "symbolic_error": None,
            }
    raise RuntimeError(f"no unrelated-state control found: {task['task_id']}")
