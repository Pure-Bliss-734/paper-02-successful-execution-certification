from __future__ import annotations
from copy import deepcopy
from typing import Any
from pcg_v3.hashing import sha256_json
from .ablation_contract import ACTION_SCHEMA_VERSION


def _identifier(seed: str, role: str, prefix: str = "E") -> str:
    return prefix + sha256_json([seed, role])[:14].upper()


def _unsigned_event(event: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_record_sha256"}


def _relation_for(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_id": _identifier(event["event_id"], "relation", "R"),
        "source_id": event["subject_id"],
        "target_id": event["target_id"],
        "relation_type": event["canonical_event_type"],
        "direction": event["relation_direction"],
    }


def apply_events_a(
    state: dict[str, Any], events: list[dict[str, Any]]
) -> dict[str, Any]:
    result = deepcopy(state)
    entities = {row["entity_id"]: row for row in result["entities"]}
    exclusions = {row["entity_id"]: row for row in result["exclusions"]}
    for raw in events:
        event = _unsigned_event(raw)
        kind, subject_id = event["canonical_event_type"], event["subject_id"]
        row = entities[subject_id]
        values = row["state"]
        if kind == "activation":
            values["active"] = True
        elif kind == "revocation":
            values.update(active=False, revoked=True)
        elif kind == "reactivation":
            values.update(active=True, revoked=False)
            exclusions.pop(subject_id, None)
        elif kind == "protect":
            values["protected"] = True
        elif kind == "unprotect":
            values["protected"] = False
        elif kind == "overwrite":
            values["value"] = event["value"]
        elif kind == "current_interface_designation":
            for candidate in entities.values():
                if candidate["entity_type"] == "interface":
                    candidate["state"]["current"] = candidate["entity_id"] == subject_id
        elif kind == "forbid":
            values["allowed"] = False
        elif kind == "path_addition":
            entities[event["target_id"]]["state"]["allowed"] = True
            result["relations"].append(_relation_for(event))
        elif kind == "replacement":
            values.update(active=False, current=False, superseded_by=event["target_id"])
            target = entities[event["target_id"]]["state"]
            target.update(active=True, current=True)
            result["relations"].append(_relation_for(event))
        elif kind in {"permission_grant", "dependency_addition"}:
            result["relations"].append(_relation_for(event))
        else:
            raise ValueError(kind)
    result["relations"] = sorted(
        {row["relation_id"]: row for row in result["relations"]}.values(),
        key=lambda row: row["relation_id"],
    )
    result["entities"] = sorted(entities.values(), key=lambda row: row["entity_id"])
    result["exclusions"] = sorted(exclusions.values(), key=lambda row: row["entity_id"])
    return result


def _safe(entity: dict[str, Any]) -> bool:
    state = entity["state"]
    return bool(state["active"]) and not any(
        state[key] for key in ("revoked", "stale", "blocked", "protected")
    )


def derive_action_a(state: dict[str, Any], action_type: str) -> dict[str, Any] | None:
    entities = {row["entity_id"]: row for row in state["entities"]}
    relations = state["relations"]
    action: dict[str, Any] | None = None
    if action_type == "apply_current_bundle":
        targets = sorted(
            key
            for key, row in entities.items()
            if row["entity_type"] in {"object", "configuration", "bundle"}
            and row["state"]["current"]
            and _safe(row)
        )
        if targets:
            action = {"action_type": action_type, "primary_target": targets}
    elif action_type == "select_current_interface":
        targets = sorted(
            key
            for key, row in entities.items()
            if row["entity_type"] == "interface"
            and row["state"]["current"]
            and _safe(row)
        )
        if len(targets) == 1:
            action = {
                "action_type": action_type,
                "primary_target": targets[0],
                "interface": targets[0],
            }
    elif action_type == "migrate_to_allowed_path":
        matches = [
            (row["source_id"], row["target_id"])
            for row in relations
            if row["relation_type"] == "path_addition"
            and _safe(entities[row["source_id"]])
            and entities[row["target_id"]]["state"]["allowed"]
        ]
        if len(matches) == 1:
            action = {
                "action_type": action_type,
                "primary_target": matches[0][0],
                "path": matches[0][1],
            }
    elif action_type == "replace_current_object":
        matches = [
            (row["source_id"], row["target_id"])
            for row in relations
            if row["relation_type"] == "replacement"
            and _safe(entities[row["target_id"]])
        ]
        if len(matches) == 1:
            action = {
                "action_type": action_type,
                "primary_target": matches[0][0],
                "replacement_target": matches[0][1],
            }
    elif action_type == "allow_or_deny_operation":
        matches = [
            (row["source_id"], row["target_id"])
            for row in relations
            if row["relation_type"] == "permission_grant"
            and _safe(entities[row["source_id"]])
        ]
        if len(matches) == 1:
            action = {
                "action_type": action_type,
                "primary_target": matches[0][1],
                "actor_or_role": matches[0][0],
                "operation": "execute",
                "decision": "allow",
            }
    if action is None:
        return None
    primaries = (
        action["primary_target"]
        if isinstance(action["primary_target"], list)
        else [action["primary_target"]]
    )
    action["declared_dependencies"] = sorted(
        {
            row["target_id"]
            for row in relations
            if row["relation_type"] == "dependency_addition"
            and row["source_id"] in primaries
        }
    )
    action["schema_version"] = ACTION_SCHEMA_VERSION
    return action
