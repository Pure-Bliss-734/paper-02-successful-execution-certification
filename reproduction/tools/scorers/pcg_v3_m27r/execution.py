from __future__ import annotations

from typing import Any

from pcg_v3_m11.schemas import ACTION_SCHEMA_VERSION, ENV_STATE_SCHEMA_VERSION

from .common import digest

from .spec import EXECUTION_BRIDGE_VERSION


def _object_row(
    entity_id: str, entity_type: str, state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "object_id": entity_id,
        "object_type": entity_type,
        "active": bool(state.get("active", True)),
        "revoked": bool(state.get("revoked", False)),
        "stale": bool(state.get("stale", False)),
        "blocked": bool(state.get("blocked", False)),
        "protected": bool(state.get("protected", False)),
        "forbidden": bool(state.get("forbidden", False)),
        "current": bool(state.get("current", False)),
        "current_interface": entity_type == "interface"
        and bool(state.get("current", False)),
        "selected": False,
        "applied": False,
        "path_id": None,
        "superseded_by": None,
        "conflicts_with": None,
        "metadata": {"source": "typed_state"},
    }


def build_runtime_state(
    candidate: dict[str, Any], final_state: dict[str, Any]
) -> dict[str, Any]:
    entity_types = {
        row["entity_id"]: row["entity_type"] for row in candidate["entities"]
    }
    objects = {
        entity_id: _object_row(entity_id, entity_types[entity_id], state)
        for entity_id, state in sorted(final_state["entities"].items())
    }
    paths = {}
    for entity_id, state in sorted(final_state["entities"].items()):
        if entity_types[entity_id] != "path":
            continue
        allowed = bool(state.get("allowed", False))
        paths[entity_id] = {
            "path_id": entity_id,
            "allowed": allowed,
            "forbidden": not allowed,
            "blocked": bool(state.get("blocked", False)),
            "revoked": bool(state.get("revoked", False)),
            "metadata": {"source": "typed_state"},
        }

    replacements = []
    dependencies = []
    permissions: dict[str, Any] = {}
    authorizations = []
    for kind, source, target in final_state["edges"]:
        if kind == "replacement":
            replacements.append({"source": source, "target": target, "active": True})
        elif kind == "dependency_addition":
            dependencies.append({"source": source, "target": target, "active": True})
        elif kind == "action_precondition_establishment":
            # The typed relation is operation -> target. The transaction
            # dependency is target -> operation; this mapping is frozen in the
            # execution interface and uses no expected action or terminal data.
            dependencies.append({"source": target, "target": source, "active": True})
        elif kind == "permission_grant":
            permissions.setdefault(source, {}).setdefault("execute", {})[target] = {
                "revoked": False
            }
            authorizations.append(
                {
                    "actor": source,
                    "operation": "execute",
                    "target": target,
                    "decision": "allow",
                    "active": True,
                }
            )
        elif kind == "path_addition":
            # The action remains pending: the path relation authorizes a
            # migration, while the object has not yet moved.
            objects[source]["path_id"] = None

    return {
        "environment_schema_version": ENV_STATE_SCHEMA_VERSION,
        "environment_version": "pcg_v3_m27r.environment.v1",
        "transaction_version": "pcg_v3_m11.atomic_transaction.v1",
        "immutable_metadata": {
            "candidate_id": candidate["candidate_id"],
            "candidate_schema_version": candidate["schema_version"],
            "execution_bridge_version": EXECUTION_BRIDGE_VERSION,
        },
        "mutable_operational_state": {
            "objects": objects,
            "paths": paths,
            "permissions": permissions,
            "replacement_relationships": replacements,
            "dependency_relationships": dependencies,
            "operation_authorizations": authorizations,
            "exclusion_records": [
                {"entity_id": source, "reason": reason}
                for source, reason in final_state["exclusions"]
            ],
        },
        "historical_events": list(final_state["history"]),
        "action_history": [],
    }


def action_envelope(
    typed_action: dict[str, Any],
    runtime_state: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    action_type = typed_action["action_type"]
    primary = typed_action["primary_target"]
    primary_targets = primary if isinstance(primary, list) else [primary]
    dependency_edges = runtime_state["mutable_operational_state"][
        "dependency_relationships"
    ]
    declared = sorted(
        {
            edge["target"]
            for edge in dependency_edges
            if edge["active"] and edge["source"] in primary_targets
        }
    )
    return {
        "action_type": action_type,
        "primary_target": primary,
        "optional_secondary_target": None,
        "optional_path": typed_action.get("path"),
        "optional_interface": typed_action.get("interface"),
        "optional_replacement_target": typed_action.get("replacement_target"),
        "optional_actor_or_role": typed_action.get("actor_or_role"),
        "optional_decision": typed_action.get("decision"),
        "declared_dependencies": declared,
        "action_parameters": {
            "operation": (
                "execute" if action_type == "allow_or_deny_operation" else None
            )
        },
        "transaction_id": f"TX-{digest([candidate_id, typed_action])[:20]}",
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }


def _terminal_invariants(
    expected: dict[str, Any], action: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = [
        {"path": "action_history", "equals": expected["action_history"]},
    ]
    objects = expected["mutable_operational_state"]["objects"]
    if action["action_type"] == "apply_current_bundle":
        rows.extend(
            {
                "path": f"mutable_operational_state.objects.{target}.applied",
                "equals": True,
            }
            for target in action["primary_target"]
        )
    elif action["action_type"] == "select_current_interface":
        target = action["optional_interface"] or action["primary_target"]
        rows.append(
            {
                "path": f"mutable_operational_state.objects.{target}.selected",
                "equals": True,
            }
        )
    elif action["action_type"] == "migrate_to_allowed_path":
        rows.append(
            {
                "path": (
                    "mutable_operational_state.objects."
                    f"{action['primary_target']}.path_id"
                ),
                "equals": action["optional_path"],
            }
        )
    elif action["action_type"] == "replace_current_object":
        rows.extend(
            [
                {
                    "path": (
                        "mutable_operational_state.objects."
                        f"{action['primary_target']}.current"
                    ),
                    "equals": False,
                },
                {
                    "path": (
                        "mutable_operational_state.objects."
                        f"{action['optional_replacement_target']}.current"
                    ),
                    "equals": True,
                },
            ]
        )
    return rows
