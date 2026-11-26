from __future__ import annotations

import json

from typing import Any

from pcg_v3.hashing import sha256_json

CARRIED_STATE_NORMALIZER_VERSION = "pcg_v3_m30.carried_state_normalizer.v2"


def normalize_carried_state(value: Any, representation: str) -> dict[str, Any]:
    errors: list[str] = []
    if representation == "structured":
        state = value if isinstance(value, dict) else {}
        if not isinstance(value, dict):
            errors.append("CARRIED_STATE_NOT_OBJECT")
    elif representation == "natural":
        state = {"entities": [], "relations": [], "exclusions": []}
        if not isinstance(value, str):
            errors.append("CARRIED_STATE_NOT_STRING")
        else:
            for line in value.splitlines():
                if not line.strip():
                    continue
                prefix, separator, payload = line.partition(" ")
                section = {
                    "ENTITY": "entities",
                    "RELATION": "relations",
                    "EXCLUSION": "exclusions",
                }.get(prefix)
                if not separator or section is None:
                    errors.append("NATURAL_STATE_LINE_INVALID")
                    continue
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    errors.append("NATURAL_STATE_PAYLOAD_INVALID")
                    continue
                if not isinstance(row, dict):
                    errors.append("NATURAL_STATE_PAYLOAD_NOT_OBJECT")
                    continue
                state[section].append(row)
    else:
        raise ValueError("unsupported state representation")
    if set(state) != {"entities", "relations", "exclusions"}:
        errors.append("CARRIED_STATE_TOP_LEVEL_SCHEMA_INVALID")
    for key in ("entities", "relations", "exclusions"):
        if not isinstance(state.get(key), list):
            errors.append(f"CARRIED_STATE_{key.upper()}_NOT_LIST")
            state[key] = []
    row_contracts = {
        "entities": {"entity_id", "entity_type", "state"},
        "relations": {
            "relation_id",
            "source_id",
            "target_id",
            "relation_type",
            "direction",
        },
        "exclusions": {"entity_id", "reason"},
    }
    identity_fields = {
        "entities": "entity_id",
        "relations": "relation_id",
        "exclusions": None,
    }
    for section, required in row_contracts.items():
        identifiers: list[str] = []
        for index, row in enumerate(state[section]):
            if not isinstance(row, dict):
                errors.append(f"{section.upper()}_ROW_{index}_NOT_OBJECT")
                continue
            if set(row) != required:
                errors.append(f"{section.upper()}_ROW_{index}_SCHEMA_INVALID")
            string_fields = {
                "entities": ("entity_id", "entity_type"),
                "relations": (
                    "relation_id",
                    "source_id",
                    "target_id",
                    "relation_type",
                    "direction",
                ),
                "exclusions": ("entity_id", "reason"),
            }[section]
            for field in string_fields:
                if not isinstance(row.get(field), str) or not row.get(field):
                    errors.append(
                        f"{section.upper()}_ROW_{index}_{field.upper()}_INVALID"
                    )
            if section == "entities" and not isinstance(row.get("state"), dict):
                errors.append(f"ENTITIES_ROW_{index}_STATE_INVALID")
            identity = identity_fields[section]
            if identity is not None and isinstance(row.get(identity), str):
                identifiers.append(row[identity])
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{section.upper()}_IDENTIFIER_DUPLICATE")
    canonical = {
        "entities": sorted(state["entities"], key=sha256_json),
        "relations": sorted(state["relations"], key=sha256_json),
        "exclusions": sorted(state["exclusions"], key=sha256_json),
    }
    return {
        "state": canonical,
        "representation": representation,
        "errors": errors,
        "parse_success": not errors,
        "state_sha256": sha256_json(canonical),
    }


def normalize_carried_state_v2(value: Any, representation: str) -> dict[str, Any]:
    """Canonicalize presentation metadata before strict carried-state checks.

    The visible surface intentionally contains richer ENTITY/RELATION records
    than the carried-state contract.  Models sometimes copy those records even
    when the prompt asks for the smaller state schema.  V2 removes only
    registered presentation fields and records every removal in an audit trace;
    it never invents identifiers, relations, or state values.  Missing required
    state fields still fail closed through the original strict normalizer.
    """
    events: list[dict[str, Any]] = []

    def entity_row(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        if "state" in row:
            state = row["state"]
        elif "initial_state" in row:
            state = row["initial_state"]
            events.append(
                {
                    "kind": "entity_initial_state_promoted",
                    "entity_id": row.get("entity_id"),
                }
            )
        else:
            state = None
        if state is None:
            return row
        removed = sorted(
            set(row) - {"entity_id", "entity_type", "state", "initial_state"}
        )
        if removed:
            events.append(
                {
                    "kind": "entity_presentation_fields_removed",
                    "entity_id": row.get("entity_id"),
                    "fields": removed,
                }
            )
        return {
            "entity_id": row.get("entity_id"),
            "entity_type": row.get("entity_type"),
            "state": state,
        }

    def relation_row(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        allowed = {
            "relation_id",
            "source_id",
            "target_id",
            "relation_type",
            "direction",
        }
        removed = sorted(set(row) - allowed)
        if removed:
            events.append(
                {
                    "kind": "relation_presentation_fields_removed",
                    "relation_id": row.get("relation_id"),
                    "fields": removed,
                }
            )
        return {key: row.get(key) for key in sorted(allowed)}

    def exclusion_row(row: Any) -> Any:
        if not isinstance(row, dict):
            return row
        reason = row.get("reason", row.get("exclusion_reason"))
        if "reason" not in row and "exclusion_reason" in row:
            events.append(
                {
                    "kind": "exclusion_reason_alias_normalized",
                    "entity_id": row.get("entity_id"),
                }
            )
        removed = sorted(set(row) - {"entity_id", "reason", "exclusion_reason"})
        if removed:
            events.append(
                {
                    "kind": "exclusion_presentation_fields_removed",
                    "entity_id": row.get("entity_id"),
                    "fields": removed,
                }
            )
        return {"entity_id": row.get("entity_id"), "reason": reason}

    if representation == "natural" and isinstance(value, str):
        lines: list[str] = []
        for line in value.splitlines():
            prefix, separator, payload = line.partition(" ")
            if separator:
                try:
                    row = json.loads(payload)
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                if prefix == "ENTITY":
                    row = entity_row(row)
                elif prefix == "RELATION":
                    row = relation_row(row)
                elif prefix == "EXCLUSION":
                    row = exclusion_row(row)
                line = (
                    prefix
                    + " "
                    + json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                )
            lines.append(line)
        value = "\n".join(lines)
    elif representation == "structured" and isinstance(value, dict):
        value = dict(value)
        value["entities"] = [entity_row(row) for row in value.get("entities", [])]
        value["relations"] = [relation_row(row) for row in value.get("relations", [])]
        value["exclusions"] = [
            exclusion_row(row) for row in value.get("exclusions", [])
        ]

    normalized = normalize_carried_state(value, representation)
    normalized["normalizer_version"] = CARRIED_STATE_NORMALIZER_VERSION
    normalized["normalization_events"] = events
    normalized["normalization_applied"] = bool(events)
    return normalized
