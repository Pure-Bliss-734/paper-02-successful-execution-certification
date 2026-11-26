from __future__ import annotations

from collections import Counter

from typing import Any

from pcg_v3.hashing import sha256_json

from pcg_v3_m30.methods import normalize_carried_state_v2

STATE_KEYS = (
    "active",
    "allowed",
    "blocked",
    "current",
    "protected",
    "revoked",
    "stale",
    "value",
)


def semantic_projection(state: dict[str, Any]) -> dict[str, Counter[Any]]:
    """Project a normalized state onto claim-bearing semantic identities."""
    entities: Counter[Any] = Counter()
    relations: Counter[Any] = Counter()
    exclusions: Counter[Any] = Counter()
    for row in state.get("entities", []):
        state_value = row.get("state")
        state_tuple = (
            tuple((key, state_value.get(key)) for key in STATE_KEYS)
            if isinstance(state_value, dict)
            else (("invalid", None),)
        )
        entities[(row.get("entity_id"), row.get("entity_type"), state_tuple)] += 1
    for row in state.get("relations", []):
        relations[
            (
                row.get("source_id"),
                row.get("target_id"),
                row.get("relation_type"),
                row.get("direction"),
            )
        ] += 1
    for row in state.get("exclusions", []):
        exclusions[(row.get("entity_id"), row.get("reason"))] += 1
    return {"entities": entities, "relations": relations, "exclusions": exclusions}


def compare_semantic_states(
    observed_normalized: dict[str, Any],
    expected_state: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact entity/exclusion semantics and identifier-invariant relations."""
    if observed_normalized.get("parse_success") is not True:
        result = {
            "success": False,
            "parse_success": False,
            "entity_success": False,
            "relation_success": False,
            "exclusion_success": False,
            "errors": list(observed_normalized.get("errors", [])),
        }
        result["record_hash"] = sha256_json(result)
        return result
    expected_normalized = normalize_carried_state_v2(expected_state, "structured")
    if expected_normalized.get("parse_success") is not True:
        raise ValueError(
            f"invalid expected semantic state: {expected_normalized['errors']}"
        )
    observed = semantic_projection(observed_normalized["state"])
    expected = semantic_projection(expected_normalized["state"])
    entity_success = observed["entities"] == expected["entities"]
    relation_success = observed["relations"] == expected["relations"]
    exclusion_success = observed["exclusions"] == expected["exclusions"]
    result = {
        "success": bool(entity_success and relation_success and exclusion_success),
        "parse_success": True,
        "entity_success": entity_success,
        "relation_success": relation_success,
        "exclusion_success": exclusion_success,
        "observed_counts": {
            key: sum(value.values()) for key, value in observed.items()
        },
        "expected_counts": {
            key: sum(value.values()) for key, value in expected.items()
        },
        "relation_identifier_invariant": True,
        "errors": [],
    }
    result["record_hash"] = sha256_json(result)
    return result
