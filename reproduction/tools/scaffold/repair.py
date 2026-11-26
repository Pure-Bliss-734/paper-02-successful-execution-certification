from __future__ import annotations
import copy
from typing import Any
from collections import Counter
from pcg_v3_supplement.independent_state_matcher import STATE_FIELDS, match_state


RELATION_FIELDS = ("relation_type", "source_id", "target_id", "direction")


def _entities(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["entity_id"]: row
        for row in state.get("entities", [])
        if isinstance(row, dict) and isinstance(row.get("entity_id"), str)
    }


def relation_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return tuple(str(row.get(field)) for field in RELATION_FIELDS)


def _relation_counter(state: dict[str, Any]) -> Counter[tuple[str, str, str, str]]:
    return Counter(
        relation_key(row) for row in state.get("relations", []) if isinstance(row, dict)
    )


def enumerate_defective_bundles(
    observed_state: dict[str, Any],
    reference_state: dict[str, Any],
    expected_action: dict[str, Any],
    *,
    parse_success: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return deterministic R-defective bundles without reading downstream outcomes.

    A critical relation incident to two required entities is owned by the
    lexicographically first endpoint.  This avoids double counting while
    retaining the paper's incident-relation definition.
    """

    score = match_state(
        observed_state,
        reference_state,
        expected_action,
        expected_action,
        parse_success=parse_success,
    )
    required_ids = list(score["required_entity_ids"])
    reference_entities = _entities(reference_state)
    observed_entities = _entities(observed_state)
    bundles = {
        entity_id: {
            "bundle_id": entity_id,
            "required_field_differences": [],
            "critical_relation_differences": [],
            "exclusion_differences": [],
        }
        for entity_id in required_ids
    }

    for entity_id in required_ids:
        expected = reference_entities[entity_id]
        observed = observed_entities.get(entity_id)
        if observed is None:
            bundles[entity_id]["required_field_differences"].append(
                {
                    "field": "__entity__",
                    "before": None,
                    "after": {
                        "entity_id": entity_id,
                        "entity_type": expected.get("entity_type"),
                        "state": {
                            field: expected.get("state", {}).get(field)
                            for field in STATE_FIELDS
                        },
                    },
                }
            )
            continue
        if observed.get("entity_type") != expected.get("entity_type"):
            bundles[entity_id]["required_field_differences"].append(
                {
                    "field": "entity_type",
                    "before": observed.get("entity_type"),
                    "after": expected.get("entity_type"),
                }
            )
        for field in STATE_FIELDS:
            before = observed.get("state", {}).get(field)
            after = expected.get("state", {}).get(field)
            if before != after:
                bundles[entity_id]["required_field_differences"].append(
                    {"field": field, "before": before, "after": after}
                )

    observed_relations = set(_relation_counter(observed_state))
    for relation in sorted(tuple(row) for row in score["critical_relations"]):
        if relation in observed_relations:
            continue
        owners = sorted({relation[1], relation[2]} & set(required_ids))
        if not owners:
            raise AssertionError("critical relation lacks a required endpoint")
        bundles[owners[0]]["critical_relation_differences"].append(
            {"before": None, "after": list(relation)}
        )

    for exclusion in observed_state.get("exclusions", []):
        entity_id = exclusion.get("entity_id") if isinstance(exclusion, dict) else None
        if entity_id in bundles:
            bundles[entity_id]["exclusion_differences"].append(
                {"before": copy.deepcopy(exclusion), "after": None}
            )

    defective = {
        entity_id: bundle
        for entity_id, bundle in bundles.items()
        if any(
            bundle[field]
            for field in (
                "required_field_differences",
                "critical_relation_differences",
                "exclusion_differences",
            )
        )
    }
    return defective, score


def apply_bundle_repair(
    baseline: dict[str, Any],
    reference: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    repaired = copy.deepcopy(baseline)
    bundle_id = bundle["bundle_id"]
    expected_entities = _entities(reference)
    observed_entities = _entities(repaired)
    field_differences = bundle["required_field_differences"]

    if any(item["field"] == "__entity__" for item in field_differences):
        if bundle_id in observed_entities:
            raise AssertionError("missing-entity repair targets an existing entity")
        repaired["entities"].append(
            copy.deepcopy(
                next(
                    item["after"]
                    for item in field_differences
                    if item["field"] == "__entity__"
                )
            )
        )
    else:
        row = observed_entities[bundle_id]
        for item in field_differences:
            if item["field"] == "entity_type":
                row["entity_type"] = item["after"]
            else:
                row.setdefault("state", {})[item["field"]] = item["after"]

    reference_relations = {
        relation_key(row): row for row in reference.get("relations", [])
    }
    for item in bundle["critical_relation_differences"]:
        wanted_key = tuple(item["after"])
        wanted = reference_relations[wanted_key]
        same_id = next(
            (
                row
                for row in repaired.get("relations", [])
                if wanted.get("relation_id") is not None
                and row.get("relation_id") == wanted.get("relation_id")
            ),
            None,
        )
        if same_id is None:
            repaired.setdefault("relations", []).append(copy.deepcopy(wanted))
        else:
            for field in RELATION_FIELDS:
                same_id[field] = wanted[field]

    if bundle["exclusion_differences"]:
        repaired["exclusions"] = [
            row
            for row in repaired.get("exclusions", [])
            if row.get("entity_id") != bundle_id
        ]
    return repaired
