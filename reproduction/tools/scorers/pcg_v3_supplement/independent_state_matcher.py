from __future__ import annotations

from collections import Counter

from typing import Any

MATCHER_VERSION = "pcg_v3.independent_state_matcher.v1"

STATE_FIELDS = (
    "active",
    "allowed",
    "blocked",
    "current",
    "protected",
    "revoked",
    "stale",
    "value",
)

ACTION_RELATION_TYPES = {
    "apply_current_bundle": {
        "dependency_addition",
        "action_precondition_establishment",
    },
    "select_current_interface": {
        "dependency_addition",
        "action_precondition_establishment",
    },
    "migrate_to_allowed_path": {
        "path_addition",
        "dependency_addition",
        "action_precondition_establishment",
    },
    "replace_current_object": {
        "replacement",
        "dependency_addition",
        "action_precondition_establishment",
    },
    "allow_or_deny_operation": {
        "permission_grant",
        "action_precondition_establishment",
    },
}


def _scalar(value: Any) -> tuple[str, str]:
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("bool", "true" if value else "false")
    if isinstance(value, (int, float)):
        return ("number", repr(value))
    return ("string", str(value))


def _entity_rows(state: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(state, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in state.get("entities", []):
        if not isinstance(row, dict):
            continue
        entity_id = row.get("entity_id")
        if isinstance(entity_id, str) and entity_id not in rows:
            rows[entity_id] = row
    return rows


def _entity_facts(state: Any) -> Counter[tuple[str, str, str, str]]:
    facts: Counter[tuple[str, str, str, str]] = Counter()
    for entity_id, row in _entity_rows(state).items():
        kind, value = _scalar(row.get("entity_type"))
        facts[(entity_id, "entity_type", kind, value)] += 1
        status = row.get("state")
        status = status if isinstance(status, dict) else {}
        for field in STATE_FIELDS:
            kind, value = _scalar(status.get(field))
            facts[(entity_id, field, kind, value)] += 1
    return facts


def _relations(state: Any) -> Counter[tuple[str, str, str, str]]:
    result: Counter[tuple[str, str, str, str]] = Counter()
    if not isinstance(state, dict):
        return result
    for row in state.get("relations", []):
        if not isinstance(row, dict):
            continue
        result[
            (
                str(row.get("relation_type")),
                str(row.get("source_id")),
                str(row.get("target_id")),
                str(row.get("direction")),
            )
        ] += 1
    return result


def _exclusions(state: Any) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if not isinstance(state, dict):
        return result
    for row in state.get("exclusions", []):
        if isinstance(row, dict):
            result.add((str(row.get("entity_id")), str(row.get("reason"))))
    return result


def _prf(observed: Counter[Any], expected: Counter[Any]) -> dict[str, Any]:
    true_positive = sum((observed & expected).values())
    observed_n = sum(observed.values())
    expected_n = sum(expected.values())
    precision = (
        true_positive / observed_n if observed_n else (1.0 if expected_n == 0 else 0.0)
    )
    recall = (
        true_positive / expected_n if expected_n else (1.0 if observed_n == 0 else 0.0)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "observed": observed_n,
        "expected": expected_n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _required_entity_ids(
    expected_action: dict[str, Any], structured_action: dict[str, Any]
) -> set[str]:
    values: set[str] = set()
    primary = expected_action.get("primary_target")
    if isinstance(primary, str):
        values.add(primary)
    elif isinstance(primary, list):
        values.update(value for value in primary if isinstance(value, str))
    for field in ("interface", "path", "replacement_target", "actor_or_role"):
        value = expected_action.get(field)
        if isinstance(value, str):
            values.add(value)
    dependencies = structured_action.get("declared_dependencies", [])
    if isinstance(dependencies, list):
        values.update(value for value in dependencies if isinstance(value, str))
    return values


def _required_relation_set(
    reference_state: dict[str, Any],
    expected_action: dict[str, Any],
    required_ids: set[str],
) -> set[tuple[str, str, str, str]]:
    action_type = str(expected_action.get("action_type"))
    allowed = ACTION_RELATION_TYPES.get(action_type, set())
    result: set[tuple[str, str, str, str]] = set()
    for relation, count in _relations(reference_state).items():
        relation_type, source_id, target_id, _ = relation
        if (
            count
            and relation_type in allowed
            and (source_id in required_ids or target_id in required_ids)
        ):
            result.add(relation)
    return result


def match_state(
    observed_state: Any,
    reference_state: dict[str, Any],
    expected_action: dict[str, Any],
    structured_action: dict[str, Any],
    *,
    parse_success: bool,
) -> dict[str, Any]:
    expected_facts = _entity_facts(reference_state)
    observed_facts = _entity_facts(observed_state)
    expected_relations = _relations(reference_state)
    observed_relations = _relations(observed_state)
    expected_exclusions = _exclusions(reference_state)
    observed_exclusions = _exclusions(observed_state)

    required = _required_entity_ids(expected_action, structured_action)
    reference_entities = _entity_rows(reference_state)
    observed_entities = _entity_rows(observed_state)
    required &= set(reference_entities)
    exact_required: set[str] = set()
    for entity_id in required:
        expected_one = {"entities": [reference_entities[entity_id]]}
        observed_row = observed_entities.get(entity_id)
        if observed_row is None:
            continue
        if _entity_facts({"entities": [observed_row]}) == _entity_facts(expected_one):
            exact_required.add(entity_id)

    critical = _required_relation_set(reference_state, expected_action, required)
    observed_relation_set = set(observed_relations)
    covered = critical & observed_relation_set
    excluded_ids = {entity_id for entity_id, _ in observed_exclusions}
    prompt_exact = bool(
        parse_success
        and observed_facts == expected_facts
        and observed_relations == expected_relations
        and observed_exclusions == expected_exclusions
    )
    action_support_exact = bool(
        parse_success
        and required == exact_required
        and critical == covered
        and not (required & excluded_ids)
    )
    return {
        "matcher_version": MATCHER_VERSION,
        "parse_success": bool(parse_success),
        "P": prompt_exact,
        "R": action_support_exact,
        "state_fact_metrics": _prf(observed_facts, expected_facts),
        "semantic_relation_metrics": _prf(observed_relations, expected_relations),
        "exclusion_semantic_set_metrics": _prf(
            Counter(expected_exclusions & observed_exclusions)
            + Counter(observed_exclusions - expected_exclusions),
            Counter(expected_exclusions),
        ),
        "required_entity_ids": sorted(required),
        "exact_required_entity_ids": sorted(exact_required),
        "critical_relations": sorted(critical),
        "covered_critical_relations": sorted(covered),
        "required_entity_excluded": bool(required & excluded_ids),
    }
