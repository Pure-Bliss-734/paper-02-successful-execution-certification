from __future__ import annotations
import json
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

PROMPT_FIELDS = (
    "active",
    "allowed",
    "blocked",
    "current",
    "protected",
    "revoked",
    "stale",
    "value",
)


def relation_tuple(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("relation_type")),
        str(row.get("source_id")),
        str(row.get("target_id")),
        str(row.get("direction")),
    )


def exclusion_tuple(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("entity_id")), str(row.get("reason")))


def relation_set(state: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    return {
        relation_tuple(row)
        for row in state.get("relations", [])
        if isinstance(row, dict)
    }


def relation_set_independent(state: dict[str, Any]) -> frozenset[str]:
    """Independent serialization-based implementation of relation-set semantics."""
    rows = []
    for row in state.get("relations", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            json.dumps(
                {
                    "direction": str(row.get("direction")),
                    "relation_type": str(row.get("relation_type")),
                    "source_id": str(row.get("source_id")),
                    "target_id": str(row.get("target_id")),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return frozenset(rows)


def exclusion_set(state: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        exclusion_tuple(row)
        for row in state.get("exclusions", [])
        if isinstance(row, dict)
    }


def entity_fact_set(state: dict[str, Any]) -> set[tuple[str, str, str]]:
    result: set[tuple[str, str, str]] = set()
    for row in state.get("entities", []):
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("entity_id"))
        result.add(
            (
                entity_id,
                "entity_type",
                json.dumps(row.get("entity_type"), sort_keys=True),
            )
        )
        nested = row.get("state") if isinstance(row.get("state"), dict) else {}
        for field in PROMPT_FIELDS:
            result.add(
                (
                    entity_id,
                    field,
                    json.dumps(nested.get(field), sort_keys=True),
                )
            )
    return result


def set_prf(observed: set[Any], expected: set[Any]) -> dict[str, Any]:
    true_positive = len(observed & expected)
    observed_count = len(observed)
    expected_count = len(expected)
    precision = (
        true_positive / observed_count
        if observed_count
        else (1.0 if not expected_count else 0.0)
    )
    recall = (
        true_positive / expected_count
        if expected_count
        else (1.0 if not observed_count else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "observed": observed_count,
        "expected": expected_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def state_from_standard_row(row: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    carried = row.get("parsed_output", {}).get("carried_state", {})
    parse_success = carried.get("parse_success") is True
    return parse_success, carried.get("state", {}) if parse_success else {}


def severity_bucket(error_count: int) -> str:
    if error_count == 0:
        return "exact"
    if error_count == 1:
        return "one"
    if error_count <= 3:
        return "two_to_three"
    if error_count <= 9:
        return "four_to_nine"
    return "ten_or_more"


def compare_state(
    *,
    provider: str,
    condition: str,
    row_id: str,
    task_id: str,
    parse_success: bool,
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    expected_facts = entity_fact_set(expected)
    observed_facts = entity_fact_set(observed)
    expected_relations = relation_set(expected)
    observed_relations = relation_set(observed)
    expected_exclusions = exclusion_set(expected)
    observed_exclusions = exclusion_set(observed)

    # The second implementation must agree on the canonical edge set.
    implementation_a = expected_relations, observed_relations
    implementation_b = (
        relation_set_independent(expected),
        relation_set_independent(observed),
    )
    implementation_a_serialized = tuple(
        frozenset(
            json.dumps(
                {
                    "direction": edge[3],
                    "relation_type": edge[0],
                    "source_id": edge[1],
                    "target_id": edge[2],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for edge in edges
        )
        for edges in implementation_a
    )
    if implementation_a_serialized != implementation_b:
        raise RuntimeError(f"independent relation-set mismatch for {row_id}")

    observed_without_extra_context = {
        edge
        for edge in observed_relations
        if not (edge[0] == "context_link" and edge not in expected_relations)
    }
    strict_p = bool(
        parse_success
        and observed_facts == expected_facts
        and observed_relations == expected_relations
        and observed_exclusions == expected_exclusions
    )
    context_ignored_p = bool(
        parse_success
        and observed_facts == expected_facts
        and observed_without_extra_context == expected_relations
        and observed_exclusions == expected_exclusions
    )
    error_count = (
        len(observed_facts ^ expected_facts)
        + len(observed_relations ^ expected_relations)
        + len(observed_exclusions ^ expected_exclusions)
    )
    relation_metrics = set_prf(observed_relations, expected_relations)
    return {
        "provider": provider,
        "condition": condition,
        "row_id": row_id,
        "task_id": task_id,
        "parse_success": parse_success,
        "P_set_semantics": strict_p,
        "P_ignore_extra_context_link": context_ignored_p,
        "state_fact_metrics": set_prf(observed_facts, expected_facts),
        "relation_metrics": relation_metrics,
        "exclusion_metrics": set_prf(observed_exclusions, expected_exclusions),
        "semantic_error_count": error_count,
        "severity_bucket": severity_bucket(error_count),
        "missing_relations": [
            list(edge) for edge in sorted(expected_relations - observed_relations)
        ],
        "extra_relations": [
            list(edge) for edge in sorted(observed_relations - expected_relations)
        ],
        "extra_context_link_count": sum(
            edge[0] == "context_link"
            for edge in observed_relations - expected_relations
        ),
        "implementation_agreement": True,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parseable = [row for row in rows if row["parse_success"]]
    rel_tp = sum(row["relation_metrics"]["true_positive"] for row in rows)
    rel_expected = sum(row["relation_metrics"]["expected"] for row in rows)
    rel_observed = sum(row["relation_metrics"]["observed"] for row in rows)
    errors = [row["semantic_error_count"] for row in rows]
    result = {
        "n": len(rows),
        "rows": len(rows),
        "parseable_rows": len(parseable),
        "P": {
            "count": sum(row["P_set_semantics"] for row in rows),
            "rate": sum(row["P_set_semantics"] for row in rows) / len(rows),
        },
        "P_ignore_extra_context_link": {
            "count": sum(row["P_ignore_extra_context_link"] for row in rows),
            "rate": sum(row["P_ignore_extra_context_link"] for row in rows) / len(rows),
        },
        "relation_micro": {
            "true_positive": rel_tp,
            "expected": rel_expected,
            "observed": rel_observed,
            "precision": rel_tp / rel_observed if rel_observed else 0.0,
            "recall": rel_tp / rel_expected if rel_expected else 0.0,
        },
        "component_macro_parseable": {
            "entity_precision": mean(
                row["state_fact_metrics"]["precision"] for row in parseable
            ),
            "entity_recall": mean(
                row["state_fact_metrics"]["recall"] for row in parseable
            ),
            "relation_precision": mean(
                row["relation_metrics"]["precision"] for row in parseable
            ),
            "relation_recall": mean(
                row["relation_metrics"]["recall"] for row in parseable
            ),
            "exclusion_precision": mean(
                row["exclusion_metrics"]["precision"] for row in parseable
            ),
            "exclusion_recall": mean(
                row["exclusion_metrics"]["recall"] for row in parseable
            ),
        },
        "semantic_error_count": {
            "median": median(errors),
            "mean": mean(errors),
            "distribution": dict(
                sorted(Counter(severity_bucket(x) for x in errors).items())
            ),
        },
        "missing_relation_types": dict(
            sorted(
                Counter(
                    edge[0] for row in rows for edge in row["missing_relations"]
                ).items()
            )
        ),
        "extra_relation_types": dict(
            sorted(
                Counter(
                    edge[0] for row in rows for edge in row["extra_relations"]
                ).items()
            )
        ),
    }
    if all("R" in row for row in rows):
        result["R"] = {
            "count": sum(bool(row["R"]) for row in rows),
            "rate": sum(bool(row["R"]) for row in rows) / len(rows),
        }
    return result


def group_robustness(
    rows: list[dict[str, Any]], metadata: dict[str, dict[str, str]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    leave_one_out: dict[str, Any] = {}
    for output_key, metadata_key in (
        ("by_family", "family"),
        ("by_action_type", "action_type"),
        ("by_pool", "pool"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[metadata[row["task_id"]][metadata_key]].append(row)
        result[output_key] = {
            group_name: {
                **summarize(group_rows),
                "task_count": len({row["task_id"] for row in group_rows}),
            }
            for group_name, group_rows in sorted(grouped.items())
        }
        leave_one_out[output_key] = {
            group_name: summarize(
                [
                    row
                    for row in rows
                    if metadata[row["task_id"]][metadata_key] != group_name
                ]
            )
            for group_name in sorted(grouped)
        }
    result["leave_one_out"] = leave_one_out
    return result


def entity_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("entity_id")): row
        for row in state.get("entities", [])
        if isinstance(row, dict)
    }


def replacement_audit(
    deepseek_rows: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    scorer: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    mismatch_fields = Counter()
    target_roles = Counter()
    for raw in deepseek_rows:
        task_id = raw["task_id"]
        action = scorer[task_id]["expected_action"]
        if action["action_type"] != "replace_current_object":
            continue
        parse_success, observed = state_from_standard_row(raw)
        expected_entities = entity_map(references[task_id])
        observed_entities = entity_map(observed)
        required = [
            ("primary_target", str(action["primary_target"])),
            ("replacement_target", str(action["replacement_target"])),
        ]
        row_fields = []
        for role, entity_id in required:
            target_roles[role] += 1
            expected_row = expected_entities.get(entity_id)
            observed_row = observed_entities.get(entity_id)
            if observed_row is None:
                mismatch_fields[f"{role}:entity_missing"] += 1
                row_fields.append(f"{role}:entity_missing")
                continue
            if expected_row is None:
                raise RuntimeError(
                    f"reference missing replacement endpoint {entity_id}"
                )
            if observed_row.get("entity_type") != expected_row.get("entity_type"):
                mismatch_fields[f"{role}:entity_type"] += 1
                row_fields.append(f"{role}:entity_type")
            for field in PROMPT_FIELDS:
                observed_value = (observed_row.get("state") or {}).get(field)
                expected_value = (expected_row.get("state") or {}).get(field)
                if observed_value != expected_value:
                    mismatch_fields[f"{role}:{field}"] += 1
                    row_fields.append(f"{role}:{field}")
        rows.append(
            {
                "row_id": raw["row_id"],
                "task_id": task_id,
                "condition": raw["method_id"],
                "parse_success": parse_success,
                "mismatch_fields": sorted(row_fields),
            }
        )
    return {
        "schema_version": "pcg_v3.review_p0_p1.replacement_r_audit.v1",
        "row_count": len(rows),
        "action_required_success": 0,
        "mismatch_field_counts": dict(sorted(mismatch_fields.items())),
        "required_endpoint_role_counts": dict(sorted(target_roles.items())),
        "rows_with_no_endpoint_field_mismatch": sum(
            not row["mismatch_fields"] for row in rows
        ),
        "interpretation": (
            "Replacement R=0 is driven by exact state mismatches on the frozen "
            "source and replacement endpoints; it is not a missing-relation-only result."
        ),
        "rows": rows,
    }
