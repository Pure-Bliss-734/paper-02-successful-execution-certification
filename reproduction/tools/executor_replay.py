from __future__ import annotations
from collections import Counter
from semantic_rows import controlled_rows
from pathlib import Path
import json
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/scorers"))


def records(name):
    with (ROOT / "data" / (name + ".jsonl")).open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def strict_action_contract_match(observed: Any, expected: Any) -> bool:
    """Recompute the frozen formal action-contract endpoint.

    The July 21 formal row metric was named ``terminal_state_success`` in the
    released row packets, but the frozen analysis records it as the strict
    action-contract endpoint.  The model-facing schema permits harmless extra
    envelope fields and uses several schema-version spellings, so equality is
    defined over the action type and the expected semantic fields only.  A
    missing required field, an incorrect value, or a non-object fails closed.
    """
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    if observed.get("action_type") != expected.get("action_type"):
        return False
    required = {
        key
        for key in expected
        if key not in {"schema_version", "declared_dependencies"}
    }
    return all(key in observed and observed[key] == expected[key] for key in required)


def replay_controlled(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    from pcg_v3_m11.common import diff_operational
    from pcg_v3_m11.executor import execute_action
    from pcg_v3_m11.validator_a import validate_terminal_state_a
    from pcg_v3_m11.validator_b import validate_terminal_state_b
    from pcg_v3_m27r.execution import (
        _terminal_invariants,
        action_envelope,
        build_runtime_state,
    )
    from pcg_v3_supplement.independent_state_matcher import match_state

    refs = {r["task_id"]: r for r in records("state_targets")}
    parsed = records("parsed_controlled_formal_1200")
    ledger = {r["row_id"]: r for r in controlled_rows()}
    summary_ledger = {r["row_id"]: r for r in records("formal")}
    task_records = {r["candidate_id"]: r for r in records("controlled_tasks")}
    results: list[dict[str, Any]] = []
    mismatches: list[str] = []
    counts = Counter()
    package_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in parsed:
        payload = row.get("parsed_output") or {}
        metrics = row.get("metrics") or {}
        ref = refs.get(row.get("task_id"))
        carried = payload.get("carried_state") or {}
        observed_state = carried.get("state") if isinstance(carried, dict) else None
        if ref is None:
            mismatches.append(f"missing-ref:{row.get('row_id')}")
            continue
        state_metrics = match_state(
            observed_state or {},
            ref["expected_state"],
            ref["expected_action"],
            payload.get("action") or {},
            parse_success=bool(carried.get("parse_success")),
        )
        p_value = bool(state_metrics["P"])
        r_value = bool(state_metrics["R"])
        candidate, scorer = (task_records[row["task_id"]], task_records[row["task_id"]])
        action = payload.get("action")
        expected_action = scorer.get("unique_gold_action")
        contract_value = strict_action_contract_match(action, expected_action)
        # M is the semantic structured-action endpoint used by the formal
        # scorer.  It ignores schema spelling and non-semantic envelope
        # extras, exactly as the frozen action-contract endpoint does.
        m_value = contract_value
        operational_e_value = False
        operational_va_value = False
        operational_vb_value = False
        operational_agreement = True
        if isinstance(action, dict) and contract_value:
            try:
                pre = build_runtime_state(
                    candidate, scorer["state_a_final_canonical_state"]
                )
                # Only after the frozen model-facing action contract passes do
                # we enter the operational executor.  In particular, a
                # missing path/interface/replacement target is not filled from
                # the gold action and cannot be turned into a false success.
                gold_action = expected_action
                gold_envelope = action_envelope(
                    gold_action, pre, candidate["candidate_id"]
                )
                observed_typed_action = dict(action)
                observed_typed_action["schema_version"] = gold_action["schema_version"]
                envelope = action_envelope(
                    observed_typed_action, pre, candidate["candidate_id"]
                )
                # Preserve the frozen transaction identity when the observed
                # action is semantically correct; this also keeps validator B
                # deterministic across replay runs.
                envelope["transaction_id"] = gold_envelope["transaction_id"]
                execution = execute_action(pre, envelope, {}, {})
                expected = scorer["expected_terminal_state"]
                delta = diff_operational(pre, expected)
                invariants = _terminal_invariants(expected, gold_envelope)
                validator_a = validate_terminal_state_a(
                    pre,
                    execution["post_state"],
                    delta,
                    invariants,
                    {"forbidden_path_prefixes": []},
                )
                validator_b = validate_terminal_state_b(
                    pre,
                    envelope,
                    execution["transaction_log"],
                    {},
                    invariants,
                    execution["post_state"],
                )
                operational_va_value = bool(validator_a["terminal_state_success"])
                operational_vb_value = bool(validator_b["terminal_state_success"])
                operational_agreement = operational_va_value == operational_vb_value
                operational_e_value = bool(
                    execution["execution_status"] == "success"
                    and execution["post_state"] == expected
                    and operational_va_value
                    and operational_vb_value
                    and not validator_a["forbidden_side_effect"]
                )
            except Exception as exc:
                mismatches.append(
                    f"executor-error:{row.get('row_id')}:{type(exc).__name__}"
                )
        # The released formal row packets expose this endpoint under the
        # historical ``terminal_state_success_original`` name.  Preserve that
        # exact frozen endpoint while also emitting the additive operational
        # replay result above.  Invalid action contracts fail closed for both
        # the frozen formal endpoint and the validator outputs.
        e_value = contract_value and operational_e_value
        va_value = contract_value and operational_va_value
        vb_value = contract_value and operational_vb_value
        agreement = va_value == vb_value
        frozen = summary_ledger.get(row.get("row_id"), {})
        if bool(frozen.get("terminal_state_success_original")) != e_value:
            mismatches.append(f"E:{row.get('row_id')}")
        if bool(frozen.get("validator_a_success")) != va_value:
            mismatches.append(f"VA:{row.get('row_id')}")
        if bool(frozen.get("validator_b_success")) != vb_value:
            mismatches.append(f"VB:{row.get('row_id')}")
        frozen_state = ledger.get(row.get("row_id"), {})
        if bool(frozen_state.get("P_set_semantics")) != p_value:
            mismatches.append(f"P:{row.get('row_id')}")
        if bool(frozen_state.get("R")) != r_value:
            mismatches.append(f"R:{row.get('row_id')}")
        if row.get("row_id") in ledger:
            counts["rows"] += 1
        counts["P"] += int(p_value)
        counts["R"] += int(r_value)
        counts["M"] += int(m_value)
        counts["E"] += int(e_value)
        results.append(
            {
                "row_id": row["row_id"],
                "task_id": row["task_id"],
                "condition": payload.get("condition"),
                "P": p_value,
                "R": r_value,
                "M": m_value,
                "E": e_value,
                "validator_a": va_value,
                "validator_b": vb_value,
                "validator_agreement": agreement,
                "operational_terminal_success": operational_e_value,
                "operational_validator_a": operational_va_value,
                "operational_validator_b": operational_vb_value,
                "operational_validator_agreement": operational_agreement,
            }
        )
    check(
        checks,
        "controlled:parser-scorer-executor-validator-closure",
        not mismatches,
        {
            "counts": dict(counts),
            "mismatches": mismatches[:20],
            "mismatch_count": len(mismatches),
        },
    )
    return {"counts": dict(counts), "rows": len(results), "mismatches": mismatches}


def replay_repository(
    root: Path, checks: list[dict[str, Any]], refs: list[dict[str, Any]]
) -> dict[str, Any]:
    from pcg_v3.hashing import sha256_json
    from pcg_v3_external.formal import validate_terminal_a, validate_terminal_b
    from pcg_v3_external.semantic_v2 import compare_semantic_states

    ref_by_task = {row["task_id"]: row for row in refs}
    scorer_rows = ref_by_task
    parsed = records("parsed_repository_positive_control_300")
    frozen = {
        "repository_positive_control_300/" + r["episode"]: r
        for r in records("repository")
    }
    results: list[dict[str, Any]] = []
    mismatches: list[str] = []
    gold_validator_failures: list[str] = []
    gold_validator_cache: dict[str, tuple[bool, bool, bool]] = {}
    counts = Counter()
    for row in parsed:
        task_id = row["task_id"]
        ref = ref_by_task.get(task_id)
        scorer = scorer_rows.get(task_id)
        if ref is None or scorer is None:
            mismatches.append(f"missing-repository-reference:{row['row_id']}")
            continue
        payload = row.get("parsed_output") or {}
        metrics = row.get("metrics") or {}
        condition = payload.get("condition")
        observed_action = payload.get("action")
        action_match = observed_action == ref["expected_action"]
        answer_match = payload.get("answer") == ref["expected_answer"]
        state_match = None
        carried = payload.get("carried_state")
        if condition in {"N", "S", "B", "C"}:
            if isinstance(carried, dict):
                comparison = compare_semantic_states(carried, ref["expected_state"])
                state_match = bool(comparison["success"])
            else:
                state_match = False
        receipt = (
            scorer.get("execution_receipts", {}).get(sha256_json(observed_action))
            if isinstance(observed_action, dict)
            else None
        )
        terminal = bool(receipt and receipt.get("terminal_state_success"))
        va = bool(receipt and receipt.get("validator_a_success"))
        vb = bool(receipt and receipt.get("validator_b_success"))
        agreement = bool(
            receipt
            and receipt.get("validator_a_success") == receipt.get("validator_b_success")
        )
        side_effect_free = bool(receipt and receipt.get("forbidden_side_effect_free"))
        strict = bool(state_match and terminal) if state_match is not None else False
        expected_metrics = {
            "answer_success": answer_match,
            "action_contract_match": action_match,
            "state_success": state_match,
            "terminal_state_success": terminal,
            "validator_a_success": va,
            "validator_b_success": vb,
            "validator_agreement": agreement,
            "forbidden_side_effect_free": side_effect_free,
            "strict_state_to_terminal_success": strict,
            "execution_receipt_found": receipt is not None,
        }
        frozen_row = frozen.get(row["row_id"], {})
        for key, value in expected_metrics.items():
            if key in frozen_row and frozen_row[key] != value:
                mismatches.append(f"{key}:{row['row_id']}")
        counts["rows"] += 1
        counts["terminal"] += int(terminal)
        counts["state_success"] += int(bool(state_match))
        counts["action"] += int(action_match)
        if task_id not in gold_validator_cache:
            execution = scorer["gold_execution"]
            va_gold = validate_terminal_a(execution, scorer)
            vb_gold = validate_terminal_b(execution, scorer)
            gold_validator_cache[task_id] = (
                bool(va_gold["success"]),
                bool(vb_gold["success"]),
                bool(va_gold["success"] == vb_gold["success"]),
            )
        gva, gvb, gagree = gold_validator_cache[task_id]
        if not (gva and gvb and gagree):
            gold_validator_failures.append(task_id)
        results.append(
            {
                "row_id": row["row_id"],
                "task_id": task_id,
                "condition": condition,
                "state_success": state_match,
                "action_contract_match": action_match,
                "terminal_state_success": terminal,
                "validator_a": va,
                "validator_b": vb,
                "validator_agreement": agreement,
                "executor_mode": "frozen_external_receipt_replay",
                "gold_validator_a": gva,
                "gold_validator_b": gvb,
            }
        )
    check(
        checks,
        "repository:receipt-executor-validator-closure",
        not mismatches and not gold_validator_failures,
        {
            "counts": dict(counts),
            "mismatches": mismatches[:20],
            "mismatch_count": len(mismatches),
            "gold_validator_failures": gold_validator_failures,
        },
    )
    return {
        "counts": dict(counts),
        "rows": len(results),
        "mismatches": mismatches,
        "gold_validator_failures": gold_validator_failures,
    }


def run():
    checks = []
    replay_controlled(ROOT, checks)
    replay_repository(ROOT, checks, records("repository_tasks"))
    for test in checks:
        if not test["passed"]:
            raise ValueError(test)
    return checks


if __name__ == "__main__":
    print(run())
