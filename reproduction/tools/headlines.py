from __future__ import annotations
import json
from collections import Counter
from typing import Any
from semantic_rows import controlled_rows
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class ReplayFailure(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayFailure(message)


def bool_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows)


def compute_results() -> dict[str, Any]:
    row_ledgers = DATA
    matched = [
        row
        for row in load_jsonl(row_ledgers / "corrected.jsonl")
        if row["condition"] == "N"
    ]
    semantic_records = load_jsonl(DATA / "semantic_scores.jsonl")
    semantic_by_id = {row["row_id"]: row for row in semantic_records}
    for row in matched:
        semantic_id = row.get("semantic_record_row_id") or row["analysis_row_id"]
        require(
            semantic_id in semantic_by_id, f"missing set-semantic row: {semantic_id}"
        )
        row["P_noctx_replayed"] = bool(
            semantic_by_id[semantic_id]["P_ignore_extra_context_link"]
        )
    success = [row for row in matched if row["E_v2"]]
    nonzero = [row for row in matched if row["nonzero_delta"]]

    def contingency(
        rows: list[dict[str, Any]], left: str, right: str, lp: str, rp: str
    ) -> dict[str, int]:
        return {
            f"{lp}{int(a)}_{rp}{int(b)}": sum(
                bool(row[left]) == a and bool(row[right]) == b for row in rows
            )
            for a in (False, True)
            for b in (False, True)
        }

    actual: dict[str, Any] = {
        "schema_version": "handoffs.results.v1",
        "matched_natural_180": {
            "n": len(matched),
            "P": bool_count(matched, "P"),
            "P_ignore_extra_context_link": bool_count(matched, "P_noctx_replayed"),
            "R": bool_count(matched, "R"),
            "M": bool_count(matched, "M_v2"),
            "E": bool_count(matched, "E_v2"),
            "success_conditioned": {
                "denominator": len(success),
                "P": bool_count(success, "P"),
                "P_ignore_extra_context_link": bool_count(success, "P_noctx_replayed"),
                "R": bool_count(success, "R"),
            },
            "E_by_P": contingency(matched, "E_v2", "P", "E", "P"),
            "E_by_R": contingency(matched, "E_v2", "R", "E", "R"),
            "nonzero_delta": {
                "n": len(nonzero),
                "successful": bool_count(nonzero, "E_v2"),
                **contingency(nonzero, "E_v2", "R", "E", "R"),
            },
        },
    }

    semantic = [
        row
        for row in controlled_rows()
        if row["condition"] == "N" and row["model_provider"] == "deepseek"
    ]
    actual["primary_deepseek_240"] = {
        "n": len(semantic),
        "P_set_semantics": bool_count(semantic, "P_set_semantics"),
        "P_ignore_extra_context_link": bool_count(
            semantic, "P_ignore_extra_context_link"
        ),
        "R": bool_count(semantic, "R"),
        "relations": {
            "expected": sum(row["relation_expected"] for row in semantic),
            "observed": sum(row["relation_observed"] for row in semantic),
            "true_positive": sum(row["relation_true_positive"] for row in semantic),
            "false_negative": sum(row["relation_false_negative"] for row in semantic),
            "false_positive": sum(row["relation_false_positive"] for row in semantic),
        },
    }

    intervention = load_jsonl(row_ledgers / "intervention.jsonl")
    actual["paired_intervention_240"] = {}
    for arm in (
        "intact_gold_state",
        "necessary_state_deletion",
        "counterfactual_wrong_state",
        "unrelated_state_perturbation",
    ):
        rows = [row for row in intervention if row["arm"] == arm]
        actual["paired_intervention_240"][arm] = {
            "n": len(rows),
            "M": bool_count(rows, "M"),
            "E": bool_count(rows, "E"),
        }

    repository = load_jsonl(row_ledgers / "repository.jsonl")
    state_handoff = [
        row for row in repository if row["condition"] in {"N", "S", "B", "C"}
    ]
    nontruncated = [row for row in state_handoff if not row["truncated"]]
    actual["repository_positive_control_300"] = {
        "n": len(repository),
        "joint_strict_state_to_terminal": bool_count(
            repository, "strict_state_to_terminal_success"
        ),
        "state_handoff_conditions_n": len(state_handoff),
        "nontruncated_state_handoff_n": len(nontruncated),
        "nontruncated_terminal_success": bool_count(
            nontruncated, "terminal_state_success"
        ),
        "nontruncated_joint_success": bool_count(
            nontruncated, "strict_state_to_terminal_success"
        ),
    }

    capacity = load_jsonl(row_ledgers / "capacity_sensitivity.jsonl")
    arms = Counter(row["cap_arm"] for row in capacity)
    actual["capacity_sensitivity_27"] = {
        "n": len(capacity),
        **{name: arms[name] for name in ("sentinel_8192", "low_8192", "high_65536")},
    }

    future_rows = load_jsonl(DATA / "future.jsonl")
    future: dict[str, Any] = {}
    for provider in ("deepseek", "qwen", "glm"):
        rows = [row for row in future_rows if row["provider"] == provider]
        metrics = [row["metrics"] for row in rows]
        future[provider] = {
            "n": len(rows),
            "A": bool_count(metrics, "stage1_answer_exact"),
            "safe": bool_count(metrics, "post_probe_state_safety"),
            "P": bool_count(metrics, "post_probe_state_completeness"),
            "M": bool_count(metrics, "post_probe_full_action_contract_match"),
            "E": bool_count(metrics, "post_probe_terminal_state_success"),
        }
    actual["future_probe_180"] = future

    replacement = load_json(ROOT / "references/replacement.json")
    mismatch = replacement["mismatch_field_counts"]
    actual["replacement_lifecycle_192"] = {
        "n": replacement["row_count"],
        "primary_target_active": mismatch["primary_target:active"],
        "primary_target_current": mismatch["primary_target:current"],
        "replacement_target_active": mismatch["replacement_target:active"],
        "replacement_target_current": mismatch["replacement_target:current"],
    }

    natural_e5 = load_jsonl(DATA / "natural_repair.jsonl")
    natural_pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in natural_e5:
        candidate, arm = row["candidate_id"], row["arm"]
        natural_pairs.setdefault(candidate, {})[arm] = row
    require(
        all(set(pair) == {"A", "B"} for pair in natural_pairs.values()),
        "natural repair has incomplete pairs",
    )
    actual["natural_error_repair_106_pairs"] = {
        "pairs": len(natural_pairs),
        "M_A": sum(pair["A"]["M"] for pair in natural_pairs.values()),
        "M_B": sum(pair["B"]["M"] for pair in natural_pairs.values()),
        "E_A": sum(pair["A"]["E"] for pair in natural_pairs.values()),
        "E_B": sum(pair["B"]["E"] for pair in natural_pairs.values()),
    }

    bridge_rows = load_jsonl(DATA / "bridge.jsonl")
    fixed = [
        row
        for row in bridge_rows
        if row["bridge_kind"] == "candidate_fixed_input_replay"
    ]
    destructive = [
        row
        for row in bridge_rows
        if row["bridge_kind"] == "destructive_intervention_replay"
    ]
    actual["deepseek_bridge_58"] = {
        "rows": len(bridge_rows),
        "fixed_rows": len(fixed),
        "fixed_M_historical": bool_count(fixed, "historical_M"),
        "fixed_M_current": bool_count(fixed, "current_M"),
        "fixed_E_historical": bool_count(fixed, "historical_E"),
        "fixed_E_current": bool_count(fixed, "current_E"),
        "fixed_M_agreement": sum(
            bool(row["historical_M"]) == bool(row["current_M"]) for row in fixed
        ),
        "fixed_E_agreement": sum(
            bool(row["historical_E"]) == bool(row["current_E"]) for row in fixed
        ),
        "destructive_pairs": len({row["task_id"] for row in destructive}),
        "current_G_M": sum(
            bool(row["current_M"]) for row in destructive if row["bridge_arm"] == "G"
        ),
        "current_D_M": sum(
            bool(row["current_M"]) for row in destructive if row["bridge_arm"] == "D"
        ),
        "current_G_E": sum(
            bool(row["current_E"]) for row in destructive if row["bridge_arm"] == "G"
        ),
        "current_D_E": sum(
            bool(row["current_E"]) for row in destructive if row["bridge_arm"] == "D"
        ),
    }
    return actual
