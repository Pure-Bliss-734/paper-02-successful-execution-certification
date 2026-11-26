"""Recompute action and state endpoints from individual saved provider messages."""

from __future__ import annotations
from collections import Counter
from copy import deepcopy
from .run import ROOT, lines, read, attempt_value, sha
from . import full_contract as complete
from . import ablation_contract as ablation
from . import future
from pcg_v3_m11.common import operational_view
from pcg_v3_m11.executor import execute_action
from pcg_v3_m11.validator_a import validate_terminal_state_a
from pcg_v3_m11.validator_b import validate_terminal_state_b
from pcg_v3_m30.methods import normalize_carried_state_v2
from pcg_v3_external.semantic_v2 import compare_semantic_states


def execute(task_id, row_id, action, target, original_ablation):
    if original_ablation:
        validate, compare, bridge = (
            ablation.validate_full_action,
            ablation.action_equal,
            ablation.bridge_to_executor_envelope,
        )
    else:
        validate, compare, bridge = (
            complete.validate_full_action_v2,
            complete.action_equal_v2,
            complete.bridge_to_executor_envelope_v2,
        )
    schema, errors = validate(action)
    scores = {
        "M": compare(action, target["expected_full_action"]),
        "E": False,
        "validator_a": False,
        "validator_b": False,
        "safe_execution": False,
    }
    if not schema:
        return scores
    env = bridge(action, task_id=task_id, row_id=row_id)
    before = deepcopy(target["future_runtime_state"])
    expected = target["expected_terminal_state"]
    result = execute_action(before, env, {}, {})
    after = result["post_state"]
    invariants = [
        inv for inv in target["terminal_invariants"] if inv["path"] != "action_history"
    ]
    a = validate_terminal_state_a(
        before,
        after,
        target["expected_delta"],
        invariants,
        {"forbidden_path_prefixes": []},
    )
    b = validate_terminal_state_b(
        before, env, result["transaction_log"], {}, invariants, after
    )
    scores.update(
        E=result["execution_status"] == "success"
        and operational_view(after) == operational_view(expected),
        validator_a=bool(a["terminal_state_success"]),
        validator_b=bool(b["terminal_state_success"]),
        safe_execution=not a["forbidden_side_effect"],
    )
    return scores


def check(scaffold):
    targets = read(ROOT / "protocol/execution_targets.json")
    ablation_actions = read(ROOT / "protocol/ablation_actions.json")
    reference = {
        (r["layer"], r["episode"]): r
        for r in lines(ROOT / "references/episode_outcomes.jsonl")
    }
    repo = {r["task_id"]: r for r in lines(ROOT / "data/repository_tasks.jsonl")}
    capacity = {
        r["episode"]: r for r in lines(ROOT / "data/capacity_sensitivity.jsonl")
    }
    mismatch = []
    tested = Counter()
    for row in scaffold.schedule:
        l, e, t = row["layer"], row["episode"], row["task_id"]
        stages = {
            st["name"]: attempt_value(scaffold.sel[l, e, st["name"]])
            for st in row["stages"]
        }
        if l == "capacity_sensitivity_27":
            cond = row["condition"]
            ref = capacity[e]
            task = repo[t]
            if cond == "H":
                start = stages["history_direct"]
                act = start.get("action")
                state_success = None
            else:
                start = stages["state_generation"]
                raw = stages.get("state_review", start).get("carried_state")
                normalized = normalize_carried_state_v2(
                    raw, "natural" if cond == "N" else "structured"
                )
                state_success = compare_semantic_states(
                    normalized, task["expected_state"]
                )["success"]
                act = stages["state_only_action"]
            receipt = task["execution_receipts"].get(sha(act), {})
            scores = {
                "terminal_state_success": bool(receipt.get("terminal_state_success")),
                "action_contract_match": act == task["expected_action"],
                "answer_success": start.get("answer") == task["expected_answer"],
                "strict_state_to_terminal_success": bool(
                    state_success and receipt.get("terminal_state_success")
                ),
            }
            if cond != "H":
                scores["state_success"] = bool(state_success)
        elif (l, e) in reference:
            ref = reference[l, e]
            last = stages[row["stages"][-1]["name"]]
            target = targets[t]
            act = last.get("action", last) if last else None
            # The response may itself be an action dictionary. A missing model
            # action is passed as None and fails the same semantic schema.
            legacy = (
                l == "stage2_ablation_240"
                or l == "bridge"
                and row["source_episode"].startswith("stage2_ablation_240/")
            )
            if legacy:
                target = {**target, "expected_full_action": ablation_actions[t]}
            scores = execute(t, e, act, target, original_ablation=legacy)
            if "state_complete" in ref:
                norm = normalize_carried_state_v2(last.get("carried_state"), "natural")
                observed = (
                    norm["state"]
                    if norm["parse_success"]
                    else {"entities": [], "relations": [], "exclusions": []}
                )
                scores.update(
                    state_complete=future.state_fact_set(observed)
                    == future.state_fact_set(target["expected_future_state"]),
                    state_safe=norm["parse_success"] and future.state_safety(observed),
                )
            if "answer" in ref:
                initial = stages[row["stages"][0]["name"]]
                scores["answer"] = (
                    initial.get("answer") == target["expected_base_answer"]
                )
        else:
            continue
        for k, v in scores.items():
            if v != ref[k]:
                mismatch.append((l, e, k, v, ref[k]))
        tested[l] += 1
    return tested, mismatch


def controlled_scores(scaffold):
    from . import generic_execution as generic
    from pcg_v3_m27r.execution import build_runtime_state
    from pcg_v3_supplement.independent_state_matcher import match_state

    tasks = {r["candidate_id"]: r for r in lines(ROOT / "data/controlled_tasks.jsonl")}
    refs = {r["task_id"]: r for r in lines(ROOT / "data/state_targets.jsonl")}
    complete_targets = read(ROOT / "protocol/execution_targets.json")
    dependencies = read(ROOT / "protocol/required_dependencies.json")
    v1 = {r["episode"]: r for r in lines(ROOT / "data/v1_execution.jsonl")}
    sem = {r["row_id"]: r for r in lines(ROOT / "data/semantic_scores.jsonl")}
    mismatches = []
    n = Counter()
    for row in scaffold.schedule:
        l, e, t = row["layer"], row["episode"], row["task_id"]
        if l not in ("controlled_formal_1200", "qwen", "glm52_third_family_180"):
            continue
        rid = l + "/" + e
        if rid not in sem and l != "controlled_formal_1200":
            continue
        stages = {
            s["name"]: attempt_value(scaffold.sel[l, e, s["name"]])
            for s in row["stages"]
        }
        last = stages[row["stages"][-1]["name"]]
        observed = last.get("action", {}) if row.get("condition") == "H" else last
        if l == "controlled_formal_1200":
            gold = refs[t]["expected_action"]
            task = tasks[t]
            pre = build_runtime_state(task, task["state_a_final_canonical_state"])
            env, _, _ = generic.observed_envelope(observed, pre, t)
            exec_result, _ = generic.safe_execute(env, pre)
            operational = bool(
                exec_result
                and exec_result["execution_status"] == "success"
                and operational_view(exec_result["post_state"])
                == operational_view(task["expected_terminal_state"])
            )
            if (
                operational != v1[e]["executed_operational_terminal_success"]
                or generic.strict_action_contract_match(observed, gold)
                != v1[e]["frozen_action_contract_success"]
            ):
                mismatches.append((l, e, "generic M/E"))
            n["generic_contract_and_execution"] += 1
        if rid in sem:
            final_state = stages.get(
                "state_review", stages.get("state_generation", {})
            ).get("carried_state")
            rep = (
                "natural"
                if row.get("condition") in ("N", "controlled_N")
                else "structured"
            )
            norm = normalize_carried_state_v2(final_state, rep)
            out = match_state(
                norm["state"] if norm["parse_success"] else {},
                refs[t]["expected_state"],
                (
                    complete_targets[t]["expected_full_action"]
                    if l == "glm52_third_family_180"
                    else refs[t]["expected_action"]
                ),
                {"declared_dependencies": dependencies[t]},
                parse_success=norm["parse_success"],
            )
            if out["R"] != sem[rid]["R"] or out["P"] != sem[rid]["P_set_semantics"]:
                mismatches.append((l, e, "P/R"))
            n["state_and_action_support_" + l] += 1
    return n, mismatches
