"""Read recorded model message text and compare action payloads with the parsed rows."""

from __future__ import annotations
from collections import defaultdict, Counter
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def decode(text, *, wrapped):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if wrapped and text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
        if match:
            text = match[1]
    try:
        obj = json.loads(text)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def replay():
    by_layer = {}
    mismatches = []
    for layer in (
        "controlled_formal_1200",
        "paired_intervention_480",
        "repository_positive_control_300",
        "glm52_third_family_180",
        "stage2_ablation_240",
        "qwen",
        "natural_repair",
        "bridge",
        "future",
    ):
        records_by_episode = defaultdict(list)
        calls = records(ROOT / "evidence/calls" / (layer + ".jsonl"))
        for call in calls:
            records_by_episode[call["episode"]].append(call)
        parsed = records(ROOT / "data" / ("parsed_" + layer + ".jsonl"))
        empty_actions = 0
        for row in parsed:
            ep = row["row_id"].split("/", 1)[1]
            stored = row["parsed_output"]
            target = (
                (stored.get("value") or {}).get("action")
                if layer == "stage2_ablation_240"
                else stored.get("action")
            )
            options = []
            for call in records_by_episode[ep]:
                stage = call["stage"]
                if stage in (
                    "state_review",
                    "state_generation",
                    "stage1",
                    "future_base_state",
                ):
                    continue
                if call.get("selected") is False:
                    continue
                choices = (call.get("returned") or {}).get("choices") or []
                if not choices:
                    continue
                wrapped = layer not in (
                    "controlled_formal_1200",
                    "repository_positive_control_300",
                )
                value = decode(choices[0].get("text"), wrapped=wrapped)
                if value is None:
                    options.append(None)
                    continue
                if stage == "history_direct":
                    options.append(value.get("action", {}))
                elif (
                    stage in ("future_transition_action", "stage2")
                    or "carried_state" in value
                ):
                    options.append(value.get("action"))
                else:
                    options.append(value)
            actual = options[-1] if options else None
            if not target:
                empty_actions += 1
            if actual != target:
                mismatches.append(
                    {
                        "layer": layer,
                        "episode": ep,
                        "observed": str(actual)[:135],
                        "reference": str(target)[:135],
                    }
                )
        by_layer[layer] = {"episodes": len(parsed), "empty_actions": empty_actions}
    return by_layer, mismatches


if __name__ == "__main__":
    result, miss = replay()
    print(result)
    print(Counter(x["layer"] for x in miss))
    for k in result:
        print(k, next((r for r in miss if r["layer"] == k), None))


def replay_states():
    import sys

    sys.path.insert(0, str(ROOT / "tools/scorers"))
    from pcg_v3_m30.methods import normalize_carried_state_v2
    from calculations.states import compare_state

    refs = {
        r["task_id"]: r["expected_state"]
        for r in records(ROOT / "data/state_targets.jsonl")
    }
    scores = {r["row_id"]: r for r in records(ROOT / "data/semantic_scores.jsonl")}
    diffs = []
    totals = Counter()
    for layer in (
        "controlled_formal_1200",
        "qwen",
        "glm52_third_family_180",
        "repository_positive_control_300",
    ):
        calls = defaultdict(list)
        for c in records(ROOT / "evidence/calls" / (layer + ".jsonl")):
            if c["stage"] in ("state_generation", "state_review"):
                calls[c["episode"]].append(c)
        for row in records(ROOT / "data" / ("parsed_" + layer + ".jsonl")):
            parsed = row["parsed_output"]
            ref = parsed.get("carried_state") or parsed.get("normalized_carried_state")
            if ref is None:
                continue
            rid = row["row_id"]
            ep = rid.split("/", 1)[1]
            stage = (
                "state_review"
                if parsed.get("condition") in ("B", "C")
                else "state_generation"
            )
            stage_calls = [
                c
                for c in calls[ep]
                if c["stage"] == stage
                and c.get("selected") is not False
                and (c.get("returned") or {}).get("choices")
            ]
            raw = (
                stage_calls[-1]["returned"]["choices"][0].get("text")
                if stage_calls
                else None
            )
            obj = decode(raw, wrapped=layer in ("qwen", "glm52_third_family_180")) or {}
            state = normalize_carried_state_v2(
                obj.get("carried_state"),
                "natural" if parsed.get("condition", "N") == "N" else "structured",
            )
            if (
                state["parse_success"] != ref["parse_success"]
                or state["parse_success"]
                and state["state"] != ref.get("state")
            ):
                diffs.append((layer, rid, "normalized"))
            if rid in scores:
                old = scores[rid]
                got = compare_state(
                    provider=old["provider"],
                    condition=old["condition"],
                    row_id=rid,
                    task_id=row["task_id"],
                    parse_success=ref["parse_success"],
                    observed=ref.get("state", {}) if ref["parse_success"] else {},
                    expected=refs[row["task_id"]],
                )
                for k in (
                    "P_set_semantics",
                    "P_ignore_extra_context_link",
                    "semantic_error_count",
                    "relation_metrics",
                    "exclusion_metrics",
                    "state_fact_metrics",
                    "extra_context_link_count",
                ):
                    if got[k] != old[k]:
                        diffs.append((layer, rid, k))
            totals[layer] += 1
    return dict(totals), diffs
