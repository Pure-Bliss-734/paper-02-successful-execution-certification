"""Numerical cells in appendix tables, with results expressed on the manuscript scale."""

from semantic_rows import controlled_rows
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]


def rows(key):
    with (ROOT / "data" / (key + ".jsonl")).open(encoding="utf8") as handle:
        return [json.loads(line) for line in handle]


def sums(rs, *fields):
    return [sum(bool(r[k]) for r in rs) for k in fields]


def where(rs, **criteria):
    return [r for r in rs if all(r[k] == v for k, v in criteria.items())]


def run():
    output = {}
    differences = {}

    def check(table, actual, reference):
        if actual != reference:
            differences[table] = {"observed": actual, "printed": reference}
        output[table] = actual

    allc = rows("corrected")
    ns = where(allc, condition="N")
    sem = rows("semantic_scores")
    lookup = {r["row_id"]: r for r in sem}
    providers = ("deepseek", "qwen", "glm")
    cc = [where(ns, model_provider=p) for p in providers] + [ns]
    check(
        "T2",
        [sums(rs, "P", "R", "M_v2", "E_v2") for rs in cc],
        [[6, 42, 51, 55], [0, 42, 51, 54], [0, 37, 52, 55], [6, 121, 154, 164]],
    )
    # Table 3/4 count denominators and the three contingent success profiles are also checked by headline replay.
    successes = [[r for r in rs if r["E_v2"]] for rs in cc]
    check(
        "T21",
        [
            [
                len(rs),
                sum(r["P"] for r in rs),
                sum(
                    lookup[r["semantic_record_row_id"]]["P_ignore_extra_context_link"]
                    for r in rs
                ),
                sum(r["R"] for r in rs),
            ]
            for rs in successes
        ],
        [[55, 6, 19, 42], [54, 0, 6, 42], [55, 0, 5, 34], [164, 6, 30, 118]],
    )
    rs = rows("v1_execution")
    state = controlled_rows()
    actual = []
    for cond in "HNSBC":
        s = where(rs, condition=cond)
        sub = where(state, condition=cond, model_provider="deepseek")
        base = sums(
            s,
            "answer_success",
            "frozen_action_contract_success",
            "executed_operational_terminal_success",
        )
        if cond != "H":
            base += sums(
                s,
                "state_exact_success",
                "strict_exact_state_to_executed_terminal_success",
            ) + sums(sub, "P_set_semantics", "R")
        actual.append([round(v / len(s) * 100, 2) for v in base])
    check(
        "T11",
        actual,
        [
            [30.42, 25, 45],
            [32.50, 18.75, 38.75, 5.83, 2.08, 5.83, 72.50],
            [30.42, 16.67, 36.67, 2.08, 1.25, 3.33, 69.17],
            [31.25, 17.08, 36.67, 2.92, 1.25, 4.17, 68.75],
            [30.42, 17.92, 37.92, 3.33, 1.25, 4.58, 68.75],
        ],
    )
    text = []
    for rs in cc:
        scored = [lookup[r["semantic_record_row_id"]] for r in rs]
        positive = [r for r in rs if r["nonzero_delta"] and r["E_v2"]]
        text.append(
            [
                sum(r["relation_metrics"]["true_positive"] for r in scored),
                sum(r["relation_metrics"]["expected"] for r in scored),
                statistics.median(r["semantic_error_count"] for r in scored),
                sum(not r["R"] for r in positive),
                len(positive),
            ]
        )
    check(
        "T12 additional columns",
        text,
        [
            [94, 105, 5, 0, 33],
            [93, 105, 5.5, 0, 33],
            [95, 105, 6, 6, 30],
            [282, 315, 5, 6, 96],
        ],
    )
    gold = rows("gold")
    check(
        "T13",
        [
            [len(rs), *sums(rs, "M_v2", "E_v2")]
            for p in providers
            for rs in [where(allc, model_provider=p)]
        ]
        + [
            [
                len(rs),
                *sums(
                    [r["metrics"] for r in rs],
                    "full_action_contract_match",
                    "terminal_state_success",
                ),
            ]
            for p in providers
            for rs in [where(gold, provider=p)]
        ],
        [
            [180, 151, 163],
            [180, 147, 155],
            [60, 52, 55],
            [60, 56, 57],
            [60, 56, 59],
            [60, 52, 55],
        ],
    )
    check(
        "T14",
        [
            [len(rs), *sums(rs, "P", "R", "M_v2", "E_v2")]
            for p in providers[:2]
            for cond in "NSBC"
            for rs in [where(allc, model_provider=p, condition=cond)]
        ],
        [
            [60, 6, 42, 51, 55],
            [40, 2, 27, 34, 36],
            [40, 3, 28, 33, 36],
            [40, 3, 28, 33, 36],
            [60, 0, 42, 51, 54],
            [40, 0, 17, 34, 36],
            [40, 0, 17, 33, 34],
            [40, 0, 12, 29, 31],
        ],
    )
    interv = rows("intervention")
    by = {(r["model_provider"], r["task_id"], r["arm"]): r for r in interv}
    changes = []
    selected = where(interv, arm="necessary_state_deletion")
    groups = {
        r["task_id"]: (
            r["deleted_entity"] in json.dumps(r["future_events"], ensure_ascii=False)
        )
        for r in rows("deletion_groups")
    }
    if any(
        groups[r["task_id"]] != r["future_event_mentions_deleted_entity_task_audit"]
        for r in selected
    ):
        raise ValueError("Table 16 event grouping disagrees with task events")
    for rs in (
        selected,
        [r for r in selected if not r["deleted_primary_target_task_audit"]],
        [r for r in selected if r["future_event_mentions_deleted_entity_task_audit"]],
        [
            r
            for r in selected
            if not r["future_event_mentions_deleted_entity_task_audit"]
        ],
    ):
        changes.append(
            [
                len(rs),
                *[
                    round(
                        100
                        * statistics.mean(
                            int(r[k])
                            - int(
                                by[
                                    r["model_provider"],
                                    r["task_id"],
                                    "intact_gold_state",
                                ][k]
                            )
                            for r in rs
                        ),
                        2,
                    )
                    for k in ("M", "E")
                ],
            ]
        )
    check(
        "T16",
        changes,
        [[60, -65, -11.67], [48, -66.67, 0], [24, -62.50, -29.17], [36, -66.67, 0]],
    )
    sev = []
    for p in providers:
        vals = [r["semantic_error_count"] for r in where(sem, provider=p)]
        sev.append(
            [
                sum(a <= v < b for v in vals)
                for a, b in [(0, 1), (1, 2), (2, 4), (4, 10), (10, 10**6)]
            ]
            + [statistics.median(vals)]
        )
    check(
        "T17",
        sev,
        [[43, 118, 277, 469, 53, 4], [0, 13, 22, 103, 42, 7], [0, 0, 19, 30, 11, 6]],
    )
    repo = rows("repository")
    check(
        "T18",
        [
            [
                len(rs),
                *sums(
                    rs,
                    "parse_success",
                    "answer_success",
                    "action_contract_match",
                    "terminal_state_success",
                    "strict_state_to_terminal_success",
                ),
            ]
            for cond in "HNSBC"
            for rs in [where(repo, condition=cond)]
        ],
        [
            [60, 60, 60, 60, 60, 0],
            [60, 57, 57, 57, 57, 51],
            [60, 56, 56, 56, 56, 48],
            [60, 57, 57, 57, 57, 40],
            [60, 57, 57, 57, 57, 35],
        ],
    )
    cap = rows("capacity_sensitivity")
    check(
        "T19",
        [
            [
                len(rs),
                *sums(
                    rs,
                    "truncated",
                    "parse_success",
                    "state_success",
                    "action_contract_match",
                    "terminal_state_success",
                ),
            ]
            for arm in ["sentinel_8192", "low_8192", "high_65536"]
            for rs in [where(cap, cap_arm=arm)]
        ],
        [[3, 0, 3, 0, 3, 3], [12, 12, 0, 0, 0, 0], [12, 0, 12, 12, 12, 12]],
    )
    check(
        "T20",
        [
            [
                sum(bool(r[x]) == a and bool(r[y]) == b for r in rs)
                for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
            ]
            for rs in [ns, [r for r in ns if r["nonzero_delta"]]]
            for x, y in [("M_v2", "P"), ("E_v2", "P"), ("M_v2", "R"), ("E_v2", "R")]
        ],
        [
            [23, 3, 151, 3],
            [16, 0, 158, 6],
            [16, 10, 43, 111],
            [13, 3, 46, 118],
            [6, 0, 90, 3],
            [3, 0, 93, 3],
            [2, 4, 4, 89],
            [0, 3, 6, 90],
        ],
    )
    comp = [r for r in rows("support_components") if not r["R"] and r["E"]]
    result = []
    for rs in [
        comp,
        where(comp, delta_class="zero"),
        where(comp, delta_class="nonzero"),
    ]:
        result.append(
            [
                len(rs),
                len({r["task_id"] for r in rs}),
                *sums(rs, "required_entity_incomplete", "critical_relation_incomplete"),
                sum(4 <= r["semantic_error_count"] < 10 for r in rs),
                sum(r["semantic_error_count"] >= 10 for r in rs),
            ]
        )
    check(
        "T22",
        result,
        [[46, 12, 46, 1, 35, 11], [40, 9, 40, 1, 30, 10], [6, 3, 6, 0, 5, 1]],
    )
    goldpairs = json.loads((ROOT / "references/gold_pairs.json").read_text())
    check(
        "T23",
        [
            [
                value
                for k in ("M", "E")
                for ds in [[r["gold_minus_model_" + k] for r in rs]]
                for value in [
                    round(100 * statistics.mean(ds), 2),
                    sum(v > 0 for v in ds),
                    sum(v == 0 for v in ds),
                    sum(v < 0 for v in ds),
                ]
            ]
            for rs in [where(goldpairs, provider=p) for p in providers] + [goldpairs]
        ],
        [
            [8.33, 3, 16, 1, 3.33, 2, 17, 1],
            [8.33, 2, 17, 1, 8.33, 2, 17, 1],
            [0, 5, 12, 3, 0, 4, 14, 2],
            [5.56, 10, 45, 5, 3.89, 8, 48, 4],
        ],
    )
    subsets = [
        where(sem, provider="deepseek", condition="N"),
        where(sem, provider="deepseek"),
    ] + [[lookup[r["semantic_record_row_id"]] for r in rs] for rs in cc]
    check(
        "T24",
        [
            [len(rs), *sums(rs, "P_set_semantics", "P_ignore_extra_context_link")]
            for rs in subsets
        ],
        [
            [240, 14, 71],
            [960, 43, 223],
            [60, 6, 19],
            [60, 0, 6],
            [60, 0, 6],
            [180, 6, 31],
        ],
    )
    matched = {r["task_id"] for r in ns}
    check(
        "T27",
        [
            [
                round(100 * statistics.mean(r[k + "_metrics"][m] for r in rs), 2)
                for k in ("state_fact", "relation", "exclusion")
                for m in ("precision", "recall")
            ]
            for p in providers
            for rs in [
                [
                    r
                    for r in sem
                    if r["task_id"] in matched
                    and r["provider"] == p
                    and r["parse_success"]
                ]
            ]
        ],
        [
            [98.03, 98.03, 61.19, 76.81, 91.25, 91.25],
            [96.46, 96.29, 61.46, 76.99, 97.73, 97.73],
            [97.36, 97.36, 57.29, 78.81, 52.54, 52.54],
        ],
    )
    rel = []
    for rs in subsets:
        ex, ob, tp = [
            sum(r["relation_metrics"][k] for r in rs)
            for k in ("expected", "observed", "true_positive")
        ]
        rel.append(
            [
                ex,
                ob,
                tp,
                ex - tp,
                ob - tp,
                round(100 * tp / ob, 2),
                round(
                    100 * tp / (ob - sum(r["extra_context_link_count"] for r in rs)), 2
                ),
                round(100 * tp / ex, 2),
            ]
        )
    check(
        "T28",
        rel,
        [
            [408, 682, 358, 50, 324, 52.49, 98.35, 87.75],
            [1632, 2671, 1413, 219, 1258, 52.90, 99.58, 86.58],
            [105, 182, 94, 11, 88, 51.65, 98.95, 89.52],
            [105, 180, 93, 12, 87, 51.67, 100, 88.57],
            [105, 186, 95, 10, 91, 51.08, 94.06, 90.48],
            [315, 548, 282, 33, 266, 51.46, 97.58, 89.52],
        ],
    )
    val = rows("validation")
    check(
        "T29",
        [
            [
                len(rs),
                sum(r["forbidden_side_effect_free"] for r in rs),
                *[
                    sum(
                        (r["validator_a_success"], r["validator_b_success"]) == (a, b)
                        for r in rs
                    )
                    for a, b in [(1, 1), (1, 0), (0, 1), (0, 0)]
                ],
            ]
            for rs in [where(val, provider=p) for p in providers] + [val]
        ],
        [
            [240, 240, 220, 20, 0, 0],
            [240, 235, 214, 18, 0, 8],
            [120, 119, 110, 5, 1, 4],
            [600, 594, 544, 43, 1, 12],
        ],
    )
    return output, differences


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
