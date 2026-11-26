from __future__ import annotations
import math, random, statistics
from collections import defaultdict
from semantic_rows import controlled_rows
from pathlib import Path
import json
from calculations import (
    repairs,
    snapshot,
    interventions,
    gold as gold_math,
    states,
    nonzero as nonzero_math,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF = ROOT / "references"
CHECKS = []


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def compare(actual, expected, label):
    def eq(a, b, loc):
        if isinstance(b, dict):
            assert isinstance(a, dict) and set(a) == set(b), f"{loc}: keys differ"
            for k in b:
                eq(a[k], b[k], loc + "." + k)
        elif isinstance(b, list):
            assert isinstance(a, list) and len(a) == len(b), (
                f"{loc}: list length differs"
            )
            for i, (x, y) in enumerate(zip(a, b)):
                eq(x, y, f"{loc}[{i}]")
        elif isinstance(b, float):
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (
                f"{loc}: {a} != {b}"
            )
        else:
            assert a == b, f"{loc}: {a} != {b}"

    eq(actual, expected, label)
    CHECKS.append({"check": label, "status": "PASS"})


def natural():
    joined = {}
    for row in lines(DATA / "natural_repair.jsonl"):
        cid, arm = row["candidate_id"], row["arm"]
        pair = joined.setdefault(
            cid,
            {
                k: row[k]
                for k in (
                    "candidate_id",
                    "task_id",
                    "provider",
                    "action_type",
                    "task_family",
                    "expected_delta_class",
                    "repair_completes_R",
                    "surfaces",
                )
            },
        )
        assert arm not in pair
        pair[arm] = {m: bool(row[m]) for m in ("M", "E")}
    pairs = list(joined.values())
    assert len(pairs) == 106 and all(set(p) >= {"A", "B"} for p in pairs)
    strata = [("primary_all", pairs)]
    for v in (True, False):
        strata.append(
            (
                f"repair_completes_R={str(v).lower()}",
                [p for p in pairs if p["repair_completes_R"] is v],
            )
        )
    for field in ("provider", "action_type", "task_family", "expected_delta_class"):
        for v in sorted({p[field] for p in pairs}):
            strata.append((f"{field}={v}", [p for p in pairs if p[field] == v]))
    for v in ("matched20", "deepseek80"):
        strata.append((f"surface={v}", [p for p in pairs if v in p["surfaces"]]))
    expected = read(REF / "repair.json")
    results = []
    for label, rows in strata:
        result = {
            "stratum": label,
            "pairs": len(rows),
            "tasks": len({r["task_id"] for r in rows}),
            **{m: repairs.summarize(rows, m, label) for m in ("M", "E")},
        }
        compare(result, expected[label], "natural." + label)
        results.append(result)
    return results


def bridge():
    details = lines(DATA / "bridge.jsonl")
    assert len(details) == 58
    bytask = {}
    for r in details:
        if r["bridge_kind"] != "destructive_intervention_replay":
            continue
        p = bytask.setdefault(r["task_id"], {"task_id": r["task_id"]})
        for m in ("M", "E"):
            for t in ("historical", "current"):
                p[f"{t}_{r['bridge_arm']}_{m}"] = r[f"{t}_{m}"]
    pairs = [bytask[t] for t in sorted(bytask)]
    result = {}
    for m in ("M", "E"):
        old = snapshot.paired_summary(pairs, m, old=True)
        new = snapshot.paired_summary(pairs, m, old=False)
        result[m] = {
            "historical": old,
            "current": new,
            "interaction_current_minus_historical_pp": new["delta_D_minus_G_pp"]
            - old["delta_D_minus_G_pp"],
        }
    compare(result, read(REF / "bridge.json"), "bridge.full_inference")
    return result


def intervention():
    rows = [
        {
            "task_id": r["task_id"],
            "provider": r["model_provider"],
            "arm": r["arm"],
            "metrics": {
                "full_action_contract_match": r["M"],
                "terminal_state_success": r["E"],
            },
        }
        for r in lines(DATA / "intervention.jsonl")
    ]
    reference = read(REF / "intervention.json")
    paired = []
    ci = []
    for m in ("full_action_contract_match", "terminal_state_success"):
        for arm in (
            "necessary_state_deletion",
            "counterfactual_wrong_state",
            "unrelated_state_perturbation",
        ):
            paired.append(interventions.paired_summary(rows, m, arm))
            ci.append(interventions.clustered_interval(rows, m, arm))
            for provider in ("deepseek", "qwen", "glm"):
                paired.append(interventions.paired_summary(rows, m, arm, provider))
    compare(paired, reference["paired_contrasts"], "intervention.all_paired_tests")
    compare(ci, reference["task_cluster_bootstrap"], "intervention.all_intervals")
    return {"paired": paired, "intervals": ci}


def gold():
    # Independent regeneration of each model/task mean from matched N rows and
    # archived gold-state action rows; no saved pair difference is used as input.
    naturalrows = [r for r in lines(DATA / "corrected.jsonl") if r["condition"] == "N"]
    grouped = defaultdict(list)
    for r in naturalrows:
        grouped[(r["model_provider"], r["task_id"])].append(r)
    goldrows = lines(DATA / "gold.jsonl")
    gg = defaultdict(list)
    for r in goldrows:
        gg[(r["provider"], r["task_id"])].append(r)
    source = {"task_level_pairs": read(REF / "gold_pairs.json")}
    rebuilt = []
    for old in source["task_level_pairs"]:
        key = (old["provider"], old["task_id"])
        ns = grouped[key]
        gs = gg[key]
        entry = {k: old[k] for k in ("provider", "task_id", "pool")}
        for m, field in [
            ("M", "full_action_contract_match"),
            ("E", "terminal_state_success"),
        ]:
            a = statistics.mean(int(r[m + "_v2"]) for r in ns)
            b = statistics.mean(int(r["metrics"][field]) for r in gs)
            entry.update(
                {
                    f"model_state_{m}_mean": a,
                    f"gold_state_{m}_mean": b,
                    f"gold_minus_model_{m}": b - a,
                }
            )
        rebuilt.append(entry)
    compare(rebuilt, source["task_level_pairs"], "gold.pairs_from_rows")
    clusters = defaultdict(list)
    for r in rebuilt:
        clusters[r["task_id"]].append(r)
    tasks = sorted(clusters)
    rng = random.Random(20260721)
    draws = {"M": [], "E": []}
    values = {
        m: {
            t: statistics.fmean(r[f"gold_minus_model_{m}"] for r in clusters[t])
            for t in tasks
        }
        for m in draws
    }
    for _ in range(20000):
        sampled = [rng.choice(tasks) for t in tasks]
        for m in draws:
            draws[m].append(statistics.fmean(values[m][t] for t in sampled))
    q = gold_math.quantile
    result = {}
    expected = read(REF / "gold_intervals.json")
    for m in draws:
        diffs = [r[f"gold_minus_model_{m}"] for r in rebuilt]
        result[m] = {
            "endpoint": {"M": "action_accuracy", "E": "outcome_success"}[m],
            "point_difference_proportion": statistics.fmean(diffs),
            "point_difference_percentage_points": statistics.fmean(diffs) * 100,
            "task_cluster_percentile_interval_95_proportion": [
                q(sorted(draws[m]), v) for v in (0.025, 0.975)
            ],
            "task_cluster_percentile_interval_95_percentage_points": [
                q(sorted(draws[m]), v) * 100 for v in (0.025, 0.975)
            ],
            "gold_better_model_task_pairs": sum(x > 0 for x in diffs),
            "tied_model_task_pairs": sum(x == 0 for x in diffs),
            "gold_worse_model_task_pairs": sum(x < 0 for x in diffs),
        }
    compare(result, expected, "gold.full_intervals")
    return result


def replacement_and_groups():
    refs = {r["task_id"]: r for r in lines(DATA / "state_targets.jsonl")}
    raw = []
    for r in lines(DATA / "parsed_controlled_formal_1200.jsonl"):
        payload = r["parsed_output"]
        if payload["condition"] in ("N", "S", "B", "C"):
            raw.append(
                {
                    "row_id": r["row_id"],
                    "task_id": r["task_id"],
                    "method_id": payload["condition"],
                    "parsed_output": payload,
                }
            )
    result = states.replacement_audit(
        raw,
        {t: r["expected_state"] for t, r in refs.items()},
        {t: {"expected_action": r["expected_action"]} for t, r in refs.items()},
    )
    expected = read(REF / "replacement.json")
    compare(
        {k: result[k] for k in expected}, expected, "replacement.from_parsed_states"
    )
    meta = {r["candidate_id"]: r for r in lines(DATA / "controlled_tasks.jsonl")}
    semantic = lines(DATA / "semantic_scores.jsonl")
    ds = [
        r
        for r in semantic
        if r["provider"] == "deepseek" and r["condition"] in ("N", "S", "B", "C")
    ]
    groups = states.group_robustness(ds, meta)
    compare(
        groups, read(REF / "group_rates.json"), "groups.all_strata_and_leave_one_out"
    )
    return {"replacement": result, "groups": groups}


def state_intervals():
    rows = controlled_rows()
    expected = read(REF / "state_contrasts.json")
    result = {}
    for provider, contrasts in expected.items():
        result[provider] = {}
        subset = [r for r in rows if r["model_provider"] == provider]
        for name, ref in contrasts.items():
            left, right = name.split("_minus_")
            idx = {(r["task_id"], r["repeat"], r["condition"]): r for r in subset}
            pairs = []
            for t, rep, cond in sorted(idx):
                if cond == left and (t, rep, right) in idx:
                    pairs.append((t, idx[t, rep, left], idx[t, rep, right]))
            taskids = sorted({p[0] for p in pairs})
            values = {}
            for m, spec in ref["metrics"].items():
                field = "P_set_semantics" if m == "P" else m
                bytask = {
                    t: [
                        int(a[field]) - int(b[field])
                        for task, a, b in pairs
                        if task == t
                    ]
                    for t in taskids
                }
                rng = random.Random(spec["seed"])
                draws = []
                for _ in range(spec["replicates"]):
                    selected = [
                        d
                        for t in [rng.choice(taskids) for _ in taskids]
                        for d in bytask[t]
                    ]
                    draws.append(sum(selected) / len(selected))
                draws.sort()
                values[m] = {
                    "estimate": sum(int(a[field]) - int(b[field]) for t, a, b in pairs)
                    / len(pairs),
                    "p025": draws[round((len(draws) - 1) * 0.025)],
                    "p975": draws[round((len(draws) - 1) * 0.975)],
                    "replicates": spec["replicates"],
                    "seed": spec["seed"],
                }
            result[provider][name] = {"task_count": len(taskids), "metrics": values}
    compare(result, expected, "state_condition_contrasts.from_row_ledgers")
    return result


def primary_and_nonzero():
    rows = [
        r
        for r in controlled_rows()
        if r["model_provider"] == "deepseek" and r["condition"] == "N"
    ]
    tasks = sorted({r["task_id"] for r in rows})
    rng = random.Random(20260721)
    q = gold_math.quantile
    result = {
        "primary": {},
        "primary_schedule": {
            "seed": 20260721,
            "replicates": 20000,
            "metric_order": ["P", "R"],
            "rng": "one sequential stream",
            "provenance": "Explicit reconstruction schedule; verified against manuscript-rounded endpoints, not asserted to recover an archived RNG state.",
        },
    }
    for m, field, expected in [
        ("P", "P_set_semantics", [1.25, 11.25]),
        ("R", "R", [62.50, 81.67]),
    ]:
        vals = {t: sum(r[field] for r in rows if r["task_id"] == t) / 3 for t in tasks}
        draws = sorted(
            sum(vals[rng.choice(tasks)] for _ in tasks) / len(tasks)
            for _ in range(20000)
        )
        ci = [100 * q(draws, v) for v in (0.025, 0.975)]
        compare(
            [round(x, 2) for x in ci],
            expected,
            "primary.manuscript_rounded_interval." + m,
        )
        result["primary"][m] = {
            "count": sum(r[field] for r in rows),
            "rows": len(rows),
            "task_clusters": len(tasks),
            "ci95_pp": ci,
        }
    nz = [
        dict(r, E=r["E_v2"])
        for r in lines(DATA / "corrected.jsonl")
        if r["condition"] == "N" and r["nonzero_delta"]
    ]
    nonzero = {
        key: nonzero_math.bootstrap_task_cluster(nz, key)
        for key in ("E1_R0_rate", "R1_given_E1")
    }
    compare(nonzero, read(REF / "nonzero.json"), "nonzero.all_frozen_intervals")
    result["nonzero"] = nonzero
    return result


def figure_values(actual, stats):
    """Validate every quantitative field in the current seven-figure ledger."""
    figs = read(ROOT / "paper/figures/frozen_figure_values.json")["figures"]
    matched = actual["matched_natural_180"]
    successes = matched["success_conditioned"]
    nz = matched["nonzero_delta"]
    for key, num, den in [
        ("strict_global_fidelity", successes["P"], successes["denominator"]),
        ("immediate_action_support", successes["R"], successes["denominator"]),
        ("state_changing_action_support", nz["E1_R1"], nz["successful"]),
    ]:
        compare(
            {
                "numerator": num,
                "denominator": den,
                "percent": round(100 * num / den, 1),
            },
            figs["figure2_certificate_profile"][key],
            "figure2." + key,
        )
    for label, src, r in [
        ("global_state_fidelity", matched["E_by_P"], "P"),
        ("immediate_action_support", matched["E_by_R"], "R"),
        ("state_changing_support", nz, "R"),
    ]:
        compare(
            [src[f"E{a}_{r}{b}"] for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]],
            figs["figure3_same_run_contingency"][label + "_cells_00_01_10_11"],
            "figure3." + label,
        )
    arms = {
        "intact_gold_state": "intact_gold_state",
        "delete_required_bundle": "necessary_state_deletion",
        "counterfactual_wrong_field": "counterfactual_wrong_state",
        "unrelated_perturbation": "unrelated_state_perturbation",
    }
    inter = actual["paired_intervention_240"]
    for endpoint, m in [("action_accuracy", "M"), ("outcome_success", "E")]:
        compare(
            {label: inter[arm][m] for label, arm in arms.items()},
            figs["figure4_deletion_intervention"][endpoint + "_counts_over_60"],
            "figure4." + m,
        )
        compare(
            round(
                100
                * (inter["necessary_state_deletion"][m] - inter["intact_gold_state"][m])
                / 60,
                1,
            ),
            figs["figure4_deletion_intervention"][
                "delete_minus_intact_percentage_points"
            ][endpoint],
            "figure4.delta." + m,
        )
    counts = stats["replacement_and_groups"]["replacement"]["mismatch_field_counts"]
    compare(
        {
            a: counts[b]
            for a, b in [
                ("old_target_active", "primary_target:active"),
                ("old_target_current", "primary_target:current"),
                ("new_target_current", "replacement_target:current"),
                ("new_target_active", "replacement_target:active"),
            ]
        },
        figs["figure5_replacement_failures"][
            "replacement_lifecycle_mismatch_counts_over_192"
        ],
        "figure5.lifecycle",
    )
    groups = stats["replacement_and_groups"]["groups"]
    actions = {
        "allow_deny": "allow_or_deny_operation",
        "apply_bundle": "apply_current_bundle",
        "migrate_path": "migrate_to_allowed_path",
        "replace_object": "replace_current_object",
        "select_interface": "select_current_interface",
    }
    compare(
        {
            a: round(100 * groups["by_action_type"][b]["R"]["rate"], 1)
            for a, b in actions.items()
        },
        figs["figure5_replacement_failures"]["action_support_percent"],
        "figure5.action_support",
    )
    families = {
        "api_migration": "api_migration_state_transition",
        "near_name_decoys": "decoy_heavy_same_prefix_config",
        "permission_boundary": "permission_boundary_with_attractive_blocked_path",
        "protected_information": "protected_information_redaction_under_task_pressure",
        "requirement_overwrite": "requirement_overwrite_cumulative_active",
    }
    for label, key, mapping in [
        ("family", "by_family", families),
        ("action", "by_action_type", actions),
        (
            "pool",
            "by_pool",
            {"structural_replication": "formal_heldout", "confirmatory": "formal_main"},
        ),
    ]:
        got = {}
        for a, b in mapping.items():
            r = groups[key][b]
            got[a] = [
                round(100 * r["P"]["rate"], 1),
                round(100 * r["R"]["rate"], 1),
                round(100 * r["relation_micro"]["recall"], 1),
                round(r["semantic_error_count"]["mean"], 2),
            ]
        compare(got, figs["figure6_subgroup_heatmap"][label], "figure6." + label)
    ds = [r for r in controlled_rows() if r["model_provider"] == "deepseek"]
    ref = {
        "strict_state_percent": round(
            100 * sum(r["P_set_semantics"] for r in ds) / len(ds), 2
        ),
        "action_support_percent": round(100 * sum(r["R"] for r in ds) / len(ds), 2),
        "relation_recall_percent": round(
            100
            * sum(r["relation_true_positive"] for r in ds)
            / sum(r["relation_expected"] for r in ds),
            2,
        ),
        "mean_semantic_errors": round(
            statistics.mean(r["semantic_error_count"] for r in ds), 2
        ),
    }
    compare(ref, figs["figure6_subgroup_heatmap"]["reference"], "figure6.reference")
    for label, name in [
        ("structured_minus_natural_percentage_points", "S_minus_N"),
        ("checker_repair_minus_blind_review_percentage_points", "C_minus_B"),
    ]:
        got = {}
        for p in ("deepseek", "qwen"):
            for endpoint, m in [("strict_state", "P"), ("action_support", "R")]:
                r = stats["state_condition_intervals"][p][name]["metrics"][m]
                got[p + "_" + endpoint] = {
                    "estimate": round(100 * r["estimate"], 1),
                    "interval": [round(100 * r["p025"], 1), round(100 * r["p975"], 1)],
                }
        compare(got, figs["figure7_state_contrasts"][label], "figure7." + name)
    return {
        "quantitative_figures_checked": 6,
        "protocol_diagram": "nonquantitative",
        "status": "PASS",
    }


def run(actual=None):
    CHECKS.clear()
    result = {}
    for name, fn in [
        ("natural_repair", natural),
        ("deepseek_bridge", bridge),
        ("intervention", intervention),
        ("gold_state", gold),
        ("replacement_and_groups", replacement_and_groups),
        ("state_condition_intervals", state_intervals),
        ("primary_and_nonzero", primary_and_nonzero),
    ]:
        print("Recomputing " + name, flush=True)
        result[name] = fn()
    if actual is not None:
        result["figure_values"] = figure_values(actual, result)
    out = ROOT / "reproduced"
    out.mkdir(exist_ok=True)
    (out / "STATISTICAL_RESULTS.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    return {
        "status": "PASS",
        "checks": CHECKS,
        "provider_calls": 0,
        "scope": "Frozen-output statistical recomputation, not model regeneration or external repository execution",
    }
