"""Offline numerical and recorded-response replay for the handoff study."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

# Mandatory regression comparisons use assertions in the historical numerical routines.
if not __debug__:
    raise SystemExit("Run the verifier with Python assertions enabled.")
sys.dont_write_bytecode = True
import appendix
import executor_replay
import headlines
import lineage
import recompute_statistics
import response_replay
import experiment

ROOT = Path(__file__).resolve().parents[1]


def validate_members():
    manifest = json.loads((ROOT / "files.json").read_text(encoding="utf8"))
    excluded = {"files.json"}
    found = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and not p.is_relative_to(ROOT / "reproduced")
        and "__pycache__" not in p.parts
        and p.name != ".DS_Store"
    }
    assert set(manifest) == found - excluded
    for name, digest in manifest.items():
        p = ROOT / name
        assert (
            not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
        ), name
    return len(manifest)


def table16(table_differences):
    expected = {
        "observed": [
            [60, -65, -11.67],
            [48, -66.67, 0],
            [24, -29.17, -29.17],
            [36, -88.89, 0],
        ],
        "printed": [
            [60, -65, -11.67],
            [48, -66.67, 0],
            [24, -62.5, -29.17],
            [36, -66.67, 0],
        ],
    }
    assert table_differences == {"T16": expected}, table_differences
    text = (ROOT / "paper/main.tex").read_text(encoding="utf8")
    for source in [
        r"Future event names deleted entity & 24 & \(-62.50\pp\) & \(-29.17\pp\)",
        r"Future event does not name deleted entity & 36 & \(-66.67\pp\) & \(0.00\pp\)",
    ]:
        assert text.count(source) == 1
    rows = appendix.rows("intervention")
    by = {(r["model_provider"], r["task_id"], r["arm"]): r for r in rows}
    historical = json.loads(
        (ROOT / "references/historical_deletion_strata.json").read_text()
    )
    assert len(historical) == 2
    observed = {}
    for flag in (True, False):
        subset = [
            r
            for r in rows
            if r["arm"] == "necessary_state_deletion"
            and r["future_event_mentions_deleted_entity_task_audit"] == flag
        ]
        ref = next(r for r in historical if r["event_mentions_entity"] == flag)
        for metric in ("M", "E"):
            counterparts = [
                (
                    by[r["model_provider"], r["task_id"], "intact_gold_state"][metric],
                    r[metric],
                )
                for r in subset
            ]
            cells = [
                counterparts.count((bool(a), bool(b)))
                for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))
            ]
            prescribed = (
                [16, 0, 7, 1]
                if flag
                else [4, 0, 32, 0]
                if metric == "M"
                else [4, 0, 0, 32]
            )
            assert cells == prescribed
            h = ref[metric]
            assert [
                sum(cells),
                cells[2] + cells[3],
                cells[1] + cells[3],
                cells[2],
                cells[1],
            ] == [
                h[k]
                for k in (
                    "n_pairs",
                    "intact_success",
                    "comparison_success",
                    "intact_success_comparison_failure",
                    "intact_failure_comparison_success",
                )
            ]
            observed[f"mentions={flag}, {metric}"] = cells
    return {"rows": expected, "cells_00_01_10_11": observed}


def main():
    members = validate_members()
    scaffold_results = experiment.verify()
    receipts = lineage.verify()
    responses, errors = response_replay.replay()
    assert not errors, errors[:2]
    states, errors = response_replay.replay_states()
    assert not errors, errors[:2]
    executors = executor_replay.run()
    actual = headlines.compute_results()
    assert actual == json.loads((ROOT / "references/headlines.json").read_text())
    ledger, discrepancies = appendix.run()
    fixed_difference = table16(discrepancies)
    statistics = recompute_statistics.run(actual)
    out = ROOT / "reproduced"
    out.mkdir(exist_ok=True)
    for file_name, value in [
        ("headline_results", actual),
        ("appendix_tables", ledger),
        ("table16_difference", fixed_difference),
    ]:
        (out / (file_name + ".json")).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "comparison": "Observed values reproduced; two recorded manuscript discrepancies in Table 16",
        "files_checked": members,
        "historical_receipt_counts": receipts,
        "actions_from_recorded_text": responses,
        "states_from_recorded_text": states,
        "executors": executors,
        "experiment_scaffold": scaffold_results,
        "statistical_checks": len(statistics["checks"]),
        "documented_manuscript_differences": 2,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
