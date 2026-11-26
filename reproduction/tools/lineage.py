"""Join plan, outcome and call records and verify their ordered receipt links."""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_lines(p):
    with p.open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def ticket(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify():
    traces = ROOT / "lineage"
    index = json.loads((traces / "sources.json").read_text())
    observed = {}
    calls = defaultdict(list)
    for p in sorted((ROOT / "evidence/calls").glob("*.jsonl")):
        for r in read_lines(p):
            calls[p.stem + "/" + r["episode"]].append(r)
    for domain, spec in index.items():
        rows = read_lines(traces / (domain + ".jsonl"))
        assert len(rows) == spec["count"]
        last = rows[0]["historical_previous_digest"]
        assert last in ("GENESIS", "0" * 64)
        for i, row in enumerate(rows, 1):
            assert row["sequence"] == i and row["historical_previous_digest"] == last
            assert row["episode"] in calls, (domain, row["episode"])
            last = row["historical_trace_digest"]
            assert ticket(last)
            for k in [
                "historical_h5_digest",
                "historical_e5_digest",
                "recorded_request_digest",
                "recorded_response_digest",
            ]:
                if row.get(k) is not None:
                    assert ticket(row[k])
        assert last == spec["terminal_historical_digest"]
        observed[domain] = len(rows)
    for row in read_lines(traces / "qwen_selected.jsonl"):
        assert row["episode"] in calls and ticket(row["source_row_digest"])
    for row in read_lines(ROOT / "data/bridge.jsonl"):
        old = calls[row["historical_episode"]]
        new = calls["bridge/" + row["episode"]]
        assert new and old
        if not any(a["request"] == b["request"] for a in new for b in old):
            raise ValueError("Snapshot replay differs from the archived fixed input")
    return observed
