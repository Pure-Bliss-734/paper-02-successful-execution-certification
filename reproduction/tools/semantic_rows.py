"""Field names used by the stratified calculations, derived from a single semantic score table."""

import json
from pathlib import Path


def controlled_rows():
    path = Path(__file__).resolve().parents[1] / "data/semantic_scores.jsonl"
    result = []
    with path.open(encoding="utf8") as stream:
        for line in stream:
            r = json.loads(line)
            m = r["relation_metrics"]
            result.append(
                {
                    **r,
                    "model_provider": r["provider"],
                    "relation_true_positive": m["true_positive"],
                    "relation_expected": m["expected"],
                    "relation_observed": m["observed"],
                    "relation_false_positive": m["observed"] - m["true_positive"],
                    "relation_false_negative": m["expected"] - m["true_positive"],
                }
            )
    return result
