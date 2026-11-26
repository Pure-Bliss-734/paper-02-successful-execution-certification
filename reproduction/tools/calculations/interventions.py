from __future__ import annotations
import math, random
from typing import Any

PROVIDERS = ("deepseek", "qwen", "glm")


def exact_mcnemar_p(losses: int, gains: int) -> float:
    n = losses + gains
    if not n:
        return 1.0
    k = min(losses, gains)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def paired_summary(
    rows: list[dict[str, Any]],
    outcome: str,
    comparison_arm: str,
    provider: str | None = None,
) -> dict[str, Any]:
    subset = [row for row in rows if provider is None or row["provider"] == provider]
    indexed = {(row["task_id"], row["provider"], row["arm"]): row for row in subset}
    pairs = []
    for task_id, provider_id, arm in sorted(indexed):
        if arm != "intact_gold_state":
            continue
        base = indexed[(task_id, provider_id, arm)]
        other = indexed[(task_id, provider_id, comparison_arm)]
        pairs.append(
            {
                "task_id": task_id,
                "provider": provider_id,
                "intact": int(bool(base["metrics"][outcome])),
                "comparison": int(bool(other["metrics"][outcome])),
            }
        )
    losses = sum(p["intact"] == 1 and p["comparison"] == 0 for p in pairs)
    gains = sum(p["intact"] == 0 and p["comparison"] == 1 for p in pairs)
    unchanged_success = sum(p["intact"] == p["comparison"] == 1 for p in pairs)
    unchanged_failure = sum(p["intact"] == p["comparison"] == 0 for p in pairs)
    return {
        "provider": provider or "pooled",
        "outcome": outcome,
        "comparison_arm": comparison_arm,
        "n_pairs": len(pairs),
        "intact_success": sum(p["intact"] for p in pairs),
        "comparison_success": sum(p["comparison"] for p in pairs),
        "comparison_minus_intact_percentage_points": 100
        * sum(p["comparison"] - p["intact"] for p in pairs)
        / len(pairs),
        "intact_success_comparison_failure": losses,
        "intact_failure_comparison_success": gains,
        "unchanged_success": unchanged_success,
        "unchanged_failure": unchanged_failure,
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(losses, gains),
        "pairs": pairs,
    }


def clustered_interval(
    rows: list[dict[str, Any]],
    outcome: str,
    comparison_arm: str,
    seed: int = 20260724,
    repetitions: int = 10000,
) -> dict[str, Any]:
    task_ids = sorted({row["task_id"] for row in rows})
    indexed = {
        (row["task_id"], row["provider"], row["arm"]): int(
            bool(row["metrics"][outcome])
        )
        for row in rows
    }

    def contrast(sample: list[str]) -> float:
        deltas = []
        for task_id in sample:
            for provider in PROVIDERS:
                deltas.append(
                    indexed[(task_id, provider, comparison_arm)]
                    - indexed[(task_id, provider, "intact_gold_state")]
                )
        return 100 * sum(deltas) / len(deltas)

    observed = contrast(task_ids)
    rng = random.Random(seed)
    draws = [
        contrast([rng.choice(task_ids) for _ in task_ids]) for _ in range(repetitions)
    ]
    return {
        "outcome": outcome,
        "comparison_arm": comparison_arm,
        "unit": "task",
        "task_count": len(task_ids),
        "provider_rows_per_task": len(PROVIDERS),
        "seed": seed,
        "repetitions": repetitions,
        "comparison_minus_intact_percentage_points": observed,
        "percentile_95_interval_percentage_points": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
    }
