from __future__ import annotations
import math, random, collections
from typing import Any


def bootstrap_task_cluster(
    rows: list[dict[str, Any]], statistic: str, replicates: int = 10_000
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    tasks = sorted(grouped)
    rng = random.Random(20260723)

    def value(sample: list[dict[str, Any]]) -> float:
        if statistic == "E1_R0_rate":
            return sum(row["E"] and not row["R"] for row in sample) / len(sample)
        if statistic == "R1_given_E1":
            eligible = [row for row in sample if row["E"]]
            return (
                sum(row["R"] for row in eligible) / len(eligible)
                if eligible
                else math.nan
            )
        raise ValueError(statistic)

    point = value(rows)
    draws = []
    for _ in range(replicates):
        sample = []
        for task in (rng.choice(tasks) for _ in tasks):
            sample.extend(grouped[task])
        candidate = value(sample)
        if not math.isnan(candidate):
            draws.append(candidate)
    draws.sort()

    def quantile(p: float) -> float:
        position = p * (len(draws) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return draws[lower]
        return draws[lower] * (upper - position) + draws[upper] * (position - lower)

    return {
        "statistic": statistic,
        "point": point,
        "task_count": len(tasks),
        "replicates": replicates,
        "seed": 20260723,
        "percentile_95_ci": [quantile(0.025), quantile(0.975)],
    }
