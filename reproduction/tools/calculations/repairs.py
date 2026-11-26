from __future__ import annotations
import math, random, hashlib
from collections import defaultdict
from typing import Any

BOOTSTRAP_RESAMPLES = 20000
BOOTSTRAP_BASE_SEED = 20260901


def exact_mcnemar_p(rescues: int, harms: int) -> float:
    discordant = rescues + harms
    if discordant == 0:
        return 1.0
    lower = min(rescues, harms)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_seed(label: str, metric: str) -> int:
    digest = hashlib.sha256(
        f"{BOOTSTRAP_BASE_SEED}:{label}:{metric}".encode()
    ).hexdigest()
    return int(digest[:16], 16)


def summarize(pairs: list[dict[str, Any]], metric: str, label: str) -> dict[str, Any]:
    arm_a = sum(pair["A"][metric] for pair in pairs)
    arm_b = sum(pair["B"][metric] for pair in pairs)
    transitions = {
        "0_to_0": sum(
            not pair["A"][metric] and not pair["B"][metric] for pair in pairs
        ),
        "0_to_1": sum(not pair["A"][metric] and pair["B"][metric] for pair in pairs),
        "1_to_0": sum(pair["A"][metric] and not pair["B"][metric] for pair in pairs),
        "1_to_1": sum(pair["A"][metric] and pair["B"][metric] for pair in pairs),
    }
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_task[pair["task_id"]].append(pair)
    tasks = sorted(by_task)
    rng = random.Random(bootstrap_seed(label, metric))
    bootstrap = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [rng.choice(tasks) for _ in tasks]
        rows = [pair for task in sampled for pair in by_task[task]]
        bootstrap.append(
            100.0
            * (
                sum(pair["B"][metric] for pair in rows)
                - sum(pair["A"][metric] for pair in rows)
            )
            / len(rows)
        )
    n = len(pairs)
    return {
        "metric": metric,
        "pairs": n,
        "tasks": len(tasks),
        "arm_A_count": arm_a,
        "arm_A_rate": arm_a / n,
        "arm_B_count": arm_b,
        "arm_B_rate": arm_b / n,
        "paired_difference_percentage_points": 100.0 * (arm_b - arm_a) / n,
        "transitions": transitions,
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(
            transitions["0_to_1"], transitions["1_to_0"]
        ),
        "task_cluster_bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "derived_seed": bootstrap_seed(label, metric),
            "percentile_method": "linear_interpolation_at_2.5_and_97.5_percent",
            "ci_level": 0.95,
            "lower_percentage_points": percentile(bootstrap, 0.025),
            "upper_percentage_points": percentile(bootstrap, 0.975),
        },
    }
