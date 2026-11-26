from __future__ import annotations
import math, random, hashlib
from typing import Any

RESAMPLES = 20000
BASE_SEED = 20260901


def exact_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(rescue, harm) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return (
        values[lo]
        if lo == hi
        else values[lo] * (hi - position) + values[hi] * (position - lo)
    )


def paired_summary(
    rows: list[dict[str, Any]], metric: str, old: bool = False
) -> dict[str, Any]:
    prefix = "historical_" if old else "current_"
    g = [bool(row[prefix + "G_" + metric]) for row in rows]
    d = [bool(row[prefix + "D_" + metric]) for row in rows]
    rescue = sum(
        not row[prefix + "G_" + metric] and row[prefix + "D_" + metric] for row in rows
    )
    harm = sum(
        row[prefix + "G_" + metric] and not row[prefix + "D_" + metric] for row in rows
    )
    rng = random.Random(
        int(
            hashlib.sha256(f"{BASE_SEED}:{prefix}:{metric}".encode()).hexdigest()[:16],
            16,
        )
    )
    bootstrap = []
    for _ in range(RESAMPLES):
        sample = [rng.choice(rows) for _ in rows]
        bootstrap.append(
            100
            * sum(
                int(x[prefix + "D_" + metric]) - int(x[prefix + "G_" + metric])
                for x in sample
            )
            / len(sample)
        )
    return {
        "pairs": len(rows),
        "G_count": sum(g),
        "D_count": sum(d),
        "delta_D_minus_G_pp": 100 * (sum(d) - sum(g)) / len(rows),
        "transitions": {
            "rescue_0_to_1": rescue,
            "harm_1_to_0": harm,
            "concordant": len(rows) - rescue - harm,
        },
        "exact_two_sided_mcnemar_p": exact_mcnemar(rescue, harm),
        "task_pair_bootstrap_95_ci_pp": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_base_seed": BASE_SEED,
    }
