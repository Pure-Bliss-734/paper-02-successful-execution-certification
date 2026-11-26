# Manuscript Table 16

The frozen deletion experiment contains 20 tasks, three providers per task and four supplied-state arms. The last two rows of Table 16 report the paired step-2 action effect by whether a later event names the deleted entity. The two action effects printed in the manuscript differ from the saved records.

| Later event names the entity | Pairs | Endpoint | Intact successes | Deletion successes | Recalculated difference (pp) | Manuscript (pp) | Cells (00, 01, 10, 11) |
|---|---:|---|---:|---:|---:|---:|---|
| Yes | 24 | Correct action (M) | 8 | 1 | −29.17 | −62.50 | 16, 0, 7, 1 |
| No | 36 | Correct action (M) | 32 | 0 | −88.89 | −66.67 | 4, 0, 32, 0 |
| Yes | 24 | Successful outcome (E) | 8 | 1 | −29.17 | −29.17 | 16, 0, 7, 1 |
| No | 36 | Successful outcome (E) | 32 | 32 | 0.00 | 0.00 | 4, 0, 0, 32 |

Each cell lists the outcome on the intact arm first and on the deletion arm second. The difference is 100 times the paired count difference divided by the number of pairs. In the full 60-pair sample the action difference is −65.00 percentage points and the outcome difference is −11.67 percentage points, as printed. Both values for the 48 pairs remaining after excluding primary-target deletions also agree.

The group can be derived from the deleted entity and future event fields in `data/deletion_groups.jsonl`. The arm-level responses and measurements are in `evidence/calls/stage2_ablation_240.jsonl` and `data/intervention.jsonl`. The numerical checker reports these two printed cells as discrepancies.
