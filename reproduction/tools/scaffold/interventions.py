"""Materialize the four supplied-state arms and single-bundle repair using frozen method code."""

from . import variants, repair


def deletion_arms(inputs, tasks):
    assembled = {}
    for task_id, t in inputs.items():
        base = t["base"]
        target = t["action"]
        task = tasks[task_id]
        wrong, _ = variants.wrong_state(base, task, target)
        deleted, _ = variants.delete_necessary(base, task, target)
        unrelated, _ = variants.unrelated_state(base, task, target)
        arms = {
            "intact_gold_state": base,
            "necessary_state_deletion": deleted,
            "counterfactual_wrong_state": wrong,
            "unrelated_state_perturbation": unrelated,
        }
        assembled[task_id] = {arm: variants.state_text(s) for arm, s in arms.items()}
    return assembled


def repair_arm(spec, target):
    bundles, baseline = repair.enumerate_defective_bundles(
        spec["baseline_state"], target, spec["expected_action"], parse_success=True
    )
    if baseline["R"] or not bundles:
        raise ValueError("Candidate has no eligible support defect")
    return repair.apply_bundle_repair(
        spec["baseline_state"], target, bundles[sorted(bundles)[0]]
    )
