from __future__ import annotations

from typing import Any

from pcg_v3.hashing import sha256_json

VALIDATOR_A_VERSION = "pcg_v3.external_terminal_validator_a.v1"

VALIDATOR_B_VERSION = "pcg_v3.external_terminal_validator_b.v1"


def validate_terminal_a(
    execution: dict[str, Any], scorer: dict[str, Any]
) -> dict[str, Any]:
    """Direct expected-tree and expected-path validator."""
    post = execution.get("post_state", {})
    failures: list[str] = []
    if execution.get("status") != "EXECUTED":
        failures.append("EXECUTION_NOT_COMPLETE")
    if post.get("actual_tree") != scorer["expected_terminal_tree"]:
        failures.append("TERMINAL_TREE_MISMATCH")
    if post.get("changed_paths") != scorer["expected_changed_paths"]:
        failures.append("CHANGED_PATH_SET_MISMATCH")
    if post.get("validation_rc") != 0:
        failures.append("AUXILIARY_VALIDATION_FAILED")
    result = {
        "validator_version": VALIDATOR_A_VERSION,
        "success": not failures,
        "failure_codes": failures,
        "terminal_tree": post.get("actual_tree"),
        "changed_paths": post.get("changed_paths"),
    }
    result["record_hash"] = sha256_json(result)
    return result


def validate_terminal_b(
    execution: dict[str, Any], scorer: dict[str, Any]
) -> dict[str, Any]:
    """Independent changed-blob and transaction-integrity validator."""
    post = execution.get("post_state", {})
    failures: list[str] = []
    if execution.get("status") != "EXECUTED":
        failures.append("EXECUTION_NOT_COMPLETE")
    if post.get("blob_map") != scorer["expected_blob_map"]:
        failures.append("CHANGED_BLOB_MANIFEST_MISMATCH")
    if execution.get("pre_state_hash") != scorer["expected_base_tree"]:
        failures.append("BASE_TREE_MISMATCH")
    if execution.get("network") != "none":
        failures.append("NETWORK_POLICY_VIOLATION")
    if post.get("head_after") != scorer["base_commit"]:
        failures.append("BASE_HEAD_CHANGED")
    result = {
        "validator_version": VALIDATOR_B_VERSION,
        "success": not failures,
        "failure_codes": failures,
        "blob_map": post.get("blob_map"),
        "network": execution.get("network"),
    }
    result["record_hash"] = sha256_json(result)
    return result
