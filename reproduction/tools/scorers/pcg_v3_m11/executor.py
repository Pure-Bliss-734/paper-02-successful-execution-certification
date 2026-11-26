from __future__ import annotations

from typing import Any

from .common import clone, digest, diff_operational, get_path, set_path, state_hash

from .schemas import ACTION_SCHEMA_VERSION, ACTION_TYPES, ENV_STATE_SCHEMA_VERSION

EXECUTOR_VERSION = "pcg_v3_m11.transaction_executor.v1"


def execute_action(
    pre_state: dict[str, Any],
    action: dict[str, Any],
    visible_context: dict[str, Any] | None = None,
    transaction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del visible_context, transaction_config
    pre = clone(pre_state)
    staged = clone(pre_state)
    pre_hash = state_hash(pre)
    txid = str(action.get("transaction_id", ""))
    log_builder = _LogBuilder(txid, pre_hash)
    failure = _schema_failure(action)
    if failure is not None:
        log_builder.fail(failure, "schema_or_type_validation_failed")
        return _result(
            pre, staged, action, False, failure, pre_hash, log_builder.entries
        )
    try:
        status = _dispatch(staged, action, log_builder)
        success = status in {"success", "noop"}
        failure_code = None if success else status
    except _PreconditionFailure as exc:
        staged = clone(pre_state)
        log_builder.fail(exc.code, exc.detail)
        success = False
        failure_code = exc.code
    return _result(
        pre, staged, action, success, failure_code, pre_hash, log_builder.entries
    )


class _PreconditionFailure(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


class _LogBuilder:
    def __init__(self, transaction_id: str, pre_state_hash: str) -> None:
        self.transaction_id = transaction_id
        self.pre_state_hash = pre_state_hash
        self.entries: list[dict[str, Any]] = []
        self.prev_hash = pre_state_hash

    def add(self, op: str, path: str, old: Any, new: Any) -> None:
        row = {
            "sequence": len(self.entries) + 1,
            "transaction_id": self.transaction_id,
            "op": op,
            "path": path,
            "old": old,
            "new": new,
            "prev_hash": self.prev_hash,
        }
        row["entry_hash"] = digest(row)
        self.prev_hash = row["entry_hash"]
        self.entries.append(row)

    def fail(self, code: str, detail: str) -> None:
        self.add(
            "failed_attempt",
            "transaction.failure",
            None,
            {"code": code, "detail": detail},
        )

    def noop(self, detail: str) -> None:
        self.add("noop", "transaction.noop", None, detail)


def _schema_failure(action: dict[str, Any]) -> str | None:
    if action.get("action_schema_version") != ACTION_SCHEMA_VERSION:
        return "ACTION_SCHEMA_INVALID"
    if action.get("action_type") not in ACTION_TYPES:
        return "ACTION_TYPE_UNSUPPORTED"
    if not action.get("transaction_id"):
        return "ACTION_SCHEMA_INVALID"
    return None


def _dispatch(state: dict[str, Any], action: dict[str, Any], log: _LogBuilder) -> str:
    action_type = action["action_type"]
    _check_runtime_state(state)
    if _has_conflict(state):
        raise _PreconditionFailure(
            "DEPENDENCY_CONFLICT", "current conflicting state present"
        )
    if action_type == "apply_current_bundle":
        return _apply_current_bundle(state, action, log)
    if action_type == "select_current_interface":
        return _select_current_interface(state, action, log)
    if action_type == "migrate_to_allowed_path":
        return _migrate_to_allowed_path(state, action, log)
    if action_type == "replace_current_object":
        return _replace_current_object(state, action, log)
    if action_type == "allow_or_deny_operation":
        return _allow_or_deny_operation(state, action, log)
    raise _PreconditionFailure("ACTION_TYPE_UNSUPPORTED", action_type)


def _check_runtime_state(state: dict[str, Any]) -> None:
    if state.get("environment_schema_version") != ENV_STATE_SCHEMA_VERSION:
        raise _PreconditionFailure(
            "ACTION_SCHEMA_INVALID", "environment schema mismatch"
        )
    forbidden = {
        "expected_terminal_state",
        "expected_delta",
        "gold_action",
        "success_label",
        "required_state_annotations",
    }
    leaked = forbidden & set(state)
    if leaked:
        raise _PreconditionFailure(
            "ACTION_PRECONDITION_FAILED", f"oracle leakage fields: {sorted(leaked)}"
        )


def _objects(state: dict[str, Any]) -> dict[str, Any]:
    return state["mutable_operational_state"]["objects"]


def _paths(state: dict[str, Any]) -> dict[str, Any]:
    return state["mutable_operational_state"]["paths"]


def _dependencies(state: dict[str, Any]) -> list[dict[str, Any]]:
    return state["mutable_operational_state"]["dependency_relationships"]


def _replacements(state: dict[str, Any]) -> list[dict[str, Any]]:
    return state["mutable_operational_state"]["replacement_relationships"]


def _permissions(state: dict[str, Any]) -> dict[str, Any]:
    return state["mutable_operational_state"]["permissions"]


def _operation_authorizations(state: dict[str, Any]) -> list[dict[str, Any]]:
    return state["mutable_operational_state"]["operation_authorizations"]


def _object(state: dict[str, Any], object_id: str) -> dict[str, Any]:
    if object_id not in _objects(state):
        raise _PreconditionFailure("TARGET_NOT_FOUND", object_id)
    return _objects(state)[object_id]


def _valid_current_object(state: dict[str, Any], object_id: str) -> dict[str, Any]:
    obj = _object(state, object_id)
    if not obj.get("current"):
        raise _PreconditionFailure("TARGET_NOT_CURRENT", object_id)
    _check_valid_flags(obj, object_id)
    return obj


def _check_valid_flags(obj: dict[str, Any], label: str) -> None:
    if obj.get("revoked"):
        raise _PreconditionFailure("TARGET_REVOKED", label)
    if obj.get("stale"):
        raise _PreconditionFailure("TARGET_STALE", label)
    if obj.get("blocked"):
        raise _PreconditionFailure("TARGET_BLOCKED", label)
    if obj.get("forbidden"):
        raise _PreconditionFailure("TARGET_FORBIDDEN", label)
    if obj.get("protected"):
        raise _PreconditionFailure("TARGET_PROTECTED", label)
    if not obj.get("active"):
        raise _PreconditionFailure("NO_VALID_TARGET", label)


def _check_path(state: dict[str, Any], path_id: str) -> None:
    path = _paths(state).get(path_id)
    if path is None:
        raise _PreconditionFailure("TARGET_NOT_FOUND", path_id)
    if path.get("forbidden"):
        raise _PreconditionFailure("PATH_FORBIDDEN", path_id)
    if path.get("blocked") or path.get("revoked") or not path.get("allowed"):
        raise _PreconditionFailure("PATH_NOT_ALLOWED", path_id)


def _check_dependencies(
    state: dict[str, Any], source: str, declared: list[str]
) -> None:
    for dep in declared:
        target = _object(state, dep)
        _check_valid_flags(target, dep)
        exists = any(
            edge.get("source") == source
            and edge.get("target") == dep
            and edge.get("active")
            for edge in _dependencies(state)
        )
        if not exists:
            raise _PreconditionFailure("DEPENDENCY_MISSING", f"{source}->{dep}")


def _has_conflict(state: dict[str, Any]) -> bool:
    current = {
        object_id
        for object_id, obj in _objects(state).items()
        if obj.get("active") and obj.get("current")
    }
    return any(
        obj.get("active")
        and obj.get("current")
        and obj.get("conflicts_with") in current
        for obj in _objects(state).values()
    )


def _set(state: dict[str, Any], log: _LogBuilder, path: str, value: Any) -> None:
    old = get_path(state, path)
    if old == value:
        return
    set_path(state, path, value)
    log.add("set", path, old, value)


def _append_audit(
    state: dict[str, Any], log: _LogBuilder, item: dict[str, Any]
) -> None:
    path = "action_history"
    old = clone(state[path])
    state[path] = old + [item]
    log.add("append", path, old, state[path])


def _apply_current_bundle(
    state: dict[str, Any], action: dict[str, Any], log: _LogBuilder
) -> str:
    targets = list(action["primary_target"])
    if not targets:
        raise _PreconditionFailure("NO_VALID_TARGET", "empty bundle")
    for target in targets:
        _valid_current_object(state, target)
        _check_dependencies(
            state, target, list(action.get("declared_dependencies", []))
        )
    if all(_objects(state)[target].get("applied") for target in targets):
        log.noop("bundle_already_applied")
        return "noop"
    for target in targets:
        _set(state, log, f"mutable_operational_state.objects.{target}.applied", True)
    _append_audit(
        state,
        log,
        {
            "transaction_id": action["transaction_id"],
            "action_type": action["action_type"],
            "status": "success",
        },
    )
    return "success"


def _select_current_interface(
    state: dict[str, Any], action: dict[str, Any], log: _LogBuilder
) -> str:
    interface_id = action.get("optional_interface") or action["primary_target"]
    obj = _valid_current_object(state, interface_id)
    if not obj.get("current_interface"):
        raise _PreconditionFailure("TARGET_NOT_CURRENT", interface_id)
    candidates = [
        item
        for item in _objects(state).values()
        if item.get("object_type") == obj.get("object_type")
        and item.get("current_interface")
        and item.get("active")
        and not item.get("revoked")
        and not item.get("stale")
        and not item.get("blocked")
        and not item.get("forbidden")
    ]
    if len(candidates) > 1:
        raise _PreconditionFailure("MULTIPLE_VALID_TARGETS", interface_id)
    if obj.get("selected"):
        log.noop("interface_already_selected")
        return "noop"
    _set(state, log, f"mutable_operational_state.objects.{interface_id}.selected", True)
    _append_audit(
        state,
        log,
        {
            "transaction_id": action["transaction_id"],
            "action_type": action["action_type"],
            "status": "success",
        },
    )
    return "success"


def _migrate_to_allowed_path(
    state: dict[str, Any], action: dict[str, Any], log: _LogBuilder
) -> str:
    target = action["primary_target"]
    path_id = action["optional_path"]
    obj = _valid_current_object(state, target)
    _check_path(state, path_id)
    _check_dependencies(state, target, list(action.get("declared_dependencies", [])))
    if obj.get("path_id") == path_id:
        log.noop("target_already_on_allowed_path")
        return "noop"
    _set(state, log, f"mutable_operational_state.objects.{target}.path_id", path_id)
    _append_audit(
        state,
        log,
        {
            "transaction_id": action["transaction_id"],
            "action_type": action["action_type"],
            "status": "success",
        },
    )
    return "success"


def _replace_current_object(
    state: dict[str, Any], action: dict[str, Any], log: _LogBuilder
) -> str:
    old_id = action["primary_target"]
    new_id = action["optional_replacement_target"]
    old = _object(state, old_id)
    new = _object(state, new_id)
    if not old.get("current") and new.get("current"):
        log.noop("replacement_already_applied")
        return "noop"
    _valid_current_object(state, old_id)
    _check_valid_flags(new, new_id)
    if not any(
        edge.get("source") == old_id
        and edge.get("target") == new_id
        and edge.get("active")
        for edge in _replacements(state)
    ):
        raise _PreconditionFailure("REPLACEMENT_NOT_SUPPORTED", f"{old_id}->{new_id}")
    _check_dependencies(state, old_id, list(action.get("declared_dependencies", [])))
    _set(state, log, f"mutable_operational_state.objects.{old_id}.current", False)
    _set(state, log, f"mutable_operational_state.objects.{old_id}.active", False)
    _set(
        state, log, f"mutable_operational_state.objects.{old_id}.superseded_by", new_id
    )
    _set(state, log, f"mutable_operational_state.objects.{new_id}.current", True)
    _set(state, log, f"mutable_operational_state.objects.{new_id}.active", True)
    _append_audit(
        state,
        log,
        {
            "transaction_id": action["transaction_id"],
            "action_type": action["action_type"],
            "status": "success",
        },
    )
    return "success"


def _allow_or_deny_operation(
    state: dict[str, Any], action: dict[str, Any], log: _LogBuilder
) -> str:
    actor = action["optional_actor_or_role"]
    decision = action["optional_decision"]
    params = action.get("action_parameters", {})
    operation = params.get("operation")
    target = action["primary_target"]
    permissions = _permissions(state).get(actor, {}).get(operation, {}).get(target)
    if permissions is None:
        raise _PreconditionFailure(
            "PERMISSION_MISSING", f"{actor}:{operation}:{target}"
        )
    if permissions.get("revoked"):
        raise _PreconditionFailure(
            "PERMISSION_REVOKED", f"{actor}:{operation}:{target}"
        )
    authorized = any(
        item.get("actor") == actor
        and item.get("operation") == operation
        and item.get("target") == target
        and item.get("decision") == decision
        and item.get("active")
        for item in _operation_authorizations(state)
    )
    if not authorized:
        raise _PreconditionFailure(
            "ACTION_PRECONDITION_FAILED", "decision_not_supported_by_policy"
        )
    _append_audit(
        state,
        log,
        {
            "transaction_id": action["transaction_id"],
            "action_type": action["action_type"],
            "decision": decision,
            "status": "success",
        },
    )
    return "success"


def _result(
    pre: dict[str, Any],
    post: dict[str, Any],
    action: dict[str, Any],
    success: bool,
    failure_code: str | None,
    pre_hash: str,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    post_hash = state_hash(post)
    return {
        "schema_version": "pcg_v3_m11.execution_result.v1",
        "executor_version": EXECUTOR_VERSION,
        "execution_status": "success" if success else "failed",
        "structured_failure_reason": failure_code,
        "pre_state_hash": pre_hash,
        "post_state": post,
        "post_state_hash": post_hash,
        "transaction_log": log,
        "actual_state_delta": diff_operational(pre, post),
        "mutated_object_list": _mutated_objects(diff_operational(pre, post)),
        "preserved_object_list": sorted(
            set(pre["mutable_operational_state"]["objects"])
            - set(_mutated_objects(diff_operational(pre, post)))
        ),
        "permission_and_path_checks": _path_permission_summary(action),
        "side_effect_report": {"forbidden_side_effect_detected": False},
        "environment_version": post.get("environment_version"),
        "transaction_id": action.get("transaction_id"),
        "result_hash": digest(
            {
                "success": success,
                "failure": failure_code,
                "post_hash": post_hash,
                "log": log,
            }
        ),
    }


def _mutated_objects(delta: list[dict[str, Any]]) -> list[str]:
    objects = set()
    for row in delta:
        if row["path"].startswith("objects."):
            objects.add(row["path"].split(".")[1])
        elif row["path"].startswith("mutable_operational_state.objects."):
            objects.add(row["path"].split(".")[2])
    return sorted(objects)


def _path_permission_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_checked": action.get("optional_path") is not None,
        "permission_checked": action.get("optional_actor_or_role") is not None,
    }
