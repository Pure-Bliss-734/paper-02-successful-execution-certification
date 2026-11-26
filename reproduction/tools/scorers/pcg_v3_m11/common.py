from __future__ import annotations

import copy

import hashlib

import json

from dataclasses import asdict, is_dataclass

from typing import Any


def normalized(value: Any) -> Any:
    if is_dataclass(value):
        return normalized(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): normalized(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [normalized(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def digest(value: Any) -> str:
    payload = json.dumps(
        normalized(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_hash(state: dict[str, Any]) -> str:
    return digest(state)


def clone(value: Any) -> Any:
    return copy.deepcopy(value)


def operational_view(state: dict[str, Any]) -> dict[str, Any]:
    mutable = clone(state.get("mutable_operational_state", {}))
    return {
        "objects": mutable.get("objects", {}),
        "paths": mutable.get("paths", {}),
        "permissions": mutable.get("permissions", {}),
        "replacement_relationships": mutable.get("replacement_relationships", []),
        "dependency_relationships": mutable.get("dependency_relationships", []),
        "operation_authorizations": mutable.get("operation_authorizations", []),
        "exclusion_records": mutable.get("exclusion_records", []),
    }


def get_path(container: dict[str, Any], dotted: str) -> Any:
    item: Any = container
    for part in dotted.split("."):
        item = item[part]
    return item


def set_path(container: dict[str, Any], dotted: str, value: Any) -> None:
    item: Any = container
    parts = dotted.split(".")
    for part in parts[:-1]:
        item = item[part]
    item[parts[-1]] = value


def flatten_operational(state: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                walk(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            flat[prefix] = normalized(value)
        else:
            flat[prefix] = value

    walk("", operational_view(state))
    return flat


def diff_operational(pre: dict[str, Any], post: dict[str, Any]) -> list[dict[str, Any]]:
    before = flatten_operational(pre)
    after = flatten_operational(post)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        if path not in before:
            rows.append({"op": "add", "path": path, "old": None, "new": after[path]})
        elif path not in after:
            rows.append(
                {"op": "remove", "path": path, "old": before[path], "new": None}
            )
        elif before[path] != after[path]:
            rows.append(
                {"op": "update", "path": path, "old": before[path], "new": after[path]}
            )
    return rows
