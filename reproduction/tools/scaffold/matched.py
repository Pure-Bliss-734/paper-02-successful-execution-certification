from __future__ import annotations
import json
from typing import Any
from .full_contract import FULL_ACTION_SCHEMA_V2 as FULL_ACTION_SCHEMA


def canonical_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_schema(action_type: str) -> dict[str, Any]:
    for schema in FULL_ACTION_SCHEMA["oneOf"]:
        if schema["properties"]["action_type"]["const"] == action_type:
            return schema
    raise ValueError(action_type)


def prompt_messages(state: dict[str, Any], action_type: str) -> list[dict[str, str]]:
    schema = action_schema(action_type)
    system = (
        "Return exactly one JSON action object. The entire response must be that object, "
        "not a wrapper, explanation, or Markdown block. Use only facts in the supplied state. "
        "Do not invent identifiers. Every schema field is required and extra fields are forbidden."
    )
    user = (
        "Requested action type:\n"
        + action_type
        + "\n\nComplete action-specific JSON schema:\n"
        + canonical_line(schema)
        + "\n\nCarried state (the only operational evidence):\n"
        + canonical_line(state)
        + "\n\nReturn the JSON action object now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
