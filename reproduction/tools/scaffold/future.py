from __future__ import annotations
import json, re
from typing import Any
from .controlled import _state_contract, _answer_semantics_contract
from .full_contract import FULL_ACTION_SCHEMA_V2 as FULL_ACTION_SCHEMA


ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "current_entity_ids",
        "current_interface_ids",
        "allowed_path_ids",
        "allowed_permission_ids",
    ],
    "properties": {
        key: {"type": "array", "items": {"type": "string"}}
        for key in (
            "current_entity_ids",
            "current_interface_ids",
            "allowed_path_ids",
            "allowed_permission_ids",
        )
    },
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_schema(action_type: str) -> dict[str, Any]:
    return next(
        schema
        for schema in FULL_ACTION_SCHEMA["oneOf"]
        if schema["properties"]["action_type"]["const"] == action_type
    )


def provider_request(
    provider: str, messages: list[dict[str, str]], config: dict[str, Any]
) -> dict[str, Any]:
    common = {
        "model": config[provider]["model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": config[provider]["max_tokens"],
        "stream": False,
    }
    if provider == "deepseek":
        return {
            **common,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "tools": [],
        }
    return {**common, "enable_thinking": False}


def stage1_request(
    provider: str, task: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    instruction = (
        "Use only the visible base task. Return exactly one JSON object with fields answer and carried_state. "
        "answer must follow ANSWER_SCHEMA. "
        + _state_contract("natural")
        + " Return JSON only."
    )
    user = (
        instruction
        + "\n\nVISIBLE_BASE_TASK:\n"
        + task["base_surface"]
        + "\n\nANSWER_FIELD_SEMANTICS:\n"
        + _answer_semantics_contract(ANSWER_SCHEMA)
        + "\n\nANSWER_SCHEMA:\n"
        + canonical(ANSWER_SCHEMA)
        + "\n\nThe future probe is not shown in this stage. Do not predict or encode a future action."
    )
    return provider_request(
        provider,
        [
            {
                "role": "system",
                "content": "Follow only supplied evidence. Return valid JSON. Do not use tools, retrieval, or unstated facts.",
            },
            {"role": "user", "content": user},
        ],
        config,
    )


def stage2_request(
    provider: str, carried_state: str, task: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    public_events = [
        {
            key: value
            for key, value in event.items()
            if key
            not in {
                "generation_evidence",
                "source_schema_version",
                "event_record_sha256",
            }
        }
        for event in task["future_events"]
    ]
    schema = action_schema(task["slot_contract"]["action_type"])
    instruction = (
        "Update CARRIED_STATE using the ordered FUTURE_EVENTS, then return exactly one JSON object with fields "
        "carried_state and action. The original base task and base answer are unavailable. "
        + _state_contract("natural")
        + " action must follow ACTION_SCHEMA exactly. Return JSON only."
    )
    user = (
        instruction
        + "\n\nCARRIED_STATE:\n"
        + carried_state
        + "\n\nFUTURE_EVENTS:\n"
        + canonical(public_events)
        + "\n\nREQUESTED_ACTION_TYPE:\n"
        + task["slot_contract"]["action_type"]
        + "\n\nACTION_SCHEMA:\n"
        + canonical(schema)
    )
    return provider_request(
        provider,
        [
            {
                "role": "system",
                "content": "Use only the carried state and future events. Return valid JSON without commentary.",
            },
            {"role": "user", "content": user},
        ],
        config,
    )


def response_content(body: str | None) -> dict[str, Any]:
    result = {
        "transport_json": False,
        "content": None,
        "content_json": False,
        "value": {},
        "errors": [],
    }
    if body is None:
        result["errors"].append("RESPONSE_BODY_MISSING")
        return result
    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        result["errors"].append("RESPONSE_ENVELOPE_NOT_JSON")
        return result
    result["transport_json"] = True
    try:
        content = envelope["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        result["errors"].append("MESSAGE_CONTENT_MISSING")
        return result
    if not isinstance(content, str):
        result["errors"].append("MESSAGE_CONTENT_NOT_STRING")
        return result
    result["content"] = content
    text = content.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.fullmatch(
            r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE
        )
        if not match:
            result["errors"].append("MODEL_CONTENT_NOT_JSON")
            return result
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            result["errors"].append("FENCED_CONTENT_NOT_JSON")
            return result
    if not isinstance(value, dict):
        result["errors"].append("MODEL_CONTENT_NOT_OBJECT")
        return result
    result.update(content_json=True, value=value)
    return result


def state_fact_set(state: dict[str, Any]) -> set[str]:
    facts = set()
    for entity in state.get("entities", []):
        eid = entity.get("entity_id")
        facts.add(canonical(["entity_type", eid, entity.get("entity_type")]))
        for key, value in sorted((entity.get("state") or {}).items()):
            facts.add(canonical(["state", eid, key, value]))
    for relation in state.get("relations", []):
        facts.add(
            canonical(
                [
                    "relation",
                    relation.get("source_id"),
                    relation.get("relation_type"),
                    relation.get("target_id"),
                    relation.get("direction"),
                ]
            )
        )
    for exclusion in state.get("exclusions", []):
        facts.add(
            canonical(
                ["exclusion", exclusion.get("entity_id"), exclusion.get("reason")]
            )
        )
    return facts


def state_safety(state: dict[str, Any]) -> bool:
    unsafe = set()
    for row in state.get("entities", []):
        values = row.get("state") or {}
        if (
            any(values.get(key) for key in ("revoked", "stale", "blocked", "protected"))
            or not values.get("active", False)
            or (row.get("entity_type") == "path" and not values.get("allowed", False))
        ):
            unsafe.add(row.get("entity_id"))
            if values.get("current"):
                return False
    for relation in state.get("relations", []):
        if relation.get("relation_type") == "replacement":
            if relation.get("target_id") in unsafe:
                return False
        elif relation.get("source_id") in unsafe or relation.get("target_id") in unsafe:
            return False
    return True
