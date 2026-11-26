from __future__ import annotations
import json
from typing import Any
from .controlled import _state_contract
from .ablation_contract import FULL_ACTION_SCHEMA


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
