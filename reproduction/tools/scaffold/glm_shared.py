from __future__ import annotations
import json
from typing import Any
from .full_contract import FULL_ACTION_SCHEMA_V2
from .controlled import _state_contract, _answer_semantics_contract


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
    for schema in FULL_ACTION_SCHEMA_V2["oneOf"]:
        if schema["properties"]["action_type"]["const"] == action_type:
            return schema
    raise ValueError(f"unsupported action type: {action_type}")


def full_action_messages(
    state: dict[str, Any], action_type: str
) -> list[dict[str, str]]:
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
        + canonical(schema)
        + "\n\nCarried state (the only operational evidence):\n"
        + canonical(state)
        + "\n\nReturn the JSON action object now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def future_stage1_messages(task: dict[str, Any]) -> list[dict[str, str]]:
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
    return [
        {
            "role": "system",
            "content": "Follow only supplied evidence. Return valid JSON. Do not use tools, retrieval, or unstated facts.",
        },
        {"role": "user", "content": user},
    ]


def future_stage2_messages(
    carried_state: str, task: dict[str, Any]
) -> list[dict[str, str]]:
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
    kind = task["slot_contract"]["action_type"]
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
        + kind
        + "\n\nACTION_SCHEMA:\n"
        + canonical(action_schema(kind))
    )
    return [
        {
            "role": "system",
            "content": "Use only the carried state and future events. Return valid JSON without commentary.",
        },
        {"role": "user", "content": user},
    ]
