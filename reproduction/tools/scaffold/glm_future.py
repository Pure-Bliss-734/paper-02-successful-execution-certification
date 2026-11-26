from __future__ import annotations
from typing import Any
from . import glm_shared as base


_base_future_stage1_messages = base.future_stage1_messages

_base_future_stage2_messages = base.future_stage2_messages


def future_stage1_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    messages = _base_future_stage1_messages(task)
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "carried_state"],
        "properties": {
            "answer": base.ANSWER_SCHEMA,
            "carried_state": {"type": "string"},
        },
    }
    messages[-1]["content"] += (
        "\n\nTOP_LEVEL_OUTPUT_SCHEMA (binding):\n"
        + base.canonical(output_schema)
        + "\ncarried_state must be one JSON string containing newline-separated ENTITY, RELATION, and EXCLUSION lines. "
        "It must not be an array."
    )
    return messages


def future_stage2_messages(
    carried_state: str, task: dict[str, Any]
) -> list[dict[str, str]]:
    messages = _base_future_stage2_messages(carried_state, task)
    kind = task["slot_contract"]["action_type"]
    output_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["carried_state", "action"],
        "properties": {
            "carried_state": {"type": "string"},
            "action": base.action_schema(kind),
        },
    }
    messages[-1]["content"] += (
        "\n\nTOP_LEVEL_OUTPUT_SCHEMA (binding):\n"
        + base.canonical(output_schema)
        + "\ncarried_state must remain one JSON string, not an array."
    )
    return messages
