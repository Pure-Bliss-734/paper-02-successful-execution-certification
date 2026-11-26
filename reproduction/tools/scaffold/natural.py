from __future__ import annotations
import json, copy
from typing import Any


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


SYSTEM_PROMPT = (
    "Return exactly one JSON action object. The entire response must be that object, "
    "not a wrapper, explanation, or Markdown block. Use only facts in the supplied state. "
    "Do not invent identifiers. Every schema field is required and extra fields are forbidden."
)

PROVIDER_CONFIGS = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "max_tokens": 4096,
        "temperature": 0.0,
        "thinking": {"type": "disabled"},
        "tools": [],
        "stream": False,
        "response_format": {"type": "json_object"},
        "route_policy": "direct_provider_no_fallback",
    },
    "qwen": {
        "model": "qwen3.7-max-2026-06-08",
        "max_tokens": 4096,
        "temperature": 0.0,
        "enable_thinking": False,
        "tools": [],
        "stream": False,
        "route_policy": "direct_provider_no_fallback",
    },
    "glm": {
        "model": "glm-5.2",
        "max_tokens": 4096,
        "temperature": 0.0,
        "enable_thinking": False,
        "reasoning_effort": "none",
        "tools": [],
        "stream": False,
        "route_policy": "direct_provider_no_fallback",
    },
}


def build_request(candidate: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in {"A", "B"}:
        raise ValueError("arm must be A or B")
    state = candidate["baseline_state"] if arm == "A" else candidate["repaired_state"]
    if "archived_stage2_request" in candidate:
        request = copy.deepcopy(candidate["archived_stage2_request"])
        if arm == "A":
            return request
        baseline_text = canonical_json(candidate["baseline_state"])
        repaired_text = canonical_json(candidate["repaired_state"])
        user = request["messages"][1]["content"]
        if user.count(baseline_text) != 1:
            raise AssertionError(
                "archived Stage-2 request does not contain one exact baseline state"
            )
        request["messages"][1]["content"] = user.replace(
            baseline_text, repaired_text, 1
        )
        return request
    action_type = candidate["action_type"]
    schema_text = canonical_json(candidate["action_schema"])
    # Both arms use the same renderer.  This controls the representation for
    # original line-oriented states while preserving exact state semantics.
    state_text = canonical_json(state)
    user = (
        f"Requested action type:\n{action_type}\n\n"
        f"Complete action-specific JSON schema:\n{schema_text}\n\n"
        f"Carried state (the only operational evidence):\n{state_text}\n\n"
        "Return the JSON action object now."
    )
    request = copy.deepcopy(PROVIDER_CONFIGS[candidate["provider"]])
    request["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return request
