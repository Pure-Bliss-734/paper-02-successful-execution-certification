from __future__ import annotations

ENV_STATE_SCHEMA_VERSION = "pcg_v3_m11.environment_state.v1"

ACTION_SCHEMA_VERSION = "pcg_v3_m11.action_envelope.v1"

ACTION_TYPES = (
    "apply_current_bundle",
    "select_current_interface",
    "migrate_to_allowed_path",
    "replace_current_object",
    "allow_or_deny_operation",
)
