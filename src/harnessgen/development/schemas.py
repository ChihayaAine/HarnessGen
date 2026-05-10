from __future__ import annotations

from copy import deepcopy
from typing import Dict


SCHEMA_LIBRARY: Dict[str, Dict[str, object]] = {
    "verifier": {
        "family": "verifier",
        "input_schema": ("plan", "action_result"),
        "output_schema": ("verification",),
        "side_effects": ("log",),
        "budget": {"latency_ms": 80},
        "default_params": {"strength": 0.8, "activation_threshold": 0.4},
        "insertion": "post-insert",
    },
    "constraint_resolver": {
        "family": "constraint_resolver",
        "input_schema": ("observation",),
        "output_schema": ("constraints", "assumptions"),
        "side_effects": ("assumption_record",),
        "budget": {"latency_ms": 60},
        "default_params": {"strength": 0.8, "activation_threshold": 0.3},
        "insertion": "pre-insert",
    },
    "state_compressor": {
        "family": "state_compressor",
        "input_schema": ("history", "observation"),
        "output_schema": ("compressed_state",),
        "side_effects": ("history_rewrite",),
        "budget": {"latency_ms": 55},
        "default_params": {"strength": 0.85, "activation_threshold": 0.5},
        "insertion": "pre-insert",
    },
    "tool_triage": {
        "family": "tool_triage",
        "input_schema": ("plan", "available_tools"),
        "output_schema": ("tool_decision",),
        "side_effects": ("tool_routing",),
        "budget": {"latency_ms": 45},
        "default_params": {"strength": 0.8, "activation_threshold": 0.4},
        "insertion": "guard-insert",
    },
    "recovery_handler": {
        "family": "recovery_handler",
        "input_schema": ("action_result", "history"),
        "output_schema": ("recovery_plan",),
        "side_effects": ("retry",),
        "budget": {"latency_ms": 75},
        "default_params": {"strength": 0.75, "activation_threshold": 0.4},
        "insertion": "post-insert",
    },
}


def schema_template(family: str) -> Dict[str, object]:
    return deepcopy(SCHEMA_LIBRARY[family])
