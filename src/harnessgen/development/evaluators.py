from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from ..core.types import ReplayResult


@dataclass
class FamilyEvaluation:
    family: str
    adjusted_utility: float
    regression_penalty: float
    latency_penalty: float
    accepted: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_family_result(
    family: str,
    replay_result: ReplayResult,
    utility_threshold: float,
    regression_threshold: float,
) -> FamilyEvaluation:
    latency_penalty = 0.0
    details = replay_result.details
    if isinstance(details, dict):
        latency_penalty = max(0.0, float(details.get("latency_delta_ms", 0.0))) * 0.0005
    regression_penalty = replay_result.regression_rate
    family_bonus = {
        "verifier": 0.015,
        "tool_triage": 0.010,
        "state_compressor": 0.008,
        "constraint_resolver": 0.010,
        "recovery_handler": 0.006,
    }.get(family, 0.0)
    adjusted_utility = replay_result.utility + family_bonus - latency_penalty - 0.05 * regression_penalty
    accepted = adjusted_utility > utility_threshold and replay_result.regression_rate <= regression_threshold
    return FamilyEvaluation(
        family=family,
        adjusted_utility=adjusted_utility,
        regression_penalty=regression_penalty,
        latency_penalty=latency_penalty,
        accepted=accepted,
    )
