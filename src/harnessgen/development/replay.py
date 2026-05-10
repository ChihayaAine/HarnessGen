from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Dict, Iterable, List

from ..core.harness import Harness
from ..core.types import Task


@dataclass
class ReplayDiagnostics:
    utility: float
    regression_rate: float
    latency_delta_ms: float
    base_pass_rate: float
    candidate_pass_rate: float
    replay_count: int
    solved_count: int
    per_task_delta: List[Dict[str, object]] = None
    family_breakdown: Dict[str, Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _trajectory_score(harness: Harness, tasks: List[Task], seed: int) -> tuple[List[int], List[float]]:
    outcomes: List[int] = []
    latencies: List[float] = []
    for task in tasks:
        trajectory = harness.execute(task, seed=seed)
        outcomes.append(trajectory.outcome)
        latencies.append(sum(step.latency_ms for step in trajectory.steps))
    return outcomes, latencies


def _family_breakdown(harness: Harness, tasks: List[Task], seed: int) -> Dict[str, Dict[str, float]]:
    breakdown: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        trajectory = harness.execute(task, seed=seed)
        for step in trajectory.steps:
            slot = breakdown.setdefault(step.family, {"activations": 0.0, "failures": 0.0, "latency_ms": 0.0})
            if step.activated:
                slot["activations"] += 1.0
            if not step.success:
                slot["failures"] += 1.0
            slot["latency_ms"] += float(step.latency_ms)
    return breakdown


def validate_replay(
    base_harness: Harness,
    candidate_harness: Harness,
    replay_tasks: Iterable[Task],
    solved_tasks: Iterable[Task],
    latency_penalty: float,
) -> ReplayDiagnostics:
    replay_list = list(replay_tasks)
    solved_list = list(solved_tasks)
    if not replay_list:
        return ReplayDiagnostics(0.0, 0.0, 0.0, 0.0, 0.0, 0, len(solved_list))
    base_scores, base_latencies = _trajectory_score(base_harness, replay_list, seed=17)
    cand_scores, cand_latencies = _trajectory_score(candidate_harness, replay_list, seed=17)
    score_delta = [cand - base for cand, base in zip(cand_scores, base_scores)]
    latency_delta = [cand - base for cand, base in zip(cand_latencies, base_latencies)]
    per_task_delta = [
        {
            "task_id": task.task_id,
            "base_outcome": base,
            "candidate_outcome": cand,
            "base_latency_ms": base_latency,
            "candidate_latency_ms": cand_latency,
            "latency_delta_ms": cand_latency - base_latency,
        }
        for task, base, cand, base_latency, cand_latency in zip(
            replay_list, base_scores, cand_scores, base_latencies, cand_latencies
        )
    ]
    regressions = 0
    for task in solved_list:
        if base_harness.execute(task, seed=17).outcome == 1 and candidate_harness.execute(task, seed=17).outcome == 0:
            regressions += 1
    utility = float(mean(score_delta)) - latency_penalty * float(mean(latency_delta))
    return ReplayDiagnostics(
        utility=utility,
        regression_rate=regressions / max(1, len(solved_list)),
        latency_delta_ms=float(mean(latency_delta)),
        base_pass_rate=float(mean(base_scores)),
        candidate_pass_rate=float(mean(cand_scores)),
        replay_count=len(replay_list),
        solved_count=len(solved_list),
        per_task_delta=per_task_delta,
        family_breakdown={
            "base": _family_breakdown(base_harness, replay_list, seed=17),
            "candidate": _family_breakdown(candidate_harness, replay_list, seed=17),
        },
    )
