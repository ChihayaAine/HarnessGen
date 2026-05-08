from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from .harness import Harness
from .types import Task, Trajectory


@dataclass
class RecalibrationResult:
    harness: Harness
    score: float
    attempted_updates: List[Dict[str, float]]


class Recalibrator:
    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def recalibrate(
        self,
        harness: Harness,
        cluster_failures: Sequence[Trajectory],
    ) -> RecalibrationResult:
        if not cluster_failures:
            return RecalibrationResult(harness=harness.copy(), score=0.0, attempted_updates=[])

        tasks = [item.task for item in cluster_failures]
        search_tasks, _, holdout_tasks = self._split(tasks)
        candidates = self._candidate_updates(cluster_failures)[: self.max_steps]
        best_harness = harness.copy()
        best_score = self._evaluate(best_harness, holdout_tasks)
        attempted: List[Dict[str, float]] = []
        for update in candidates:
            candidate = harness.copy()
            for module_name, params in update.items():
                candidate.modules[module_name].params.update(params)
            score = self._evaluate(candidate, search_tasks)
            attempted.append({f"{module}.{key}": value for module, params in update.items() for key, value in params.items()})
            if score >= best_score:
                best_score = score
                best_harness = candidate
        holdout_score = self._evaluate(best_harness, holdout_tasks)
        return RecalibrationResult(harness=best_harness, score=holdout_score, attempted_updates=attempted)

    def _split(self, tasks: Sequence[Task]) -> Tuple[List[Task], List[Task], List[Task]]:
        if len(tasks) < 4:
            return list(tasks), [], list(tasks)
        n = len(tasks)
        i = max(1, int(n * 0.5))
        j = max(i + 1, int(n * 0.75))
        return list(tasks[:i]), list(tasks[i:j]), list(tasks[j:])

    def _evaluate(self, harness: Harness, tasks: Iterable[Task]) -> float:
        task_list = list(tasks)
        if not task_list:
            return 0.0
        outcomes = [harness.execute(task).outcome for task in task_list]
        return float(mean(outcomes))

    def _candidate_updates(self, failures: Sequence[Trajectory]) -> List[Dict[str, Dict[str, float]]]:
        dominant = max(
            set(item.failure_type for item in failures if item.failure_type),
            key=[item.failure_type for item in failures if item.failure_type].count,
        )
        updates: List[Dict[str, Dict[str, float]]] = [
            {"plan": {"prompt_strength": 0.55}},
            {"plan": {"prompt_strength": 0.65}},
            {"act": {"tool_policy_strength": 0.55}},
            {"act": {"tool_policy_strength": 0.65}},
        ]
        if dominant == "answer_error":
            updates.extend([{"plan": {"prompt_strength": 0.75}}, {"act": {"tool_policy_strength": 0.5}}])
        elif dominant == "tool_error":
            updates.extend([{"act": {"tool_policy_strength": 0.75}}, {"plan": {"prompt_strength": 0.5}}])
        elif dominant == "ambiguous_goal":
            updates.extend([{"plan": {"prompt_strength": 0.7}}, {"plan": {"activation_threshold": 0.0}}])
        return updates
