from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

from ..core.harness import Harness
from ..core.types import Task, Trajectory


@dataclass
class RecalibrationResult:
    harness: Harness
    score: float
    attempted_updates: List[Dict[str, float]]
    search_score: float = 0.0
    selection_score: float = 0.0


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
        search_tasks, selection_tasks, holdout_tasks = self._split(tasks)
        candidates = self._candidate_updates(cluster_failures)[: self.max_steps]
        best_harness = harness.copy()
        best_score = self._evaluate(best_harness, holdout_tasks)
        best_search_score = self._evaluate(best_harness, search_tasks)
        best_selection_score = self._evaluate(best_harness, selection_tasks)
        attempted: List[Dict[str, float]] = []
        for update in candidates:
            candidate = harness.copy()
            for module_name, params in update.items():
                candidate.modules[module_name].params.update(params)
            score = self._evaluate(candidate, search_tasks)
            selection_score = self._evaluate(candidate, selection_tasks)
            attempted.append(
                {
                    **{f"{module}.{key}": value for module, params in update.items() for key, value in params.items()},
                    "search_score": score,
                    "selection_score": selection_score,
                }
            )
            if (selection_score, score) >= (best_selection_score, best_search_score):
                best_search_score = score
                best_selection_score = selection_score
                best_harness = candidate
        holdout_score = self._evaluate(best_harness, holdout_tasks)
        return RecalibrationResult(
            harness=best_harness,
            score=holdout_score,
            attempted_updates=attempted,
            search_score=best_search_score,
            selection_score=best_selection_score,
        )

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
            {"plan": {"activation_threshold": 0.45}},
            {"act": {"activation_threshold": 0.45}},
        ]
        if dominant == "answer_error":
            updates.extend([{"plan": {"prompt_strength": 0.75}}, {"act": {"tool_policy_strength": 0.5}}, {"respond": {"verification_quality": 0.75}}])
        elif dominant == "tool_error":
            updates.extend([{"act": {"tool_policy_strength": 0.75}}, {"plan": {"prompt_strength": 0.5}}, {"act": {"activation_threshold": 0.35}}])
        elif dominant == "ambiguous_goal":
            updates.extend([{"plan": {"prompt_strength": 0.7}}, {"plan": {"activation_threshold": 0.0}}, {"observe": {"prompt_strength": 0.6}}])
        elif dominant == "subgoal_stall":
            updates.extend([{"act": {"tool_policy_strength": 0.7}}, {"respond": {"activation_threshold": 0.3}}])
        return updates
