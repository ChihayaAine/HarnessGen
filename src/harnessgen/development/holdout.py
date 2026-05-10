from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from ..core.types import Task


@dataclass
class HoldoutSlice:
    solved_tasks: List[Task]
    stress_tasks: List[Task]

    def to_dict(self) -> Dict[str, object]:
        return {
            "solved_task_ids": [task.task_id for task in self.solved_tasks],
            "stress_task_ids": [task.task_id for task in self.stress_tasks],
        }


class HoldoutManager:
    def build(self, tasks: Iterable[Task]) -> HoldoutSlice:
        task_list = list(tasks)
        solved = [task for task in task_list if task.metadata.get("failure_mode") == "clean"]
        stress = [
            task
            for task in task_list
            if task.metadata.get("failure_mode") in {"answer_error", "tool_misuse", "trajectory_drift"}
        ]
        return HoldoutSlice(solved_tasks=solved, stress_tasks=stress[: max(1, len(stress) // 2)] if stress else [])

    def summary(self, slice_: HoldoutSlice) -> Dict[str, object]:
        return {
            "solved_count": len(slice_.solved_tasks),
            "stress_count": len(slice_.stress_tasks),
            "payload": slice_.to_dict(),
        }
