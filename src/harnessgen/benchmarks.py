from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .types import Task


FAILURE_MODES = (
    "ambiguous_goal",
    "trajectory_drift",
    "tool_misuse",
    "subgoal_stall",
    "answer_error",
    "clean",
)


@dataclass
class BenchmarkDataset:
    name: str
    tasks: List[Task]

    @classmethod
    def from_jsonl(cls, name: str, tasks_path: str) -> "BenchmarkDataset":
        tasks = _load_tasks_jsonl(tasks_path)
        return cls(name=name, tasks=tasks)


@dataclass
class SyntheticHarnessBenchmark:
    seed: int = 7

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def build_dataset(self, count: int = 48, name: str = "synthetic_harness_benchmark") -> BenchmarkDataset:
        return BenchmarkDataset(name=name, tasks=self.sample(count))

    def sample(self, count: int) -> List[Task]:
        tasks: List[Task] = []
        for idx in range(count):
            failure_mode = self.rng.choices(
                FAILURE_MODES,
                weights=[0.18, 0.18, 0.18, 0.16, 0.15, 0.15],
                k=1,
            )[0]
            difficulty = self.rng.uniform(0.2, 1.0)
            tasks.append(
                Task(
                    task_id=f"task-{idx}-{self.rng.randrange(10_000)}",
                    goal=f"Complete the benchmark task for failure mode {failure_mode}",
                    metadata={
                        "failure_mode": failure_mode,
                        "difficulty": difficulty,
                        "context_load": self.rng.randint(1, 10),
                        "tool_pressure": self.rng.randint(0, 4),
                        "ambiguity": 1 if failure_mode == "ambiguous_goal" else 0,
                        "requires_verification": 1 if failure_mode == "answer_error" else 0,
                    },
                    gold={
                        "expected_pass": failure_mode == "clean",
                        "failure_mode": failure_mode,
                    },
                )
            )
        return tasks

    def solved_subset(self, tasks: Iterable[Task]) -> List[Task]:
        return [task for task in tasks if task.metadata["failure_mode"] == "clean"]

    def task_scope(self, tasks: Iterable[Task]) -> Dict[str, object]:
        modes = [task.metadata["failure_mode"] for task in tasks]
        dominant = max(set(modes), key=modes.count) if modes else "clean"
        return {"dominant_failure_mode": dominant, "count": len(modes)}


def load_benchmark_dataset(
    benchmark_name: str,
    task_count: int = 48,
    seed: int = 7,
    tasks_path: Optional[str] = None,
) -> BenchmarkDataset:
    if benchmark_name == "synthetic":
        return SyntheticHarnessBenchmark(seed=seed).build_dataset(count=task_count)
    if benchmark_name == "jsonl":
        if not tasks_path:
            raise ValueError("tasks_path is required for benchmark_name='jsonl'")
        return BenchmarkDataset.from_jsonl(name=Path(tasks_path).stem, tasks_path=tasks_path)
    raise ValueError(f"Unknown benchmark_name: {benchmark_name}")


def _load_tasks_jsonl(path: str) -> List[Task]:
    rows: List[Task] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(Task.from_dict(json.loads(line)))
    return rows
