from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass
class ReplayFailureSlice:
    improved_tasks: List[str]
    regressed_tasks: List[str]
    unchanged_tasks: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze_replay_deltas(per_task_delta: Iterable[Dict[str, object]]) -> ReplayFailureSlice:
    improved: List[str] = []
    regressed: List[str] = []
    unchanged: List[str] = []
    for item in per_task_delta:
        task_id = str(item["task_id"])
        base = int(item["base_outcome"])
        candidate = int(item["candidate_outcome"])
        if candidate > base:
            improved.append(task_id)
        elif candidate < base:
            regressed.append(task_id)
        else:
            unchanged.append(task_id)
    return ReplayFailureSlice(
        improved_tasks=improved,
        regressed_tasks=regressed,
        unchanged_tasks=unchanged,
    )


def replay_slice_summary(slice_: ReplayFailureSlice) -> Dict[str, object]:
    return {
        "improved_count": len(slice_.improved_tasks),
        "regressed_count": len(slice_.regressed_tasks),
        "unchanged_count": len(slice_.unchanged_tasks),
        "slice": slice_.to_dict(),
    }
