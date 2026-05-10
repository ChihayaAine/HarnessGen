from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Task:
    task_id: str
    goal: str
    metadata: Dict[str, Any]
    gold: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Task":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleEvent:
    name: str
    family: str
    activated: bool
    latency_ms: float
    success: bool
    error_code: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    activation_score: float = 0.0
    threshold: float = 0.0
    missing_inputs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    task: Task
    steps: List[ModuleEvent]
    state_snapshots: List[Dict[str, Any]]
    outcome: int
    failure_type: Optional[str]
    final_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureCluster:
    cluster_id: int
    member_indices: List[int]
    center: List[float]
    stable_cycles: int = 1
    signature: Dict[str, Any] = field(default_factory=dict)
    lineage_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayResult:
    utility: float
    regression_rate: float
    accepted: bool
    details: Dict[str, Any] = field(default_factory=dict)
    decision_trace: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
