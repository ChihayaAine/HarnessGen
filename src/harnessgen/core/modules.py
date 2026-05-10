from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from .types import Task

Executor = Callable[[Dict[str, Any], Task, "ModuleSpec"], Dict[str, Any]]
ActivationFn = Callable[[Dict[str, Any], Task, "ModuleSpec"], float]


@dataclass
class ModuleSpec:
    name: str
    family: str
    input_schema: Tuple[str, ...]
    output_schema: Tuple[str, ...]
    executor: Executor
    side_effects: Tuple[str, ...] = ()
    budget: Dict[str, Any] = field(default_factory=dict)
    activation: Optional[ActivationFn] = None
    params: Dict[str, Any] = field(default_factory=dict)
    constitutional: bool = False
    lifecycle: Dict[str, Any] = field(default_factory=dict)
    insertion_primitive: str = "post-insert"

    def activation_score(self, state: Dict[str, Any], task: Task) -> float:
        if self.activation is None:
            return 1.0
        return float(self.activation(state, task, self))

    def threshold(self) -> float:
        return float(self.params.get("activation_threshold", 0.5))

    def activate(self, state: Dict[str, Any], task: Task) -> bool:
        return self.activation_score(state, task) >= self.threshold()

    def required_inputs_missing(self, state: Dict[str, Any]) -> Iterable[str]:
        for field_name in self.input_schema:
            if field_name not in state:
                yield field_name
