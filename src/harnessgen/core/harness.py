from __future__ import annotations

import copy
import time
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

from .modules import ModuleSpec
from .types import ModuleEvent, Task, Trajectory


class HarnessExecutionError(RuntimeError):
    pass


class Harness:
    def __init__(self, modules: Iterable[ModuleSpec], edges: Iterable[Tuple[str, str]]) -> None:
        self.modules: Dict[str, ModuleSpec] = {module.name: module for module in modules}
        self.edges: List[Tuple[str, str]] = list(edges)
        self._validate_acyclic()

    def copy(self) -> "Harness":
        return Harness(
            modules=[copy.deepcopy(module) for module in self.modules.values()],
            edges=list(self.edges),
        )

    def topological_order(self) -> List[str]:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.modules.keys())
        for src, dst in self.edges:
            graph.add_edge(src, dst)
        if not nx.is_directed_acyclic_graph(graph):
            raise HarnessExecutionError("Harness graph contains a cycle.")
        return list(nx.lexicographical_topological_sort(graph))

    def _validate_acyclic(self) -> None:
        self.topological_order()

    def predecessors(self, node: str) -> List[str]:
        return [src for src, dst in self.edges if dst == node]

    def successors(self, node: str) -> List[str]:
        return [dst for src, dst in self.edges if src == node]

    def active_subgraph(self, state: Dict[str, Any], task: Task) -> List[str]:
        order = self.topological_order()
        return [name for name in order if self.modules[name].activate(state, task)]

    def execute(self, task: Task, seed: int = 0) -> Trajectory:
        state: Dict[str, Any] = {
            "task_id": task.task_id,
            "goal": task.goal,
            "metadata": copy.deepcopy(task.metadata),
            "history": [],
            "available_tools": ["shell", "editor", "tests", "search"],
            "random_seed": seed,
            "module_metrics": {},
            }
        events: List[ModuleEvent] = []
        snapshots: List[Dict[str, Any]] = []
        for module_name in self.topological_order():
            module = self.modules[module_name]
            activation_score = module.activation_score(state, task)
            threshold = module.threshold()
            activated = activation_score >= threshold
            latency_ms = float(module.budget.get("latency_ms", 0))
            start = time.perf_counter()
            success = True
            error_code = None
            notes: Dict[str, Any] = {}
            missing: List[str] = []
            if activated:
                missing = list(module.required_inputs_missing(state))
                if missing:
                    success = False
                    error_code = f"missing:{','.join(missing)}"
                else:
                    try:
                        updates = module.executor(state, task, module)
                        if updates:
                            state.update(updates)
                    except Exception as exc:  # pragma: no cover - defensive
                        success = False
                        error_code = exc.__class__.__name__
                        notes["message"] = str(exc)
            state["history"].append(
                {
                    "module": module_name,
                    "activated": activated,
                    "activation_score": activation_score,
                    "threshold": threshold,
                    "success": success,
                    "error_code": error_code,
                }
            )
            state["module_metrics"][module_name] = {
                "activation_score": activation_score,
                "threshold": threshold,
                "success": success,
                "missing_inputs": list(missing),
            }
            elapsed_ms = (time.perf_counter() - start) * 1000.0 + latency_ms
            event = ModuleEvent(
                name=module_name,
                family=module.family,
                activated=activated,
                latency_ms=elapsed_ms,
                success=success,
                error_code=error_code,
                notes=notes,
                activation_score=activation_score,
                threshold=threshold,
                missing_inputs=list(missing),
            )
            events.append(event)
            snapshots.append(copy.deepcopy(state))
        outcome = int(bool(state.get("task_success", False)))
        failure_type = None if outcome else str(state.get("failure_type", "unknown_failure"))
        return Trajectory(
            task=task,
            steps=events,
            state_snapshots=snapshots,
            outcome=outcome,
            failure_type=failure_type,
            final_state=state,
        )

    def add_module(self, module: ModuleSpec, edges: Iterable[Tuple[str, str]]) -> None:
        if module.name in self.modules:
            raise ValueError(f"Module already exists: {module.name}")
        for src, dst in edges:
            if src != module.name and src not in self.modules:
                raise ValueError(f"Unknown source vertex: {src}")
            if dst != module.name and dst not in self.modules:
                raise ValueError(f"Unknown destination vertex: {dst}")
        self.modules[module.name] = module
        self.edges.extend(edges)
        self._validate_acyclic()

    def remove_module(self, name: str) -> None:
        if name not in self.modules:
            return
        del self.modules[name]
        self.edges = [(src, dst) for src, dst in self.edges if src != name and dst != name]
        self._validate_acyclic()
