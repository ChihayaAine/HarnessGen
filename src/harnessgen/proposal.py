from __future__ import annotations

import copy
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from .harness import Harness
from .modules import ModuleSpec
from .schemas import schema_template
from .types import FailureCluster, ReplayResult, Task


def _family_for_failure(failure_type: str) -> str:
    return {
        "ambiguous_goal": "constraint_resolver",
        "trajectory_drift": "state_compressor",
        "tool_error": "tool_triage",
        "subgoal_stall": "recovery_handler",
        "answer_error": "verifier",
    }.get(failure_type, "verifier")


def _activation_for_family(family: str):
    def activation(state: Dict[str, Any], task: Task, module: ModuleSpec) -> float:
        mode = task.metadata["failure_mode"]
        mode_match = {
            "constraint_resolver": "ambiguous_goal",
            "state_compressor": "trajectory_drift",
            "tool_triage": "tool_misuse",
            "recovery_handler": "subgoal_stall",
            "verifier": "answer_error",
        }[family]
        score = 0.85 if mode == mode_match else 0.15
        if family == "state_compressor":
            score += min(0.2, 0.03 * int(task.metadata.get("context_load", 0)))
        if family == "tool_triage":
            score += 0.1 * int(task.metadata.get("tool_pressure", 0))
        return min(1.0, score)

    return activation


def _executor_for_family(family: str):
    if family == "constraint_resolver":
        return lambda state, task, module: {
            "constraints": {"self_elicited": True},
            "assumptions": [f"assumption-for-{task.task_id}"],
            "constraint_quality": float(module.params.get("strength", 0.8)),
        }
    if family == "state_compressor":
        return lambda state, task, module: {
            "compressed_state": True,
            "context_quality": float(module.params.get("strength", 0.8)),
        }
    if family == "tool_triage":
        return lambda state, task, module: {
            "tool_decision": "safe_tool_path",
            "tool_quality": float(module.params.get("strength", 0.8)),
        }
    if family == "recovery_handler":
        return lambda state, task, module: {
            "recovery_plan": f"recover-{task.task_id}",
            "recovery_quality": float(module.params.get("strength", 0.75)),
        }
    return lambda state, task, module: {
        "verification": "pass-with-repair",
        "verification_quality": float(module.params.get("strength", 0.8)),
    }


@dataclass
class ModuleProposal:
    module: ModuleSpec
    insertion_vertex: str
    scope: Dict[str, Any]
    lifecycle: Dict[str, Any]


class ProposalEngine:
    def propose(self, cluster: FailureCluster, index: int = 0) -> ModuleProposal:
        failure_type = str(cluster.signature.get("dominant_failure_type", "answer_error"))
        family = _family_for_failure(failure_type)
        template = schema_template(family)
        module = ModuleSpec(
            name=f"{family}_{index}",
            family=template["family"],
            input_schema=tuple(template["input_schema"]),
            output_schema=tuple(template["output_schema"]),
            executor=_executor_for_family(family),
            side_effects=tuple(template["side_effects"]),
            budget=dict(template["budget"]),
            activation=_activation_for_family(family),
            params=dict(template["default_params"]),
            lifecycle={"u_min": 0.02, "window": 2, "shadow_mode": False},
            insertion_primitive=str(template["insertion"]),
        )
        insertion_vertex = {
            "constraint_resolver": "plan",
            "state_compressor": "plan",
            "tool_triage": "act",
            "recovery_handler": "act",
            "verifier": "act",
        }[family]
        return ModuleProposal(
            module=module,
            insertion_vertex=insertion_vertex,
            scope={"dominant_failure_type": failure_type},
            lifecycle=module.lifecycle,
        )

    def integrate(self, harness: Harness, proposal: ModuleProposal) -> Harness:
        updated = harness.copy()
        module = copy.deepcopy(proposal.module)
        edges = self._edges_for_insertion(updated, module.name, proposal.insertion_vertex, module.insertion_primitive)
        updated.add_module(module, edges)
        return updated

    def _edges_for_insertion(
        self,
        harness: Harness,
        module_name: str,
        vertex: str,
        primitive: str,
    ) -> List[Tuple[str, str]]:
        outgoing = [dst for src, dst in harness.edges if src == vertex]
        incoming = [src for src, dst in harness.edges if dst == vertex]
        if primitive == "pre-insert":
            for src in incoming:
                harness.edges.remove((src, vertex))
            edges = [(src, module_name) for src in incoming] + [(module_name, vertex)]
        elif primitive == "post-insert":
            for dst in outgoing:
                harness.edges.remove((vertex, dst))
            edges = [(vertex, module_name)] + [(module_name, dst) for dst in outgoing]
        elif primitive == "guard-insert":
            edges = incoming and [(incoming[0], module_name), (module_name, vertex)] or [(module_name, vertex)]
        else:  # branch-insert
            edges = [(vertex, module_name), (module_name, "respond")]
        return edges

    def validate(
        self,
        base_harness: Harness,
        candidate_harness: Harness,
        replay_tasks: Iterable[Task],
        solved_tasks: Iterable[Task],
        latency_penalty: float = 0.001,
        utility_threshold: float = 0.02,
        regression_threshold: float = 0.20,
    ) -> ReplayResult:
        replay_list = list(replay_tasks)
        solved_list = list(solved_tasks)
        if not replay_list:
            return ReplayResult(utility=0.0, regression_rate=0.0, accepted=False, details={"replay_count": 0, "solved_count": len(solved_list)})
        base_scores = [base_harness.execute(task).outcome for task in replay_list]
        cand_scores = [candidate_harness.execute(task).outcome for task in replay_list]
        base_latency = [
            sum(step.latency_ms for step in base_harness.execute(task).steps)
            for task in replay_list
        ]
        cand_latency = [
            sum(step.latency_ms for step in candidate_harness.execute(task).steps)
            for task in replay_list
        ]
        score_delta = [cand - base for cand, base in zip(cand_scores, base_scores)]
        latency_delta = [cand - base for cand, base in zip(cand_latency, base_latency)]
        utility = float(mean(score_delta)) - latency_penalty * float(mean(latency_delta))
        regressions = 0
        for task in solved_list:
            if base_harness.execute(task).outcome == 1 and candidate_harness.execute(task).outcome == 0:
                regressions += 1
        regression_rate = regressions / max(1, len(solved_list))
        accepted = utility > utility_threshold and regression_rate <= regression_threshold
        return ReplayResult(
            utility=utility,
            regression_rate=regression_rate,
            accepted=accepted,
            details={"replay_count": len(replay_list), "solved_count": len(solved_list)},
        )
