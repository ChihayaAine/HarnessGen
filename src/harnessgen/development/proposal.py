from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ..core.harness import Harness
from ..core.modules import ModuleSpec
from ..core.types import FailureCluster, ReplayResult, Task
from .evaluators import evaluate_family_result
from .replay import validate_replay
from .replay_analysis import analyze_replay_deltas, replay_slice_summary
from .schemas import schema_template
from .surgery import apply_graph_surgery, edge_set_snapshot


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
            "constraints": {"inferred_locally": True},
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
    def _proposal_family(self, base_harness: Harness, candidate_harness: Harness) -> str:
        new_names = [name for name in candidate_harness.modules if name not in base_harness.modules]
        if not new_names:
            return "verifier"
        return str(candidate_harness.modules[new_names[-1]].family)

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
        module = copy.deepcopy(proposal.module)
        updated, edges = apply_graph_surgery(harness, module, proposal.insertion_vertex, module.insertion_primitive)
        module.lifecycle["edge_snapshot"] = edge_set_snapshot(edges)
        return updated

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
        replay_diag = validate_replay(
            base_harness=base_harness,
            candidate_harness=candidate_harness,
            replay_tasks=replay_tasks,
            solved_tasks=solved_tasks,
            latency_penalty=latency_penalty,
        )
        replay_result = ReplayResult(
            utility=replay_diag.utility,
            regression_rate=replay_diag.regression_rate,
            accepted=False,
            details=replay_diag.to_dict(),
        )
        proposal_family = self._proposal_family(base_harness, candidate_harness)
        family_eval = evaluate_family_result(
            proposal_family,
            replay_result=replay_result,
            utility_threshold=utility_threshold,
            regression_threshold=regression_threshold,
        )
        replay_slice = analyze_replay_deltas(replay_diag.per_task_delta or [])
        return ReplayResult(
            utility=family_eval.adjusted_utility,
            regression_rate=replay_diag.regression_rate,
            accepted=family_eval.accepted,
            details={
                **replay_diag.to_dict(),
                "family_evaluation": family_eval.to_dict(),
                "replay_slice": replay_slice_summary(replay_slice),
            },
            decision_trace=[
                {
                    "utility_threshold": utility_threshold,
                    "regression_threshold": regression_threshold,
                    "latency_penalty": latency_penalty,
                    "accepted": family_eval.accepted,
                    "family_evaluation": family_eval.to_dict(),
                }
            ],
        )
