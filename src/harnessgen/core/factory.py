from __future__ import annotations

from typing import Any, Dict, Tuple

from .harness import Harness
from .modules import ModuleSpec
from .types import Task


def _score_success(state: Dict[str, Any], task: Task) -> Tuple[bool, str]:
    mode = task.metadata["failure_mode"]
    difficulty = float(task.metadata["difficulty"])
    plan_quality = float(state.get("plan_quality", 0.35))
    tool_quality = float(state.get("tool_quality", 0.35))
    context_quality = float(state.get("context_quality", 0.35))
    recovery_quality = float(state.get("recovery_quality", 0.2))
    verification_quality = float(state.get("verification_quality", 0.2))
    constraint_quality = float(state.get("constraint_quality", 0.2))

    success_score = 0.25 + 0.4 * plan_quality - 0.25 * difficulty
    if mode == "ambiguous_goal":
        success_score += 0.6 * constraint_quality
    elif mode == "trajectory_drift":
        success_score += 0.6 * context_quality
    elif mode == "tool_misuse":
        success_score += 0.6 * tool_quality
    elif mode == "subgoal_stall":
        success_score += 0.6 * recovery_quality
    elif mode == "answer_error":
        success_score += 0.6 * verification_quality
    else:
        success_score += 0.25

    success = success_score >= 0.6
    failure_map = {
        "ambiguous_goal": "ambiguous_goal",
        "trajectory_drift": "trajectory_drift",
        "tool_misuse": "tool_error",
        "subgoal_stall": "subgoal_stall",
        "answer_error": "answer_error",
        "clean": "none",
    }
    return success, failure_map[mode]


def observe_executor(state: Dict[str, Any], task: Task, module: ModuleSpec) -> Dict[str, Any]:
    return {"observation": task.goal, "context_quality": 0.3}


def plan_executor(state: Dict[str, Any], task: Task, module: ModuleSpec) -> Dict[str, Any]:
    base = float(module.params.get("prompt_strength", 0.45))
    boost = 0.15 if state.get("constraints") else 0.0
    drift_penalty = min(0.2, 0.02 * int(task.metadata.get("context_load", 0)))
    return {
        "plan": f"plan-for-{task.task_id}",
        "plan_quality": max(0.0, min(1.0, base + boost - drift_penalty)),
    }


def act_executor(state: Dict[str, Any], task: Task, module: ModuleSpec) -> Dict[str, Any]:
    tool_quality = float(module.params.get("tool_policy_strength", 0.4))
    if state.get("tool_decision") == "safe_tool_path":
        tool_quality += 0.35
    recovery_quality = 0.3 + (0.35 if state.get("recovery_plan") else 0.0)
    action_result = {
        "used_tool": "shell",
        "tool_quality": max(0.0, min(1.0, tool_quality)),
        "recovery_quality": max(0.0, min(1.0, recovery_quality)),
    }
    return {"action_result": action_result, "tool_quality": action_result["tool_quality"], "recovery_quality": action_result["recovery_quality"]}


def respond_executor(state: Dict[str, Any], task: Task, module: ModuleSpec) -> Dict[str, Any]:
    success, failure_type = _score_success(state, task)
    return {
        "response": f"response-for-{task.task_id}",
        "task_success": success,
        "failure_type": None if success else failure_type,
    }


def build_default_harness() -> Harness:
    modules = [
        ModuleSpec(
            name="observe",
            family="observe",
            input_schema=(),
            output_schema=("observation", "context_quality"),
            executor=observe_executor,
            params={"activation_threshold": 0.0},
            constitutional=True,
        ),
        ModuleSpec(
            name="plan",
            family="plan",
            input_schema=("observation",),
            output_schema=("plan", "plan_quality"),
            executor=plan_executor,
            params={"prompt_strength": 0.45, "activation_threshold": 0.0},
            constitutional=True,
        ),
        ModuleSpec(
            name="act",
            family="act",
            input_schema=("plan",),
            output_schema=("action_result", "tool_quality", "recovery_quality"),
            executor=act_executor,
            params={"tool_policy_strength": 0.4, "activation_threshold": 0.0},
            constitutional=True,
        ),
        ModuleSpec(
            name="respond",
            family="respond",
            input_schema=("action_result",),
            output_schema=("response", "task_success", "failure_type"),
            executor=respond_executor,
            params={"activation_threshold": 0.0},
            constitutional=True,
        ),
    ]
    edges = [("observe", "plan"), ("plan", "act"), ("act", "respond")]
    return Harness(modules=modules, edges=edges)
