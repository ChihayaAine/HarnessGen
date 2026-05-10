from __future__ import annotations

from statistics import mean
from typing import Iterable, List

from ..core.harness import Harness
from ..core.types import Task
from .signals import SignalTracker


class LifecycleManager:
    def __init__(self, beta: float = 0.5, gamma: float = 0.7, epsilon_prune: float = 0.0) -> None:
        self.beta = beta
        self.gamma = gamma
        self.epsilon_prune = epsilon_prune
        self.signal_tracker = SignalTracker(beta=beta)

    def update(self, harness: Harness, tasks: Iterable[Task]) -> List[str]:
        pruned: List[str] = []
        task_list = list(tasks)
        for module in list(harness.modules.values()):
            if module.constitutional:
                continue
            alpha = self._counterfactual_utility(harness, module.name, task_list)
            signal = self.signal_tracker.update(module.name, alpha, module.lifecycle.get("u_min", 0.02))
            module.lifecycle["utility"] = signal.utility
            module.lifecycle["signal_stage"] = signal.stage
            module.lifecycle["last_alpha"] = signal.last_alpha
            module.lifecycle["max_utility"] = signal.max_utility
            module.lifecycle["utility_history"] = signal.history[-8:]
            if signal.low_utility_windows == 1:
                module.params["activation_threshold"] = min(0.95, module.threshold() / self.gamma)
                signal.stage = "activation_decay"
            elif signal.low_utility_windows == 2:
                module.lifecycle["shadow_mode"] = True
                signal.stage = "shadow"
            elif signal.low_utility_windows >= module.lifecycle.get("window", 2) + 1 and signal.utility <= self.epsilon_prune:
                harness.remove_module(module.name)
                pruned.append(module.name)
                signal.stage = "pruned"
            else:
                module.lifecycle["shadow_mode"] = bool(module.lifecycle.get("shadow_mode", False))
        return pruned

    def _counterfactual_utility(self, harness: Harness, module_name: str, tasks: Iterable[Task]) -> float:
        task_list = list(tasks)
        if not task_list:
            return 0.0
        baseline = mean([harness.execute(task, seed=11).outcome for task in task_list])
        ablated = harness.copy()
        ablated.remove_module(module_name)
        without = mean([ablated.execute(task, seed=11).outcome for task in task_list])
        return float(baseline - without)
