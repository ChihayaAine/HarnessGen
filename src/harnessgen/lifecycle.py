from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List

from .harness import Harness
from .types import Task


class LifecycleManager:
    def __init__(self, beta: float = 0.5, gamma: float = 0.7, epsilon_prune: float = 0.0) -> None:
        self.beta = beta
        self.gamma = gamma
        self.epsilon_prune = epsilon_prune
        self.utility_state: Dict[str, float] = {}
        self.low_utility_windows: Dict[str, int] = {}

    def update(self, harness: Harness, tasks: Iterable[Task]) -> List[str]:
        pruned: List[str] = []
        for module in list(harness.modules.values()):
            if module.constitutional:
                continue
            alpha = self._counterfactual_utility(harness, module.name, tasks)
            prev = self.utility_state.get(module.name, 0.0)
            utility = (1 - self.beta) * prev + self.beta * alpha
            self.utility_state[module.name] = utility
            if utility < module.lifecycle.get("u_min", 0.02):
                self.low_utility_windows[module.name] = self.low_utility_windows.get(module.name, 0) + 1
            else:
                self.low_utility_windows[module.name] = 0
            if self.low_utility_windows[module.name] == 1:
                module.params["activation_threshold"] = min(0.95, module.threshold() / self.gamma)
            elif self.low_utility_windows[module.name] == 2:
                module.lifecycle["shadow_mode"] = True
            elif self.low_utility_windows[module.name] >= module.lifecycle.get("window", 2) + 1 and utility <= self.epsilon_prune:
                harness.remove_module(module.name)
                pruned.append(module.name)
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
