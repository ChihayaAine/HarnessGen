from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModuleSignal:
    utility: float = 0.0
    low_utility_windows: int = 0
    stage: str = "active"
    history: list[float] = field(default_factory=list)
    last_alpha: float = 0.0
    max_utility: float = 0.0


class SignalTracker:
    def __init__(self, beta: float) -> None:
        self.beta = beta
        self.state: Dict[str, ModuleSignal] = {}

    def update(self, module_name: str, alpha: float, u_min: float) -> ModuleSignal:
        signal = self.state.setdefault(module_name, ModuleSignal())
        signal.utility = (1 - self.beta) * signal.utility + self.beta * alpha
        signal.history.append(alpha)
        signal.last_alpha = alpha
        signal.max_utility = max(signal.max_utility, signal.utility)
        if signal.utility < u_min:
            signal.low_utility_windows += 1
        else:
            signal.low_utility_windows = 0
            signal.stage = "active"
        return signal
