from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class BudgetSnapshot:
    cycle_index: int
    grow_cap: int
    grown_modules: int
    remaining_slots: int
    can_grow: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class GrowthBudget:
    def __init__(self, base_budget: int, linear_increment: int) -> None:
        self.base_budget = base_budget
        self.linear_increment = linear_increment

    def snapshot(self, cycle_index: int, grown_modules: int) -> BudgetSnapshot:
        grow_cap = self.base_budget + cycle_index * self.linear_increment
        remaining_slots = max(0, grow_cap - grown_modules)
        return BudgetSnapshot(
            cycle_index=cycle_index,
            grow_cap=grow_cap,
            grown_modules=grown_modules,
            remaining_slots=remaining_slots,
            can_grow=remaining_slots > 0,
        )
