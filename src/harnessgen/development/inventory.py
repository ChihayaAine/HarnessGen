from __future__ import annotations

from typing import Any, Dict

from ..core.harness import Harness


def module_inventory(harness: Harness) -> Dict[str, Any]:
    grown = [module for module in harness.modules.values() if not module.constitutional]
    family_counts: Dict[str, int] = {}
    primitive_counts: Dict[str, int] = {}
    for module in harness.modules.values():
        family_counts[module.family] = family_counts.get(module.family, 0) + 1
        primitive_counts[module.insertion_primitive] = primitive_counts.get(module.insertion_primitive, 0) + 1
    return {
        "total_modules": len(harness.modules),
        "grown_modules": len(grown),
        "family_counts": family_counts,
        "primitive_counts": primitive_counts,
        "module_thresholds": {name: module.threshold() for name, module in harness.modules.items()},
        "shadow_modules": [module.name for module in grown if module.lifecycle.get("shadow_mode")],
        "families": {name: module.family for name, module in harness.modules.items()},
    }
