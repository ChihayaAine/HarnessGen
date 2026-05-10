from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

from ..core.types import Trajectory


@dataclass
class EvaluationReport:
    benchmark_name: str
    cycle_count: int
    final_pass_rate: float
    accepted_module_count: int
    pruned_module_count: int
    average_cluster_count: float
    regression_free_cycles: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def evaluate_development(benchmark_name: str, reports: Sequence[object]) -> EvaluationReport:
    if not reports:
        return EvaluationReport(
            benchmark_name=benchmark_name,
            cycle_count=0,
            final_pass_rate=0.0,
            accepted_module_count=0,
            pruned_module_count=0,
            average_cluster_count=0.0,
            regression_free_cycles=0.0,
        )
    final_pass_rate = reports[-1].pass_rate
    accepted = sum(len(report.accepted_modules) for report in reports)
    pruned = sum(len(report.pruned_modules) for report in reports)
    avg_clusters = sum(len(report.clusters) for report in reports) / len(reports)
    regression_free = sum(
        1 for report in reports
        if all(cluster.get("proposal", {}).get("regression_rate", 0.0) <= 0.20 for cluster in report.clusters if "proposal" in cluster)
    ) / len(reports)
    return EvaluationReport(
        benchmark_name=benchmark_name,
        cycle_count=len(reports),
        final_pass_rate=final_pass_rate,
        accepted_module_count=accepted,
        pruned_module_count=pruned,
        average_cluster_count=avg_clusters,
        regression_free_cycles=regression_free,
    )
