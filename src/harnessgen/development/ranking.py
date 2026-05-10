from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from ..core.types import FailureCluster


@dataclass
class ClusterPriority:
    lineage_id: str
    cluster_id: int
    priority: float
    residual_gap: float
    stable_cycles: int
    failure_type: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def rank_clusters(
    clusters: Iterable[FailureCluster],
    residual_gaps: Dict[str, float],
) -> List[ClusterPriority]:
    ranked: List[ClusterPriority] = []
    for cluster in clusters:
        lineage_id = cluster.lineage_id or str(cluster.cluster_id)
        residual_gap = float(residual_gaps.get(lineage_id, 0.0))
        priority = residual_gap + 0.15 * cluster.stable_cycles + 0.02 * float(cluster.signature.get("size", 0))
        ranked.append(
            ClusterPriority(
                lineage_id=lineage_id,
                cluster_id=cluster.cluster_id,
                priority=priority,
                residual_gap=residual_gap,
                stable_cycles=cluster.stable_cycles,
                failure_type=str(cluster.signature.get("dominant_failure_type", "unknown_failure")),
            )
        )
    ranked.sort(key=lambda item: item.priority, reverse=True)
    return ranked


def ranking_summary(priorities: Iterable[ClusterPriority]) -> Dict[str, object]:
    priority_list = list(priorities)
    return {
        "count": len(priority_list),
        "top_lineages": [item.lineage_id for item in priority_list[:5]],
        "priorities": [item.to_dict() for item in priority_list],
    }
