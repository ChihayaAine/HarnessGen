from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ClusterHistoryRecord:
    lineage_id: str
    observations: int = 0
    accepted_proposals: int = 0
    rejected_proposals: int = 0
    last_failure_type: str = "unknown_failure"
    residual_gaps: List[float] = field(default_factory=list)

    def register_observation(self, failure_type: str, residual_gap: float) -> None:
        self.observations += 1
        self.last_failure_type = failure_type
        self.residual_gaps.append(float(residual_gap))
        self.residual_gaps = self.residual_gaps[-8:]

    def register_proposal(self, accepted: bool) -> None:
        if accepted:
            self.accepted_proposals += 1
        else:
            self.rejected_proposals += 1

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["avg_residual_gap"] = (
            sum(self.residual_gaps) / len(self.residual_gaps) if self.residual_gaps else 0.0
        )
        return payload


class DevelopmentHistory:
    def __init__(self) -> None:
        self.cluster_history: Dict[str, ClusterHistoryRecord] = {}

    def record_cluster(self, lineage_id: str, failure_type: str, residual_gap: float) -> ClusterHistoryRecord:
        record = self.cluster_history.setdefault(lineage_id, ClusterHistoryRecord(lineage_id=lineage_id))
        record.register_observation(failure_type=failure_type, residual_gap=residual_gap)
        return record

    def record_proposal(self, lineage_id: str, accepted: bool) -> None:
        record = self.cluster_history.setdefault(lineage_id, ClusterHistoryRecord(lineage_id=lineage_id))
        record.register_proposal(accepted=accepted)

    def summary(self) -> Dict[str, Any]:
        return {
            "cluster_count": len(self.cluster_history),
            "clusters": {lineage_id: record.to_dict() for lineage_id, record in self.cluster_history.items()},
        }
