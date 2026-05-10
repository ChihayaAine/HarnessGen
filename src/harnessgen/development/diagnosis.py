from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..core.types import FailureCluster, Trajectory


def _l2_distance(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


@dataclass
class DiagnosisState:
    previous_clusters: List[FailureCluster]
    lineage_counter: int = 0


class FailureDiagnoser:
    def __init__(self, max_components: int = 15, pca_dim: int = 32, random_state: int = 7) -> None:
        self.max_components = max_components
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.state = DiagnosisState(previous_clusters=[])

    def descriptor(self, trajectory: Trajectory) -> List[float]:
        activated = [int(step.activated) for step in trajectory.steps]
        latencies = [step.latency_ms for step in trajectory.steps]
        errors = sum(1 for step in trajectory.steps if not step.success)
        tool_errors = sum(1 for step in trajectory.steps if step.error_code and "tool" in step.error_code)
        verification_failures = sum(1 for step in trajectory.steps if step.family == "verifier" and not step.success)
        missing_input_failures = sum(1 for step in trajectory.steps if step.error_code and step.error_code.startswith("missing:"))
        activated_count = sum(activated)
        state = trajectory.final_state
        mode = trajectory.task.metadata["failure_mode"]
        mode_map = {
            "ambiguous_goal": [1, 0, 0, 0, 0],
            "trajectory_drift": [0, 1, 0, 0, 0],
            "tool_misuse": [0, 0, 1, 0, 0],
            "subgoal_stall": [0, 0, 0, 1, 0],
            "answer_error": [0, 0, 0, 0, 1],
            "clean": [0, 0, 0, 0, 0],
        }[mode]
        vector = activated + latencies + [
            errors,
            tool_errors,
            verification_failures,
            missing_input_failures,
            activated_count,
            len(state.get("history", [])),
            float(state.get("plan_quality", 0.0)),
            float(state.get("tool_quality", 0.0)),
            float(state.get("context_quality", 0.0)),
            float(state.get("verification_quality", 0.0)),
            float(state.get("constraint_quality", 0.0)),
            float(state.get("recovery_quality", 0.0)),
            float(sum(step.activation_score for step in trajectory.steps) / max(1, len(trajectory.steps))),
            float(trajectory.outcome),
        ] + mode_map
        return vector

    def cluster_failures(self, failures: Sequence[Trajectory]) -> Tuple[List[FailureCluster], List[List[float]]]:
        if not failures:
            return [], []
        features = [self.descriptor(item) for item in failures]
        labels = self._bucket_labels(failures)
        clusters: List[FailureCluster] = []
        for label in sorted(set(labels)):
            member_indices = [idx for idx, candidate_label in enumerate(labels) if candidate_label == label]
            center = np.mean(np.asarray([features[idx] for idx in member_indices], dtype=float), axis=0).tolist()
            signature = self._signature([failures[idx] for idx in member_indices])
            clusters.append(
                FailureCluster(
                    cluster_id=int(label),
                    member_indices=member_indices,
                    center=center,
                    signature=signature,
                )
            )
        self._match_clusters(clusters)
        return clusters, features

    def _bucket_labels(self, failures: Sequence[Trajectory]) -> List[int]:
        label_map: Dict[str, int] = {}
        labels: List[int] = []
        next_label = 0
        for item in failures:
            key = item.failure_type or "unknown_failure"
            if key not in label_map:
                label_map[key] = next_label
                next_label += 1
            labels.append(label_map[key])
        return labels

    def _signature(self, items: Sequence[Trajectory]) -> Dict[str, Any]:
        failure_types = [item.failure_type for item in items if item.failure_type]
        dominant_failure = max(set(failure_types), key=failure_types.count) if failure_types else "unknown_failure"
        avg_latency = float(np.mean([sum(step.latency_ms for step in item.steps) for item in items]))
        avg_context = float(np.mean([item.task.metadata.get("context_load", 0) for item in items]))
        avg_tool_pressure = float(np.mean([item.task.metadata.get("tool_pressure", 0) for item in items]))
        avg_missing_inputs = float(np.mean([
            sum(1 for step in item.steps if step.error_code and step.error_code.startswith("missing:"))
            for item in items
        ]))
        return {
            "dominant_failure_type": dominant_failure,
            "avg_latency_ms": avg_latency,
            "avg_context_load": avg_context,
            "avg_tool_pressure": avg_tool_pressure,
            "avg_missing_inputs": avg_missing_inputs,
            "size": len(items),
        }

    def _match_clusters(self, current: List[FailureCluster]) -> None:
        previous = self.state.previous_clusters
        if not previous or not current:
            for cluster in current:
                cluster.lineage_id = self._next_lineage_id()
            self.state.previous_clusters = current
            return
        matched_current = set()
        for prev_cluster in previous:
            best_idx = None
            best_distance = None
            for idx, cur_cluster in enumerate(current):
                if idx in matched_current:
                    continue
                distance = _l2_distance(prev_cluster.center, cur_cluster.center)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_idx = idx
            if best_idx is not None:
                current[best_idx].stable_cycles = prev_cluster.stable_cycles + 1
                current[best_idx].lineage_id = prev_cluster.lineage_id or self._next_lineage_id()
                matched_current.add(best_idx)
        for idx, cluster in enumerate(current):
            if idx not in matched_current:
                cluster.stable_cycles = 1
                cluster.lineage_id = self._next_lineage_id()
        self.state.previous_clusters = current

    def _next_lineage_id(self) -> str:
        self.state.lineage_counter += 1
        return f"cluster-lineage-{self.state.lineage_counter}"
