from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .budgeting import GrowthBudget
from .diagnosis import FailureDiagnoser
from .history import DevelopmentHistory
from .holdout import HoldoutManager
from .inventory import module_inventory
from .lifecycle import LifecycleManager
from .proposal import ProposalEngine
from .recalibration import Recalibrator
from .ranking import rank_clusters, ranking_summary
from ..core.factory import build_default_harness
from ..core.harness import Harness
from ..core.types import Task, Trajectory
from ..evaluation.benchmarks import BenchmarkDataset, SyntheticHarnessBenchmark


@dataclass
class DevelopmentConfig:
    cycles: int = 5
    batch_size: int = 48
    epsilon_struct: float = 0.55
    persistent_k: int = 2
    utility_threshold: float = 0.02
    regression_threshold: float = 0.20
    grow_budget: int = 6
    alpha_grow: int = 1
    min_cluster_size: int = 2


@dataclass
class CycleReport:
    cycle_index: int
    pass_rate: float
    failures: int
    clusters: List[Dict[str, Any]] = field(default_factory=list)
    accepted_modules: List[str] = field(default_factory=list)
    pruned_modules: List[str] = field(default_factory=list)
    module_inventory: Dict[str, Any] = field(default_factory=dict)
    budget_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HarnessGenEngine:
    def __init__(
        self,
        harness: Optional[Harness] = None,
        benchmark: Optional[SyntheticHarnessBenchmark] = None,
        config: Optional[DevelopmentConfig] = None,
    ) -> None:
        self.harness = harness or build_default_harness()
        self.benchmark = benchmark or SyntheticHarnessBenchmark()
        self.config = config or DevelopmentConfig()
        self.diagnoser = FailureDiagnoser()
        self.recalibrator = Recalibrator()
        self.proposal_engine = ProposalEngine()
        self.lifecycle = LifecycleManager()
        self.reports: List[CycleReport] = []
        self.history = DevelopmentHistory()
        self.growth_budget = GrowthBudget(
            base_budget=self.config.grow_budget,
            linear_increment=self.config.alpha_grow,
        )
        self.holdout_manager = HoldoutManager()

    def develop(self, dataset: Optional[BenchmarkDataset] = None) -> List[CycleReport]:
        if dataset is None:
            dataset = self.benchmark.build_dataset(count=self.config.batch_size)
        for cycle_index in range(self.config.cycles):
            tasks = self._tasks_for_cycle(dataset, cycle_index)
            trajectories = [self.harness.execute(task, seed=cycle_index) for task in tasks]
            failures = [traj for traj in trajectories if traj.outcome == 0]
            pass_rate = 1.0 - (len(failures) / max(1, len(trajectories)))
            report = CycleReport(cycle_index=cycle_index, pass_rate=pass_rate, failures=len(failures))
            report.budget_snapshot = self.growth_budget.snapshot(
                cycle_index=cycle_index,
                grown_modules=self._grown_module_count(),
            ).to_dict()
            clusters, _ = self.diagnoser.cluster_failures(failures)
            residual_by_lineage: Dict[str, float] = {}
            for cluster in clusters:
                cluster_trajectories = [failures[idx] for idx in cluster.member_indices]
                if len(cluster_trajectories) < self.config.min_cluster_size:
                    continue
                recal = self.recalibrator.recalibrate(self.harness, cluster_trajectories)
                residual_by_lineage[cluster.lineage_id or str(cluster.cluster_id)] = self.config.epsilon_struct - recal.score
            report.module_inventory["cluster_ranking"] = ranking_summary(rank_clusters(clusters, residual_by_lineage))
            holdout_slice = self.holdout_manager.build(tasks)
            report.module_inventory["holdout"] = self.holdout_manager.summary(holdout_slice)
            for cluster in clusters:
                cluster_trajectories = [failures[idx] for idx in cluster.member_indices]
                if len(cluster_trajectories) < self.config.min_cluster_size:
                    continue
                recal = self.recalibrator.recalibrate(self.harness, cluster_trajectories)
                residual_gap = self.config.epsilon_struct - recal.score
                lineage_id = cluster.lineage_id or str(cluster.cluster_id)
                cluster_record = self.history.record_cluster(
                    lineage_id=lineage_id,
                    failure_type=str(cluster.signature.get("dominant_failure_type", "unknown_failure")),
                    residual_gap=residual_gap,
                )
                cluster_payload = {
                    "cluster_id": cluster.cluster_id,
                    "lineage_id": lineage_id,
                    "stable_cycles": cluster.stable_cycles,
                    "signature": cluster.signature,
                    "recalibrated_score": recal.score,
                    "search_score": recal.search_score,
                    "selection_score": recal.selection_score,
                    "residual_gap": residual_gap,
                    "recalibration_attempts": recal.attempted_updates,
                    "history": cluster_record.to_dict(),
                }
                report.clusters.append(cluster_payload)
                if residual_gap <= 0 or cluster.stable_cycles < self.config.persistent_k:
                    continue
                budget_snapshot = self.growth_budget.snapshot(cycle_index=cycle_index, grown_modules=self._grown_module_count())
                if not budget_snapshot.can_grow:
                    continue
                proposal = self.proposal_engine.propose(cluster, index=self._grown_module_count())
                candidate = self.proposal_engine.integrate(self.harness, proposal)
                solved_tasks = holdout_slice.solved_tasks or self.benchmark.solved_subset(tasks)
                replay = self.proposal_engine.validate(
                    base_harness=self.harness,
                    candidate_harness=candidate,
                    replay_tasks=[item.task for item in cluster_trajectories],
                    solved_tasks=solved_tasks,
                    utility_threshold=self.config.utility_threshold,
                    regression_threshold=self.config.regression_threshold,
                )
                cluster_payload["proposal"] = {
                    "module": proposal.module.name,
                    "family": proposal.module.family,
                    "accepted": replay.accepted,
                    "utility": replay.utility,
                    "regression_rate": replay.regression_rate,
                    "details": replay.details,
                    "decision_trace": replay.decision_trace,
                }
                self.history.record_proposal(lineage_id=lineage_id, accepted=replay.accepted)
                if replay.accepted:
                    self.harness = candidate
                    report.accepted_modules.append(proposal.module.name)
            pruned = self.lifecycle.update(self.harness, tasks)
            report.pruned_modules.extend(pruned)
            report.module_inventory.update(self._module_inventory())
            self.reports.append(report)
        return self.reports

    def replay(self, tasks: Sequence[Task], seed: int = 0) -> List[Trajectory]:
        return [self.harness.execute(task, seed=seed) for task in tasks]

    def summary(self) -> Dict[str, Any]:
        return {
            "cycles": [report.to_dict() for report in self.reports],
            "final_modules": list(self.harness.modules.keys()),
            "history": self.history.summary(),
        }

    def _grown_module_count(self) -> int:
        return sum(1 for module in self.harness.modules.values() if not module.constitutional)

    def _tasks_for_cycle(self, dataset: BenchmarkDataset, cycle_index: int) -> List[Task]:
        if len(dataset.tasks) <= self.config.batch_size:
            return list(dataset.tasks)
        start = (cycle_index * self.config.batch_size) % len(dataset.tasks)
        end = start + self.config.batch_size
        if end <= len(dataset.tasks):
            return dataset.tasks[start:end]
        wrap = end - len(dataset.tasks)
        return dataset.tasks[start:] + dataset.tasks[:wrap]

    def _module_inventory(self) -> Dict[str, Any]:
        return module_inventory(self.harness)
