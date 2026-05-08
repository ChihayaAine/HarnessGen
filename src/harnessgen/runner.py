from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .benchmarks import BenchmarkDataset
from .development import DevelopmentConfig, HarnessGenEngine
from .evaluation import EvaluationReport, evaluate_development


@dataclass
class ExperimentArtifacts:
    reports: List[dict]
    metrics: EvaluationReport
    final_state: Dict[str, object]


def run_experiment(
    dataset: BenchmarkDataset,
    config: Optional[DevelopmentConfig] = None,
) -> ExperimentArtifacts:
    engine = HarnessGenEngine(config=config)
    reports = engine.develop(dataset=dataset)
    metrics = evaluate_development(dataset.name, reports)
    return ExperimentArtifacts(
        reports=[report.to_dict() for report in reports],
        metrics=metrics,
        final_state=engine.summary(),
    )


def write_artifacts(artifacts: ExperimentArtifacts, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cycles.jsonl", "w", encoding="utf-8") as handle:
        for report in artifacts.reports:
            handle.write(json.dumps(report, ensure_ascii=False) + "\n")
    with open(path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.metrics.to_dict(), handle, indent=2, ensure_ascii=False)
    with open(path / "final_state.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.final_state, handle, indent=2, ensure_ascii=False)
