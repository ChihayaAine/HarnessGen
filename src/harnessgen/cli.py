from __future__ import annotations

import argparse
import json

from .benchmarks import load_benchmark_dataset
from .development import DevelopmentConfig
from .runner import run_experiment, write_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HarnessGen experiments.")
    parser.add_argument("--benchmark", choices=["synthetic", "jsonl"], default="synthetic")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DevelopmentConfig(cycles=args.cycles, batch_size=args.batch_size)
    dataset = load_benchmark_dataset(
        benchmark_name=args.benchmark,
        task_count=args.batch_size * args.cycles,
        seed=args.seed,
        tasks_path=args.tasks_file,
    )
    artifacts = run_experiment(dataset=dataset, config=config)
    if args.output_dir:
        write_artifacts(artifacts, args.output_dir)
    if args.json:
        print(
            json.dumps(
                {
                    "benchmark": dataset.name,
                    "metrics": artifacts.metrics.to_dict(),
                    "cycles": artifacts.reports,
                    "state": artifacts.final_state,
                },
                indent=2,
            )
        )
        return
    for report in artifacts.reports:
        print(
            f"cycle={report['cycle_index']} pass_rate={report['pass_rate']:.3f} "
            f"failures={report['failures']} accepted={report['accepted_modules']} pruned={report['pruned_modules']}"
        )
    print(json.dumps(artifacts.metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
