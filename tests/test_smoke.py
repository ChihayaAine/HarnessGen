import unittest
from pathlib import Path

from harnessgen.benchmarks import SyntheticHarnessBenchmark, load_benchmark_dataset
from harnessgen.development import DevelopmentConfig, HarnessGenEngine
from harnessgen.factory import build_default_harness
from harnessgen.runner import run_experiment

class SmokeTests(unittest.TestCase):
    def test_default_harness_executes(self):
        harness = build_default_harness()
        traj = harness.execute(
            type(
                "TaskLike",
                (),
                {
                    "task_id": "sample",
                    "goal": "sample benchmark task",
                    "metadata": {
                        "failure_mode": "clean",
                        "difficulty": 0.2,
                        "context_load": 1,
                        "tool_pressure": 0,
                        "ambiguity": 0,
                        "requires_verification": 0,
                    },
                    "gold": {},
                },
            )()
        )
        self.assertEqual(len(traj.steps), 4)
        self.assertTrue(traj.final_state["response"].startswith("response-for-"))

    def test_development_loop_grows_modules(self):
        benchmark = SyntheticHarnessBenchmark(seed=9)
        dataset = benchmark.build_dataset(count=30)
        engine = HarnessGenEngine(config=DevelopmentConfig(cycles=3, batch_size=30))
        reports = engine.develop(dataset=dataset)
        self.assertEqual(len(reports), 3)
        self.assertGreaterEqual(len(engine.harness.modules), 4)

    def test_jsonl_dataset_and_runner_work(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset = load_benchmark_dataset(
            benchmark_name="jsonl",
            tasks_path=str(repo_root / "data" / "sample_tasks.jsonl"),
        )
        artifacts = run_experiment(dataset=dataset, config=DevelopmentConfig(cycles=2, batch_size=4))
        self.assertEqual(artifacts.metrics.benchmark_name, "sample_tasks")
        self.assertGreaterEqual(len(artifacts.reports), 1)
        self.assertIn("final_modules", artifacts.final_state)


if __name__ == "__main__":
    unittest.main()
