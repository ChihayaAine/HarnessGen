# HarnessGen

Diagnosis-triggered structural adaptation for LLM harnesses.

This repository is a reference implementation of the HarnessGen framework. It
contains:

- a typed DAG harness runtime
- structured trajectory collection
- recurring failure diagnosis
- bounded recalibration over existing modules
- schema-constrained module proposal and graph surgery
- replay validation with regression checks
- lifecycle management with decay, shadow mode, and pruning
- benchmark loading from synthetic generation or JSONL files
- experiment running, metric computation, and artifact export

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

## Running Experiments

### 1. Synthetic benchmark

```bash
harnessgen --benchmark synthetic --cycles 5 --batch-size 48
```

### 2. JSONL dataset with artifact export

```bash
harnessgen \
  --benchmark jsonl \
  --tasks-file data/sample_tasks.jsonl \
  --cycles 3 \
  --batch-size 4 \
  --output-dir outputs/sample_run
```

### 3. Machine-readable output

```bash
harnessgen --benchmark synthetic --cycles 5 --batch-size 48 --json
```

## Artifact Export

When `--output-dir` is provided, the runner writes:

- `cycles.jsonl`
- `metrics.json`
- `final_state.json`

## Dataset Format

One task per line:

```json
{
  "task_id": "sample-001",
  "goal": "Resolve the benchmark task with ambiguous requirements and incomplete task constraints.",
  "metadata": {
    "failure_mode": "ambiguous_goal",
    "difficulty": 0.61,
    "context_load": 4,
    "tool_pressure": 1,
    "ambiguity": 1,
    "requires_verification": 0
  },
  "gold": {
    "expected_pass": false,
    "failure_mode": "ambiguous_goal"
  }
}
```

Example file:

- [data/sample_tasks.jsonl](/Users/Zhuanz1/Desktop/new_terminal_bench/HarnessGen/data/sample_tasks.jsonl)

## Metrics

The evaluator currently reports:

- `final_pass_rate`
- `accepted_module_count`
- `pruned_module_count`
- `average_cluster_count`
- `regression_free_cycles`

## Repository Layout

```text
src/harnessgen/
  benchmarks.py     # synthetic benchmark + JSONL task loading
  cli.py            # experiment CLI
  development.py    # development engine
  diagnosis.py      # feature extraction, clustering, cross-cycle matching
  evaluation.py     # metric computation
  factory.py        # default constitutional harness
  harness.py        # DAG execution runtime
  lifecycle.py      # atrophy, shadow mode, pruning
  modules.py        # typed module definitions
  proposal.py       # schema-driven module proposal and graph surgery
  recalibration.py  # bounded recalibration gate
  runner.py         # experiment execution + artifact writing
  schemas.py        # schema library
  types.py          # shared dataclasses
data/
  sample_tasks.jsonl
tests/
  test_smoke.py
```

## Notes

This repository covers the main framework mechanics from the paper and exposes
a clean JSONL protocol for external benchmark adapters. It does not include
official adapters for Terminal-Bench 2.0, SWE-bench Verified, SWE-bench
Multilingual, OSWorld-Verified, or tau^2-Bench; those can be layered onto the
task-loading and experiment interfaces provided here.
