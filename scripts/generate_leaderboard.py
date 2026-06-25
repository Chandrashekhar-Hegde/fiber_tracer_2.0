"""Generate a Markdown leaderboard from benchmark JSON results.

Example:
    python scripts/generate_leaderboard.py \
        --results results/benchmark_*.json \
        --output docs/BENCHMARK_LEADERBOARD.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HEADER = """# Benchmark Leaderboard

This page is auto-generated from JSON results produced by
``scripts/run_benchmark.py``.  Each row is one model/run evaluated on one
task/dataset split.

| Run | Task | N samples | Mean Dice | Mean IoU | Pixel Acc | Inf. time (s) | Timestamp |
|-----|------|-----------|-----------|----------|-----------|---------------|-----------|
"""


def _row(result: dict) -> str:
    metrics = result.get("metrics", {})
    return (
        f"| {result.get('run_name', 'unknown')} "
        f"| {result.get('task', 'unknown')} "
        f"| {result.get('n_samples', 0)} "
        f"| {metrics.get('mean_dice', '-'):.4f} "
        f"| {metrics.get('mean_iou', '-'):.4f} "
        f"| {metrics.get('pixel_accuracy', '-'):.4f} "
        f"| {result.get('inference_seconds', '-'):.3f} "
        f"| {result.get('timestamp', '-')} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown benchmark leaderboard")
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/BENCHMARK_LEADERBOARD.md"))
    args = parser.parse_args(argv)

    rows: list[str] = []
    for path in sorted(args.results):
        with open(path) as f:
            result = json.load(f)
        rows.append(_row(result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(_HEADER + "\n")
        f.write("\n".join(rows) + "\n")

    print(f"Leaderboard written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
