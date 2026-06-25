"""Run a benchmark evaluation and write JSON results.

Example:
    python scripts/run_benchmark.py \
        --checkpoint /tmp/fx_smoke/checkpoint.pt \
        --corpus /tmp/synth_corpus \
        --task segment \
        --output results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fiber_tracer.benchmark.runner import BenchmarkRunner, build_synthetic_loader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a FiberTracer-X checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--task", choices=["segment", "orient"], default="segment")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="fibertracer-x")
    args = parser.parse_args(argv)

    runner = BenchmarkRunner.from_fibertracer_x_checkpoint(
        checkpoint_path=args.checkpoint,
        task_name=args.task,
        device=args.device,
    )
    loader = build_synthetic_loader(
        corpus_dir=args.corpus,
        split="val",
        batch_size=args.batch_size,
    )
    result = runner.run(loader, output_path=args.output, run_name=args.run_name)

    print(json.dumps(result, indent=2))
    print(f"Result written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
