"""Regression test for the phantom benchmark harness."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_phantom_benchmark_meets_acceptance_thresholds():
    """Run scripts/benchmark_phantoms.py and assert resolved-regime thresholds."""
    output_dir = REPO_ROOT / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_phantoms.py",
            "--output",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Benchmark script failed with exit code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    # The script prints a markdown table; we check the JSON it writes.
    results_path = output_dir / "benchmark_results.json"
    assert results_path.exists(), f"Expected {results_path} to be created"
    with open(results_path) as f:
        data = json.load(f)
    resolved = next((r for r in data if r.get("regime") == "resolved"), None)
    assert resolved is not None, "No resolved-regime result found"
    assert resolved["mean_dice"] > 0.85
    assert resolved["mean_angular_error_deg"] < 5.0
    # Ensure the markdown table was emitted to stdout.
    assert "| Regime |" in result.stdout
