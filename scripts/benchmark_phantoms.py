"""Benchmark all three RAFA regimes against a synthetic phantom."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import load_tiff_stack, save_tiff_stack
from fiber_tracer.orientation.tensor import fractional_anisotropy
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.benchmark import _align_labels, mean_dice_per_label
from fiber_tracer.validation.metrics import mean_angular_error
from fiber_tracer.validation.phantoms import FiberPhantom, generate_fiber_phantom


def _config_for_phantom(phantom: FiberPhantom, output_dir: Path, regime: str) -> Config:
    """Build a Config for *regime* using the phantom's physical parameters."""
    return Config(
        data_path=str(output_dir / "input.tif"),
        output_dir=str(output_dir),
        voxel_spacing_um=VoxelSpacing(*phantom.voxel_spacing_um),
        fiber_diameter_um=phantom.fiber_diameter_um,
        regime=regime,
    )


def run_resolved_benchmark(phantom: FiberPhantom, output_dir: Path) -> dict:
    """Run resolved pipeline and report Dice and mean angular error."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "input.tif"
    save_tiff_stack(data_path, phantom.volume)

    config = _config_for_phantom(phantom, output_dir, "resolved")
    summary = FiberAnalysisPipeline(config).run()

    pred_labels_path = output_dir / "labels.tif"
    if not pred_labels_path.exists():
        raise FileNotFoundError(f"Pipeline did not write {pred_labels_path}")

    pred_labels = load_tiff_stack(pred_labels_path).astype(np.int32)
    true_labels = phantom.labels

    aligned_pred_labels, mapping = _align_labels(pred_labels, true_labels)
    mean_dice = mean_dice_per_label(aligned_pred_labels, true_labels)

    orientation_by_label = {fiber["label"]: fiber["orientation"] for fiber in summary["fibers"]}
    pred_orientations = []
    true_orientations = []
    for true_id in np.setdiff1d(np.unique(true_labels), [0]):
        pred_id = next((p for p, t in mapping.items() if t == true_id), None)
        if pred_id is None or pred_id not in orientation_by_label:
            raise RuntimeError(f"Missing predicted orientation for label {true_id}")
        pred_orientations.append(np.asarray(orientation_by_label[pred_id]))
        true_orientations.append(phantom.orientations[true_id - 1])

    mean_angle = mean_angular_error(np.array(pred_orientations), np.array(true_orientations))

    return {
        "regime": "resolved",
        "mean_dice": float(mean_dice),
        "mean_angular_error_deg": float(mean_angle),
        "n_pred_labels": int(summary["n_labels"]),
        "n_true_labels": int(len(np.setdiff1d(np.unique(true_labels), [0]))),
    }


def run_marginal_benchmark(phantom: FiberPhantom, output_dir: Path) -> dict:
    """Run marginal pipeline and report global A2 / FA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tiff_stack(output_dir / "input.tif", phantom.volume)

    config = _config_for_phantom(phantom, output_dir, "marginal")
    FiberAnalysisPipeline(config).run()

    a2_map_path = output_dir / "a2_map.npy"
    a2_map = np.load(a2_map_path)

    if a2_map.size == 0:
        global_a2 = np.zeros((3, 3), dtype=np.float64)
        n_windows = 0
    else:
        # Average window-level A2 tensors to a single global tensor.
        global_a2 = a2_map.mean(axis=(0, 1, 2))
        n_windows = int(np.prod(a2_map.shape[:3]))

    global_fa = fractional_anisotropy(global_a2)

    return {
        "regime": "marginal",
        "global_fa": float(global_fa),
        "a2_map_shape": tuple(a2_map.shape),
        "n_windows": n_windows,
    }


def run_subvoxel_benchmark(phantom: FiberPhantom, output_dir: Path) -> dict:
    """Run subvoxel pipeline and report global A2 / FA / orientation distribution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tiff_stack(output_dir / "input.tif", phantom.volume)

    config = _config_for_phantom(phantom, output_dir, "subvoxel")
    FiberAnalysisPipeline(config).run()

    summary_path = output_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    return {
        "regime": "subvoxel",
        "fa": float(summary["fa"]),
        "a2": summary["a2"],
        "orientation_distribution": summary.get("orientation_distribution"),
    }


def _print_markdown_table(results: list[dict[str, Any]]) -> None:
    """Print a markdown table summarising the benchmark results."""
    headers = [
        "Regime",
        "mean_dice",
        "mean_angular_error_deg",
        "global_fa",
        "n_windows",
        "fa",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["regime"],
                f"{result.get('mean_dice', '-'):.4f}" if "mean_dice" in result else "-",
                (
                    f"{result.get('mean_angular_error_deg', '-'):.4f}"
                    if "mean_angular_error_deg" in result
                    else "-"
                ),
                f"{result.get('global_fa', '-'):.4f}" if "global_fa" in result else "-",
                str(result.get("n_windows", "-")),
                f"{result.get('fa', '-'):.4f}" if "fa" in result else "-",
            ]
        )

    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join([" --- " for _ in headers]) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark all three RAFA regimes against a synthetic phantom."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results",
        help="Directory to write benchmark_results.json to (default: benchmark_results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    phantom = generate_fiber_phantom(
        shape=(96, 96, 96),
        n_fibers=5,
        fiber_diameter_um=6.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output_dir = Path(tmpdir) / "benchmark"
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            run_resolved_benchmark(phantom, tmp_output_dir / "resolved"),
            run_marginal_benchmark(phantom, tmp_output_dir / "marginal"),
            run_subvoxel_benchmark(phantom, tmp_output_dir / "subvoxel"),
        ]

    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_markdown_table(results)

    resolved = results[0]
    assert resolved["mean_dice"] > 0.85, f"mean_dice {resolved['mean_dice']:.4f} <= 0.85"
    assert (
        resolved["mean_angular_error_deg"] < 5.0
    ), f"mean_angular_error {resolved['mean_angular_error_deg']:.4f} >= 5.0"

    return 0


if __name__ == "__main__":
    sys.exit(main())
