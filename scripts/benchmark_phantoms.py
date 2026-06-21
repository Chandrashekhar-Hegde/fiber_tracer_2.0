"""Benchmark the resolved-regime pipeline against a synthetic phantom."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import load_tiff_stack, save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.metrics import mean_angular_error, mean_dice_score
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def _align_labels(
    pred_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """Remap predicted label IDs to match ground-truth IDs by overlap.

    Returns a relabeled prediction volume and the mapping used.  Each true
    foreground label is matched to the predicted label with which it shares the
    most voxels.
    """
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)
    true_ids = np.setdiff1d(np.unique(true_labels), [0])
    pred_ids = np.setdiff1d(np.unique(pred_labels), [0])

    mapping: dict[int, int] = {}
    used_pred_ids: set = set()
    for true_id in true_ids:
        true_mask = true_labels == true_id
        best_pred = None
        best_overlap = 0
        for pred_id in pred_ids:
            if pred_id in used_pred_ids:
                continue
            overlap = int(np.sum(true_mask & (pred_labels == pred_id)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred = pred_id
        if best_pred is None:
            raise RuntimeError(f"Could not find a predicted label overlapping true label {true_id}")
        mapping[best_pred] = int(true_id)
        used_pred_ids.add(best_pred)

    aligned = np.zeros_like(pred_labels)
    for pred_id, true_id in mapping.items():
        aligned[pred_labels == pred_id] = true_id
    return aligned, mapping


def main() -> int:
    phantom = generate_fiber_phantom(
        shape=(96, 96, 96),
        n_fibers=5,
        fiber_diameter_um=6.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "phantom.tif"
        output_dir = tmpdir / "output"
        save_tiff_stack(data_path, phantom.volume)

        config = Config(
            data_path=str(data_path),
            output_dir=str(output_dir),
            voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
            fiber_diameter_um=6.0,
            regime="resolved",
        )

        pipeline = FiberAnalysisPipeline(config)
        summary = pipeline.run()

        pred_labels_path = output_dir / "labels.tif"
        if not pred_labels_path.exists():
            raise FileNotFoundError(f"Pipeline did not write {pred_labels_path}")

        pred_labels = load_tiff_stack(pred_labels_path).astype(np.int32)
        true_labels = phantom.labels

        aligned_pred_labels, mapping = _align_labels(pred_labels, true_labels)
        mean_dice = mean_dice_score(aligned_pred_labels, true_labels)

        orientation_by_label = {fiber["label"]: fiber["orientation"] for fiber in summary["fibers"]}
        pred_orientations = []
        true_orientations = []
        for true_id in np.setdiff1d(np.unique(true_labels), [0]):
            # Find the predicted label that was aligned to this true label.
            pred_id = None
            for p, t in mapping.items():
                if t == true_id:
                    pred_id = p
                    break
            if pred_id is None or pred_id not in orientation_by_label:
                raise RuntimeError(f"Missing predicted orientation for label {true_id}")
            pred_orientations.append(np.asarray(orientation_by_label[pred_id]))
            true_orientations.append(phantom.orientations[true_id - 1])

        mean_angle = mean_angular_error(np.array(pred_orientations), np.array(true_orientations))

        report = {
            "mean_dice": float(mean_dice),
            "mean_angular_error_deg": float(mean_angle),
            "n_pred_labels": int(summary["n_labels"]),
            "n_true_labels": int(len(np.setdiff1d(np.unique(true_labels), [0]))),
        }
        print(json.dumps(report, indent=2))

        assert mean_dice > 0.85, f"mean_dice {mean_dice:.4f} <= 0.85"
        assert mean_angle < 5.0, f"mean_angular_error {mean_angle:.4f} >= 5.0"

    return 0


if __name__ == "__main__":
    sys.exit(main())
