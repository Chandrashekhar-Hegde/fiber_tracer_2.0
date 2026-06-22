"""Napari viewer helpers for RAFA results."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from fiber_tracer.exceptions import BackendNotAvailableError
from fiber_tracer.io import load_tiff_stack


def _import_napari() -> Any:
    try:
        import napari

        return napari
    except ImportError as exc:
        raise BackendNotAvailableError("Install viz extra: pip install fiber-tracer[viz]") from exc


def load_results_for_viewer(data_path: str, output_dir: str) -> dict[str, Any]:
    """Load raw volume and pipeline outputs for napari.

    Returns a dict with keys:
    - "raw": raw input volume
    - "normalized": normalized input volume (if available)
    - "labels": segmentation labels (if available)
    - "skeleton": skeleton mask (if available)
    - "summary": parsed summary.json (if available)
    """
    out = Path(output_dir)
    results: dict[str, Any] = {"raw": load_tiff_stack(data_path)}
    for name, filename in [
        ("normalized", "normalized_input.tif"),
        ("labels", "labels.tif"),
        ("skeleton", "skeleton.tif"),
    ]:
        path = out / filename
        if path.exists():
            results[name] = load_tiff_stack(path)
    summary_path = out / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results["summary"] = json.load(f)
    return results


def add_fiber_analysis_to_viewer(
    viewer: Any,
    data_path: str,
    output_dir: str,
) -> None:
    """Add RAFA output layers to an existing napari viewer."""
    results = load_results_for_viewer(data_path, output_dir)
    viewer.add_image(results["raw"], name="raw")
    if "normalized" in results:
        viewer.add_image(results["normalized"], name="normalized", visible=False)
    if "labels" in results:
        viewer.add_labels(results["labels"].astype(np.int32), name="labels")
    if "skeleton" in results:
        viewer.add_labels(results["skeleton"].astype(np.int32), name="skeleton")
    if "summary" in results and "fibers" in results["summary"]:
        # Add orientation vectors as a vectors layer anchored at each fiber centroid.
        fibers = results["summary"]["fibers"]
        labels = results.get("labels")
        if fibers and labels is not None:
            vectors = []
            for fiber in fibers:
                orientation = np.asarray(fiber["orientation"], dtype=float)
                label_id = fiber["label"]
                coords = np.argwhere(labels == label_id)
                if len(coords) == 0:
                    continue
                centroid = coords.mean(axis=0)
                vectors.append(np.concatenate([centroid, orientation]))
            if vectors:
                viewer.add_vectors(np.array(vectors), name="fiber_orientations", length=10)


def run_napari_viewer(data_path: str, output_dir: str) -> int:
    """Launch a napari viewer with RAFA results."""
    napari = _import_napari()
    viewer = napari.Viewer()
    add_fiber_analysis_to_viewer(viewer, data_path, output_dir)
    napari.run()
    return 0
