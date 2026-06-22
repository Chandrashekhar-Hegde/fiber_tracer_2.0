import builtins
from unittest.mock import MagicMock

import pytest

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.exceptions import BackendNotAvailableError
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom
from fiber_tracer.viz.napari_viewer import (
    add_fiber_analysis_to_viewer,
    load_results_for_viewer,
    run_napari_viewer,
)


def _make_resolved_output(tmp_path, shape=(32, 32, 32)):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=shape,
        n_fibers=2,
        fiber_diameter_um=4.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=4.0,
        regime="resolved",
    )
    summary = FiberAnalysisPipeline(config).run()
    return stack_path, out_dir, summary


def test_load_results_for_viewer_returns_expected_keys(tmp_path):
    stack_path, out_dir, summary = _make_resolved_output(tmp_path)

    results = load_results_for_viewer(str(stack_path), str(out_dir))

    assert "raw" in results
    assert "normalized" in results
    assert "labels" in results
    assert "skeleton" in results
    assert "summary" in results
    assert results["summary"]["n_labels"] == summary["n_labels"]
    assert results["raw"].shape == results["labels"].shape


def test_add_fiber_analysis_to_viewer_adds_layers(tmp_path):
    stack_path, out_dir, _summary = _make_resolved_output(tmp_path)

    viewer = MagicMock()
    add_fiber_analysis_to_viewer(viewer, str(stack_path), str(out_dir))

    names = [call.kwargs.get("name") for call in viewer.add_image.call_args_list]
    names += [call.kwargs.get("name") for call in viewer.add_labels.call_args_list]
    names += [call.kwargs.get("name") for call in viewer.add_vectors.call_args_list]

    assert "raw" in names
    assert "labels" in names
    assert "skeleton" in names
    assert "fiber_orientations" in names

    # normalized is hidden by default
    normalized_call = None
    for call in viewer.add_image.call_args_list:
        if call.kwargs.get("name") == "normalized":
            normalized_call = call
            break
    assert normalized_call is not None
    assert normalized_call.kwargs.get("visible") is False


def test_run_napari_viewer_raises_when_napari_missing(monkeypatch):
    original_import = builtins.__import__

    def fail_napari_import(name, *args, **kwargs):
        if name == "napari":
            raise ImportError("No module named 'napari'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_napari_import)

    with pytest.raises(BackendNotAvailableError):
        run_napari_viewer("data.tif", "out")
