"""Tests for reporting exporters and pipeline report contents."""

import json

import numpy as np
import pandas as pd
import pytest

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.reporting import (
    CITATIONS,
    write_csv_report,
    write_html_report,
    write_json_report,
)
from fiber_tracer.validation.phantoms import generate_fiber_phantom


@pytest.fixture
def resolved_summary():
    return {
        "regime": "resolved",
        "n_labels": 2,
        "voxel_spacing_um": (1.0, 1.0, 1.0),
        "fibers": [
            {
                "label": 1,
                "n_voxels": 100,
                "equivalent_diameter_um": 5.0,
                "orientation": [0.0, 0.0, 1.0],
            },
            {
                "label": 2,
                "n_voxels": 200,
                "equivalent_diameter_um": 7.0,
                "orientation": [1.0, 0.0, 0.0],
            },
        ],
    }


def test_write_json_report_roundtrip(tmp_path, resolved_summary):
    path = tmp_path / "summary.json"
    write_json_report(path, resolved_summary)
    assert path.exists()
    loaded = json.loads(path.read_text())
    # JSON round-trips tuples as lists, so compare against the JSON-normalized form.
    assert loaded == json.loads(json.dumps(resolved_summary))


def test_write_csv_report_resolved_columns(tmp_path, resolved_summary):
    path = tmp_path / "report.csv"
    write_csv_report(path, resolved_summary)
    assert path.exists()
    df = pd.read_csv(path)
    expected_columns = {"regime", "label", "n_voxels", "equivalent_diameter_um", "orientation"}
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 2
    assert set(df["regime"]) == {"resolved"}


def test_write_csv_report_empty_summary(tmp_path):
    path = tmp_path / "report.csv"
    write_csv_report(path, {})
    assert path.exists()
    df = pd.read_csv(path)
    assert list(df.columns) == ["regime"]
    assert len(df) == 0


@pytest.mark.parametrize(
    "regime, caveat_snippet",
    [
        ("resolved", "Overlapping or sub-voxel fibers"),
        ("marginal", "Accuracy degrades when the fiber diameter"),
        ("subvoxel", "Only population-level orientation statistics"),
    ],
)
def test_write_html_report_contains_regime_and_caveat(tmp_path, regime, caveat_snippet):
    summary = {"regime": regime, "n_labels": 3}
    path = tmp_path / "report.html"
    write_html_report(path, summary)
    assert path.exists()
    html = path.read_text()
    assert regime in html
    assert caveat_snippet in html


def test_write_html_report_contains_citation(tmp_path):
    summary = {"regime": "resolved", "n_labels": 3}
    path = tmp_path / "report.html"
    write_html_report(path, summary)
    assert path.exists()
    html = path.read_text()
    assert CITATIONS[0] in html


def test_pipeline_summary_json_contains_config_and_citations(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(32, 32, 32),
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

    with open(out_dir / "summary.json") as f:
        disk_summary = json.load(f)

    assert "config" in disk_summary
    assert disk_summary["config"] == config.to_dict()
    assert "citations" in disk_summary
    assert disk_summary["citations"] == CITATIONS
    assert summary["config"] == config.to_dict()
    assert summary["citations"] == CITATIONS


def test_pipeline_summary_json_contains_caveats_after_resolved_run(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(32, 32, 32),
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

    with open(out_dir / "summary.json") as f:
        disk_summary = json.load(f)

    assert "caveats" in disk_summary
    assert isinstance(disk_summary["caveats"], str)
    assert len(disk_summary["caveats"]) > 0
    assert "caveats" in summary
    assert summary["caveats"] == disk_summary["caveats"]


def test_marginal_csv_contains_per_window_columns(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"

    phantom = generate_fiber_phantom(
        shape=(48, 48, 48),
        n_fibers=3,
        fiber_diameter_um=2.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    stack_path = data_dir / "input.tif"
    save_tiff_stack(stack_path, phantom.volume)

    config = Config(
        data_path=str(stack_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=2.0,
        regime="marginal",
    )
    FiberAnalysisPipeline(config).run()

    df = pd.read_csv(out_dir / "report.csv")
    expected_columns = {"regime", "window_id", "center_z", "center_y", "center_x", "fa"}
    assert expected_columns.issubset(set(df.columns))
    assert len(df) > 0
