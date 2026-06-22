"""Tests for Plotly-based interactive report visualizations."""

import json

import numpy as np
import pytest

pytest.importorskip("plotly")

from fiber_tracer.viz.plotly_plots import (
    generate_interactive_report,
    plot_a2_ellipsoid,
    plot_fiber_property_histogram,
    plot_orientation_distribution,
)


@pytest.fixture
def resolved_summary():
    return {
        "regime": "resolved",
        "n_labels": 2,
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


@pytest.fixture
def subvoxel_summary():
    a2 = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=float,
    )
    return {
        "regime": "subvoxel",
        "a2": a2.tolist(),
        "fa": 0.25,
    }


def test_plot_orientation_distribution_creates_html(tmp_path, resolved_summary):
    output = tmp_path / "orientation.html"
    plot_orientation_distribution(resolved_summary, str(output))
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()


def test_plot_fiber_property_histogram_empty_summary(tmp_path):
    output = tmp_path / "empty.html"
    plot_fiber_property_histogram({}, "equivalent_diameter_um", str(output))
    assert output.exists()


def test_plot_a2_ellipsoid_creates_html(tmp_path):
    output = tmp_path / "a2.html"
    a2 = np.array(
        [
            [0.6, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=float,
    )
    plot_a2_ellipsoid(a2, str(output))
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "Surface" in html or "surface" in html.lower()


def test_generate_interactive_report_resolved(tmp_path, resolved_summary):
    output = tmp_path / "report.html"
    generate_interactive_report(resolved_summary, str(output))
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert html.count('class="plotly-graph-div"') >= 2


def test_generate_interactive_report_subvoxel(tmp_path, subvoxel_summary):
    output = tmp_path / "report.html"
    generate_interactive_report(subvoxel_summary, str(output))
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "plotly" in html.lower()
    assert "a2" in html.lower()


def test_generate_interactive_report_empty(tmp_path):
    output = tmp_path / "report.html"
    generate_interactive_report({}, str(output))
    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "No visualizations available" in html


def test_generate_interactive_report_roundtrip_json(tmp_path, resolved_summary):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(resolved_summary))
    output = tmp_path / "report.html"

    loaded = json.loads(summary_path.read_text())
    generate_interactive_report(loaded, str(output))

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert 'id="orientation-distribution"' in html
    assert 'id="diameter-distribution"' in html
