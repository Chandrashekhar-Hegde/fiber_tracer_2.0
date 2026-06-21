"""Tests for reporting exporters."""

import json

import pandas as pd
import pytest

from fiber_tracer.reporting.csv import write_csv_report
from fiber_tracer.reporting.html import write_html_report
from fiber_tracer.reporting.json import write_json_report


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
