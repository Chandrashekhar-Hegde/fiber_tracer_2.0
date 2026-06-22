"""Tests for the GF-PA66 validation harness scripts."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Load harness scripts from the project scripts directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


download_gfpa66 = _load_script("download_gfpa66")
validate_gfpa66 = _load_script("validate_gfpa66")


@pytest.fixture
def synthetic_h5(tmp_path):
    """Create a small synthetic HDF5 volume for pipeline testing."""
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "synthetic_gfpa66.h5"
    volume = np.zeros((20, 20, 20), dtype=np.uint16)
    # Add a bright central rod-like region to ensure some foreground after Otsu.
    volume[6:14, 8:12, 8:12] = 4000
    # Add modest background noise.
    rng = np.random.default_rng(0)
    volume = (volume + rng.integers(0, 200, size=volume.shape).astype(np.uint16)).astype(np.uint16)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("data", data=volume)
    return h5_path


def test_download_gfpa66_main_prints_metadata(capsys):
    download_gfpa66.main()
    captured = capsys.readouterr().out
    assert download_gfpa66.ZENODO_RECORD_URL in captured
    assert "CC BY-SA 4.0" in captured
    assert "GF-PA66_3D_XCT.h5" in captured


def test_validate_gfpa66_main_runs_pipeline(synthetic_h5, tmp_path):
    out_dir = tmp_path / "results"
    exit_code = validate_gfpa66.main(
        [
            "--data",
            str(synthetic_h5),
            "--output",
            str(out_dir),
            "--voxel-spacing",
            "1",
            "1",
            "1",
            "--fiber-diameter",
            "4",
            "--regime",
            "resolved",
        ]
    )
    assert exit_code == 0
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)
    assert summary["regime"] == "resolved"
    assert "config" in summary


def test_validate_gfpa66_main_auto_detects_dataset(tmp_path):
    """Auto-detection should fall back to the only dataset in the file."""
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "single_dataset.h5"
    volume = np.zeros((10, 10, 10), dtype=np.uint8)
    volume[3:7, 3:7, 3:7] = 200
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("custom", data=volume)

    out_dir = tmp_path / "results"
    exit_code = validate_gfpa66.main(
        [
            "--data",
            str(h5_path),
            "--output",
            str(out_dir),
            "--voxel-spacing",
            "1",
            "1",
            "1",
            "--fiber-diameter",
            "4",
            "--regime",
            "resolved",
        ]
    )
    assert exit_code == 0
    assert (out_dir / "summary.json").exists()
