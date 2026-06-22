import pytest
import yaml

from fiber_tracer.batch import load_batch_config, process_batch
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_load_batch_config_yaml(tmp_path):
    config_path = tmp_path / "batch.yaml"
    config_data = {
        "common": {"voxel_spacing_um": [1.0, 1.0, 1.0], "fiber_diameter_um": 4.0},
        "volumes": [
            {"data_path": "/data/vol1.tif", "output_dir": "/out/vol1"},
            {"data_path": "/data/vol2.tif", "output_dir": "/out/vol2"},
        ],
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(config_data, f)

    loaded = load_batch_config(str(config_path))

    assert loaded["common"]["fiber_diameter_um"] == 4.0
    assert len(loaded["volumes"]) == 2
    assert loaded["volumes"][0]["data_path"] == "/data/vol1.tif"


def test_process_batch_runs_two_volumes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    stack_paths = []
    for i in range(2):
        phantom = generate_fiber_phantom(
            shape=(32, 32, 32),
            n_fibers=2,
            fiber_diameter_um=4.0,
            voxel_spacing_um=(1.0, 1.0, 1.0),
            seed=40 + i,
        )
        stack_path = data_dir / f"input_{i}.tif"
        save_tiff_stack(stack_path, phantom.volume)
        stack_paths.append(stack_path)

    batch_config = {
        "common": {
            "voxel_spacing_um": [1.0, 1.0, 1.0],
            "fiber_diameter_um": 4.0,
            "regime": "resolved",
        },
        "volumes": [
            {"data_path": str(stack_paths[0]), "output_dir": str(out_dir / "vol0")},
            {"data_path": str(stack_paths[1]), "output_dir": str(out_dir / "vol1")},
        ],
    }
    config_path = tmp_path / "batch.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(batch_config, f)

    aggregate_csv = tmp_path / "summary.csv"
    results = process_batch(str(config_path), aggregate_csv=str(aggregate_csv))

    assert len(results) == 2
    assert all(r["regime"] == "resolved" for r in results)
    assert aggregate_csv.exists()

    import pandas as pd

    df = pd.read_csv(aggregate_csv)
    expected_columns = {"data_path", "output_dir", "regime", "n_labels", "elapsed_s"}
    assert set(df.columns) == expected_columns
    assert len(df) == 2


def test_process_batch_empty_volumes_raises(tmp_path):
    config_path = tmp_path / "empty_batch.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump({"common": {}, "volumes": []}, f)

    with pytest.raises(ValueError, match="Batch config must contain a 'volumes' list"):
        process_batch(str(config_path))
