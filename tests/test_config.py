# tests/test_config.py
import pytest
from fiber_tracer.config import Config, VoxelSpacing


def test_default_config_validates_with_existing_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = Config(data_path=str(data_dir), output_dir=str(tmp_path / "out"))
    cfg.validate()


def test_invalid_voxel_spacing_raises():
    cfg = Config()
    cfg.voxel_spacing_um = VoxelSpacing(-1, 1, 1)
    with pytest.raises(ValueError):
        cfg.validate()
