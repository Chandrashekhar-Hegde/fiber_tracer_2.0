# tests/test_config.py
import pytest

from fiber_tracer.config import (
    AnalysisConfig,
    Config,
    OrientationConfig,
    ProcessingConfig,
    SegmentationConfig,
    VoxelSpacing,
)


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


def test_config_round_trip_yaml(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = Config(
        data_path=str(data_dir),
        output_dir=str(tmp_path / "out"),
        voxel_spacing_um=VoxelSpacing(0.5, 0.6, 1.0),
        fiber_diameter_um=12.0,
        regime="resolved",
        processing=ProcessingConfig(
            denoise_sigma=1.2,
            normalize=False,
            anisotropic_spacing=VoxelSpacing(0.5, 0.6, 1.0),
        ),
        segmentation=SegmentationConfig(
            method="watershed",
            min_fiber_diameter_um=5.0,
            max_fiber_diameter_um=30.0,
            watershed_seed_sigma_um=2.0,
        ),
        orientation=OrientationConfig(method="pca", sigma_um=1.0, rho_um=2.0, window_size_um=3.0),
        analysis=AnalysisConfig(
            compute_morphometry=True,
            compute_orientation_tensor=False,
            compute_tda_descriptors=True,
        ),
    )

    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = Config.from_file(path)

    loaded.validate()

    assert isinstance(loaded.voxel_spacing_um, VoxelSpacing)
    assert loaded.voxel_spacing_um.z == pytest.approx(0.5)
    assert loaded.voxel_spacing_um.y == pytest.approx(0.6)
    assert loaded.voxel_spacing_um.x == pytest.approx(1.0)

    assert isinstance(loaded.processing, ProcessingConfig)
    assert loaded.processing.denoise_sigma == pytest.approx(1.2)
    assert loaded.processing.normalize is False
    assert isinstance(loaded.processing.anisotropic_spacing, VoxelSpacing)
    assert loaded.processing.anisotropic_spacing.x == pytest.approx(1.0)

    assert isinstance(loaded.segmentation, SegmentationConfig)
    assert loaded.segmentation.method == "watershed"
    assert loaded.segmentation.watershed_seed_sigma_um == pytest.approx(2.0)

    assert isinstance(loaded.orientation, OrientationConfig)
    assert loaded.orientation.method == "pca"
    assert loaded.orientation.window_size_um == pytest.approx(3.0)

    assert isinstance(loaded.analysis, AnalysisConfig)
    assert loaded.analysis.compute_morphometry is True
    assert loaded.analysis.compute_orientation_tensor is False
    assert loaded.analysis.compute_tda_descriptors is True

    assert loaded.fiber_diameter_um == pytest.approx(12.0)
    assert loaded.regime == "resolved"


def test_config_from_dict_accepts_voxel_spacing_as_list():
    """voxel_spacing_um may be supplied as [z, y, x] for YAML/JSON brevity."""
    cfg = Config.from_dict(
        {
            "data_path": "dummy.tif",
            "output_dir": "out",
            "voxel_spacing_um": [0.5, 0.6, 1.0],
        }
    )
    assert isinstance(cfg.voxel_spacing_um, VoxelSpacing)
    assert cfg.voxel_spacing_um.z == pytest.approx(0.5)
    assert cfg.voxel_spacing_um.y == pytest.approx(0.6)
    assert cfg.voxel_spacing_um.x == pytest.approx(1.0)


def test_config_from_dict_accepts_voxel_spacing_as_tuple():
    cfg = Config.from_dict(
        {
            "data_path": "dummy.tif",
            "output_dir": "out",
            "voxel_spacing_um": (2.0, 2.0, 2.0),
        }
    )
    assert isinstance(cfg.voxel_spacing_um, VoxelSpacing)
    assert cfg.voxel_spacing_um.z == pytest.approx(2.0)
