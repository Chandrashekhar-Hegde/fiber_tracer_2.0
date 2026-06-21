# tests/test_regime.py
import pytest

from fiber_tracer.config import Config, VoxelSpacing, validate_regime
from fiber_tracer.regime import detect_regime


def test_detect_regime_resolved():
    cfg = Config(voxel_spacing_um=VoxelSpacing(0.1, 0.1, 0.1), fiber_diameter_um=10.0)
    assert detect_regime(cfg) == "resolved"


def test_detect_regime_marginal():
    cfg = Config(voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0), fiber_diameter_um=1.0)
    assert detect_regime(cfg) == "marginal"


def test_detect_regime_subvoxel():
    cfg = Config(voxel_spacing_um=VoxelSpacing(5.0, 5.0, 5.0), fiber_diameter_um=1.0)
    assert detect_regime(cfg) == "subvoxel"


def test_detect_regime_uses_minimum_anisotropic_spacing():
    # z is the largest, but x is the smallest and drives the regime classification.
    cfg = Config(voxel_spacing_um=VoxelSpacing(5.0, 1.0, 0.1), fiber_diameter_um=1.0)
    assert detect_regime(cfg) == "resolved"


@pytest.mark.parametrize("regime", ["auto", "resolved", "marginal", "subvoxel"])
def test_validate_regime_accepts_valid(regime):
    validate_regime(regime)  # should not raise


@pytest.mark.parametrize("regime", ["", "unknown", "RESOLVED", "sub-voxel", "automatic"])
def test_validate_regime_rejects_invalid(regime):
    with pytest.raises(ValueError):
        validate_regime(regime)
