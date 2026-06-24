"""Select analysis regime from physical voxel/fiber ratio."""

from __future__ import annotations

from fiber_tracer.config import VALID_REGIMES, Config, validate_regime

__all__ = ["detect_regime", "VALID_REGIMES", "validate_regime"]


def detect_regime(config: Config) -> str:
    """Return one of: resolved, marginal, subvoxel."""
    spacing = min(config.voxel_spacing_um.z, config.voxel_spacing_um.y, config.voxel_spacing_um.x)
    ratio = spacing / config.fiber_diameter_um
    if ratio <= 0.3:
        return "resolved"
    elif ratio <= 3.0:
        return "marginal"
    return "subvoxel"
