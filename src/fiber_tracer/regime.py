"""Select analysis regime from physical voxel/fiber ratio."""

from fiber_tracer.config import Config


VALID_REGIMES = ("auto", "resolved", "marginal", "subvoxel")


def detect_regime(config: Config) -> str:
    """Return one of: resolved, marginal, subvoxel."""
    spacing = min(config.voxel_spacing_um.z, config.voxel_spacing_um.y, config.voxel_spacing_um.x)
    ratio = spacing / config.fiber_diameter_um
    if ratio <= 0.3:
        return "resolved"
    elif ratio <= 3.0:
        return "marginal"
    return "subvoxel"


def validate_regime(regime: str) -> None:
    """Raise ValueError if *regime* is not a valid regime identifier."""
    if regime not in VALID_REGIMES:
        raise ValueError(f"invalid regime {regime!r}; expected one of {VALID_REGIMES}")
