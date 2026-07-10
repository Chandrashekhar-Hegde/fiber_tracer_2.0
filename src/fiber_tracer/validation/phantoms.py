"""Synthetic fiber phantoms with ground truth and XCT domain randomization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FiberPhantom:
    volume: np.ndarray
    labels: np.ndarray
    orientations: np.ndarray  # Nx3 unit vectors
    diameters_um: np.ndarray
    lengths_um: np.ndarray
    voxel_spacing_um: tuple[float, float, float]
    fiber_diameter_um: float
    void_mask: np.ndarray | None = None  # added for multi-class pre-training


def generate_straight_fiber(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    direction: np.ndarray,
    radius_voxels: float,
    intensity: float = 1.0,
) -> np.ndarray:
    """Draw a single straight cylinder in a binary volume."""
    direction = direction / np.linalg.norm(direction)
    z, y, x = np.indices(shape, dtype=float)
    coords = np.stack([z, y, x], axis=-1)
    center_vec = np.array(center)
    to_center = coords - center_vec
    projection = np.dot(to_center, direction)
    perpendicular = to_center - projection[:, :, :, None] * direction
    distance = np.linalg.norm(perpendicular, axis=-1)
    volume = np.zeros(shape, dtype=float)
    volume[distance <= radius_voxels] = intensity
    return volume


def _sample_direction(rng: np.random.Generator, mode: str, axis: int = 2) -> np.ndarray:
    """Return a unit fiber direction according to *mode*."""
    if mode == "random":
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        return direction / norm
    if mode == "aligned":
        direction = np.zeros(3)
        direction[axis] = 1.0
        return direction
    if mode == "in_plane":
        # Random orientation within the XY plane.
        angle = rng.uniform(0, 2 * np.pi)
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "angle":
        # In-plane at a fixed angle (degrees) around Z.
        angle = np.deg2rad(rng.uniform(0, 360))
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "orthogonal":
        # Choose one of the three cardinal axes.
        direction = np.zeros(3)
        direction[rng.integers(0, 3)] = 1.0
        return direction
    if mode == "woven":
        # Alternating 0/90 in-plane tows.
        angle = 0.0 if rng.random() > 0.5 else np.pi / 2
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    if mode == "twill":
        # ±45° in-plane tows.
        angle = np.pi / 4 if rng.random() > 0.5 else -np.pi / 4
        return np.array([0.0, np.cos(angle), np.sin(angle)])
    raise ValueError(f"Unknown orientation mode: {mode}")


def _sample_von_mises_fisher(
    rng: np.random.Generator,
    mean_direction: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Sample a unit direction around *mean_direction* with concentration *kappa*.

    A simple rejection-free approximation: sample from a normal distribution
    centered on the mean direction and normalize.  Low *concentration* gives
    nearly isotropic directions; high concentration gives directions tightly
    clustered around the mean.
    """
    mean_direction = np.asarray(mean_direction, dtype=float)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    if concentration <= 0:
        return _sample_direction(rng, "random")
    sample = rng.normal(loc=mean_direction, scale=1.0 / concentration, size=3)
    norm = np.linalg.norm(sample)
    if norm == 0:
        return mean_direction
    return sample / norm


def _clip_fiber_length(
    shape: tuple[int, int, int],
    center: np.ndarray,
    direction: np.ndarray,
    length_um: float,
    voxel_spacing_um: tuple[float, float, float],
) -> float:
    """Return the in-bounds length for a fiber centered at *center*."""
    diag = float(np.linalg.norm(np.array(shape) * np.array(voxel_spacing_um)))
    if length_um <= 0 or length_um > diag:
        return diag
    return length_um


def _generate_finite_fiber(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    direction: np.ndarray,
    radius_voxels: float,
    length_um: float,
    voxel_spacing_um: tuple[float, float, float],
    intensity: float = 1.0,
    supersample: int = 1,
) -> np.ndarray:
    """Draw a finite-length cylinder (caps included) on the voxel grid.

    With ``supersample == 1`` each voxel is a hard in/out test at its centre, giving
    aliased (staircase) fibre boundaries. With ``supersample = s > 1`` each voxel is
    evaluated at ``s**3`` sub-voxel sample points and averaged, so edge voxels take a
    fractional value in ``(0, 1)`` — an anti-aliased rasterisation that reproduces the
    partial-volume effect seen in real X-ray CT.
    """
    direction = direction / np.linalg.norm(direction)
    center_vec = np.array(center, dtype=float)
    half_length_voxels = 0.5 * length_um / min(voxel_spacing_um)
    axes = [np.arange(n, dtype=float) for n in shape]
    # Sub-voxel sample offsets centred on the voxel: [0.0] when supersample == 1.
    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    occupancy = np.zeros(shape, dtype=float)
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                zz, yy, xx = np.meshgrid(axes[0] + dz, axes[1] + dy, axes[2] + dx, indexing="ij")
                to_center = np.stack([zz, yy, xx], axis=-1) - center_vec
                projection = to_center @ direction
                perpendicular = to_center - projection[:, :, :, None] * direction
                distance = np.linalg.norm(perpendicular, axis=-1)
                inside = (distance <= radius_voxels) & (np.abs(projection) <= half_length_voxels)
                occupancy += inside
    occupancy /= supersample**3
    return occupancy * intensity


def _add_voids(
    volume: np.ndarray,
    labels: np.ndarray,
    porosity: float,
    radius_voxels: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Insert spherical voids and return a void mask (1 = void)."""
    shape = volume.shape
    void_mask = np.zeros(shape, dtype=bool)
    if porosity <= 0:
        return void_mask
    n_voids = int(np.prod(shape) * porosity / (4 / 3 * np.pi * radius_voxels**3))
    for _ in range(max(1, n_voids)):
        void_center = rng.uniform(0, np.array(shape))
        void_radius = rng.uniform(radius_voxels, 2 * radius_voxels)
        zz, yy, xx = np.indices(shape, dtype=float)
        dist = np.sqrt(
            (zz - void_center[0]) ** 2 + (yy - void_center[1]) ** 2 + (xx - void_center[2]) ** 2
        )
        void_mask_local = dist <= void_radius
        volume[void_mask_local] = 0.0
        labels[void_mask_local] = 0
        void_mask |= void_mask_local
    return void_mask


def generate_fiber_phantom(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 10,
    fiber_diameter_um: float = 4.0,
    fiber_length_um: float = -1.0,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.02,
    orientation_mode: str = "random",
    broken_fraction: float = 0.0,
    n_broken_pieces: int = 2,
    porosity: float = 0.0,
    seed: int | None = None,
    supersample: int = 1,
) -> FiberPhantom:
    """Generate a phantom with configurable fiber architecture.

    Parameters
    ----------
    orientation_mode:
        ``random`` (default), ``aligned``, ``in_plane``, ``orthogonal``,
        ``woven`` (0/90), or ``twill`` (±45).
    fiber_length_um:
        Fiber length in micrometres.  Negative values use the volume diagonal.
    broken_fraction:
        Fraction of fibers that are broken into *n_broken_pieces* segments.
    n_broken_pieces:
        Number of segments per broken fiber.
    porosity:
        Probability of inserting spherical voids (porosity) into the matrix.
    supersample:
        Sub-voxel sampling factor for anti-aliased voxelisation.  ``1`` (default)
        gives hard, aliased fibre edges; ``>1`` averages ``supersample**3`` sub-voxel
        samples per voxel to reproduce the partial-volume effect of real X-ray CT.
    """
    rng = np.random.default_rng(seed)
    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []
    next_label = 1

    for _i in range(n_fibers):
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        direction = _sample_direction(rng, orientation_mode)
        length = _clip_fiber_length(shape, center, direction, fiber_length_um, voxel_spacing_um)

        if rng.random() < broken_fraction:
            # Split the fiber into segments along its axis.
            segment_length = length / n_broken_pieces
            half_total = length / 2
            starts = [-half_total + k * segment_length for k in range(n_broken_pieces)]
            for start in starts:
                seg_center = center + direction * (start + segment_length / 2)
                fiber = _generate_finite_fiber(
                    shape,
                    tuple(seg_center),
                    direction,
                    radius_voxels,
                    segment_length * 0.9,  # tiny gap between pieces
                    voxel_spacing_um,
                    supersample=supersample,
                )
                mask = fiber >= 0.5
                if np.any(labels[mask] > 0):
                    continue
                labels[mask] = next_label
                next_label += 1
                volume += fiber
                orientations.append(direction)
                diameters.append(fiber_diameter_um)
                lengths.append(segment_length * 0.9)
        else:
            fiber = _generate_finite_fiber(
                shape,
                tuple(center),
                direction,
                radius_voxels,
                length,
                voxel_spacing_um,
                supersample=supersample,
            )
            mask = fiber >= 0.5
            if np.any(labels[mask] > 0):
                continue
            labels[mask] = next_label
            next_label += 1
            volume += fiber
            orientations.append(direction)
            diameters.append(fiber_diameter_um)
            lengths.append(length)

    void_mask = _add_voids(volume, labels, porosity, radius_voxels, rng)
    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=fiber_diameter_um,
        void_mask=void_mask,
    )


def generate_short_fiber_phantom(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 100,
    fiber_diameter_um: float = 4.0,
    fiber_length_um: tuple[float, float] = (20.0, 80.0),
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.02,
    mean_direction: np.ndarray | None = None,
    concentration: float = 0.0,
    porosity: float = 0.0,
    seed: int | None = None,
) -> FiberPhantom:
    """Generate a short/discontinuous fiber phantom with variable lengths.

    Parameters
    ----------
    fiber_length_um:
        Min/max fiber length in micrometres.
    mean_direction:
        Optional mean orientation for a von-Mises-Fisher-like distribution.
    concentration:
        Concentration around *mean_direction* (0 = isotropic, >5 = strongly aligned).
    """
    rng = np.random.default_rng(seed)
    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []
    next_label = 1

    mean_dir = (
        np.array([0.0, 0.0, 1.0])
        if mean_direction is None
        else np.asarray(mean_direction, dtype=float)
    )

    for _i in range(n_fibers):
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        direction = _sample_von_mises_fisher(rng, mean_dir, concentration)
        length = rng.uniform(fiber_length_um[0], fiber_length_um[1])
        length = _clip_fiber_length(shape, center, direction, length, voxel_spacing_um)

        fiber = _generate_finite_fiber(
            shape, tuple(center), direction, radius_voxels, length, voxel_spacing_um
        )
        mask = fiber > 0
        if np.any(labels[mask] > 0):
            continue
        labels[mask] = next_label
        next_label += 1
        volume += fiber
        orientations.append(direction)
        diameters.append(fiber_diameter_um)
        lengths.append(length)

    void_mask = _add_voids(volume, labels, porosity, radius_voxels, rng)
    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=fiber_diameter_um,
        void_mask=void_mask,
    )


def generate_woven_bundle_phantom(
    shape: tuple[int, int, int] = (128, 128, 128),
    n_bundles: int = 8,
    bundle_diameter_um: float = 20.0,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation_mode: str = "woven",
    noise_std: float = 0.02,
    porosity: float = 0.0,
    seed: int | None = None,
) -> FiberPhantom:
    """Generate a woven/twill bundle phantom.

    Bundles are modeled as thick cylinders alternating between two in-plane
    orientations (0/90 for ``woven``, ±45 for ``twill``).  This is a
    meso-scale approximation suitable for training bundle-detection heads.
    """
    rng = np.random.default_rng(seed)
    radius_voxels = 0.5 * bundle_diameter_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []
    next_label = 1

    for i in range(n_bundles):
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        if orientation_mode == "woven":
            angle = 0.0 if i % 2 == 0 else np.pi / 2
        elif orientation_mode == "twill":
            angle = np.pi / 4 if i % 2 == 0 else -np.pi / 4
        else:
            raise ValueError(f"Unknown woven mode: {orientation_mode}")
        direction = np.array([0.0, np.cos(angle), np.sin(angle)])
        length = _clip_fiber_length(shape, center, direction, -1.0, voxel_spacing_um)

        fiber = _generate_finite_fiber(
            shape, tuple(center), direction, radius_voxels, length, voxel_spacing_um
        )
        mask = fiber > 0
        if np.any(labels[mask] > 0):
            continue
        labels[mask] = next_label
        next_label += 1
        volume += fiber
        orientations.append(direction)
        diameters.append(bundle_diameter_um)
        lengths.append(length)

    void_mask = _add_voids(volume, labels, porosity, radius_voxels, rng)
    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=bundle_diameter_um,
        void_mask=void_mask,
    )


def generate_recycled_fiber_phantom(
    shape: tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 80,
    fiber_diameter_um: tuple[float, float] = (3.0, 8.0),
    fiber_length_um: tuple[float, float] = (10.0, 60.0),
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.03,
    porosity: float = 0.005,
    seed: int | None = None,
) -> FiberPhantom:
    """Generate a recycled/discontinuous fiber phantom with variable diameter/length."""
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []
    next_label = 1

    for _i in range(n_fibers):
        diameter = rng.uniform(fiber_diameter_um[0], fiber_diameter_um[1])
        radius_voxels = 0.5 * diameter / min(voxel_spacing_um)
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        direction = _sample_direction(rng, "random")
        length = rng.uniform(fiber_length_um[0], fiber_length_um[1])
        length = _clip_fiber_length(shape, center, direction, length, voxel_spacing_um)

        fiber = _generate_finite_fiber(
            shape, tuple(center), direction, radius_voxels, length, voxel_spacing_um
        )
        mask = fiber > 0
        if np.any(labels[mask] > 0):
            continue
        labels[mask] = next_label
        next_label += 1
        volume += fiber
        orientations.append(direction)
        diameters.append(diameter)
        lengths.append(length)

    mean_diameter = float(np.mean(diameters)) if diameters else fiber_diameter_um[0]
    void_mask = _add_voids(volume, labels, porosity, 0.5 * mean_diameter, rng)
    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=mean_diameter,
        void_mask=void_mask,
    )


def generate_phantom_with_known_orientation(
    shape: tuple[int, int, int] = (64, 64, 64),
    direction: np.ndarray | None = None,
    center: tuple[float, float, float] | None = None,
    fiber_diameter_um: float = 4.0,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
) -> FiberPhantom:
    """Generate a deterministic phantom with a single fiber along a known direction.

    This is useful for validating orientation estimation because the ground-truth
    orientation is exactly the requested unit direction.
    """
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("direction must be a non-zero vector")
    direction = direction / norm

    if center is None:
        center = tuple((np.array(shape) - 1.0) / 2.0)

    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = generate_straight_fiber(shape, center, direction, radius_voxels, intensity=intensity)
    labels = (volume > 0).astype(np.int32)
    length = min(shape) * min(voxel_spacing_um)

    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array([direction]),
        diameters_um=np.array([fiber_diameter_um]),
        lengths_um=np.array([length]),
        voxel_spacing_um=voxel_spacing_um,
        fiber_diameter_um=fiber_diameter_um,
    )


# ---------------------------------------------------------------------------
# Orientation tensor and semantic-mask helpers
# ---------------------------------------------------------------------------


def compute_orientation_tensor(
    orientations: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the second-order orientation tensor A2 from unit directions.

    Parameters
    ----------
    orientations:
        Array of shape (N, 3) with unit fiber direction vectors.
    weights:
        Optional array of shape (N,) with per-fiber weights (e.g. length or
        volume).  If omitted, all fibers are weighted equally.

    Returns
    -------
    A2 : np.ndarray
        3x3 symmetric orientation tensor (Advani--Tucker second-order tensor).
    """
    orientations = np.asarray(orientations, dtype=float)
    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError("orientations must have shape (N, 3)")
    if weights is None:
        weights = np.ones(len(orientations))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    a2 = np.zeros((3, 3), dtype=float)
    for p, w in zip(orientations, weights):
        p = p / np.linalg.norm(p)
        a2 += w * np.outer(p, p)
    return a2


def semantic_mask_from_phantom(phantom: FiberPhantom) -> np.ndarray:
    """Return a semantic mask with classes 0=matrix, 1=fiber, 2=void.

    The instance labels are collapsed to the fiber class; any recorded void
    mask is assigned class 2.
    """
    mask = np.zeros(phantom.labels.shape, dtype=np.int32)
    mask[phantom.labels > 0] = 1
    if phantom.void_mask is not None:
        mask[phantom.void_mask] = 2
    return mask


# ---------------------------------------------------------------------------
# XCT domain-randomization augmentations
# ---------------------------------------------------------------------------


def add_beam_hardening(
    volume: np.ndarray,
    strength: float = 0.1,
    mode: str = "cup",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate beam-hardening cupping as a radial polynomial bias.

    Parameters
    ----------
    strength:
        Maximum intensity deviation (relative to [0, 1]).
    mode:
        ``cup`` darkens the centre, ``reverse_cup`` brightens it.
    """
    rng = rng or np.random.default_rng()
    shape = volume.shape
    center = np.array(shape) / 2.0
    zz, yy, xx = np.indices(shape, dtype=float)
    # Normalised radial distance per XY slice (average across z).
    r = np.sqrt(
        ((zz - center[0]) / shape[0]) ** 2
        + ((yy - center[1]) / shape[1]) ** 2
        + ((xx - center[2]) / shape[2]) ** 2
    )
    if mode == "cup":
        bias = -strength * (r**2)
    elif mode == "reverse_cup":
        bias = strength * (r**2)
    else:
        raise ValueError(f"Unknown beam-hardening mode: {mode}")
    return np.asarray(np.clip(volume + bias, 0.0, 1.0))


def add_partial_volume_blur(
    volume: np.ndarray,
    sigma_voxels: float = 0.7,
    anisotropic_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Blur a volume to simulate partial-volume / limited-resolution effects."""
    from scipy.ndimage import gaussian_filter

    sigma = tuple(sigma_voxels * f for f in anisotropic_factor)
    blurred = gaussian_filter(volume, sigma=sigma)
    return np.asarray(np.clip(blurred, 0.0, 1.0))


def add_poisson_noise(
    volume: np.ndarray,
    scale: float = 1000.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Poisson/quantization noise to a [0, 1] volume."""
    rng = rng or np.random.default_rng()
    scaled = volume * scale
    noisy = rng.poisson(scaled).astype(float) / scale
    return np.clip(noisy, 0.0, 1.0)


def add_ring_artifacts(
    volume: np.ndarray,
    n_rings: int = 3,
    strength: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add image-space ring artifacts to a subset of slices."""
    rng = rng or np.random.default_rng()
    volume = volume.copy()
    shape = volume.shape
    center = np.array(shape[1:]) / 2.0
    yy, xx = np.indices(shape[1:], dtype=float)
    r = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    for _ in range(n_rings):
        ring_radius = rng.uniform(min(shape[1:]) * 0.1, min(shape[1:]) * 0.45)
        ring_width = rng.uniform(1.0, 3.0)
        ring_mask = np.abs(r - ring_radius) <= ring_width
        z_slice = rng.integers(0, shape[0])
        sign = rng.choice([-1.0, 1.0])
        volume[z_slice][ring_mask] += sign * strength
    return np.asarray(np.clip(volume, 0.0, 1.0))


def add_contrast_jitter(
    volume: np.ndarray,
    gamma_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    brightness_range: tuple[float, float] = (-0.05, 0.05),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random gamma, contrast, and brightness jitter."""
    rng = rng or np.random.default_rng()
    gamma = rng.uniform(*gamma_range)
    contrast = rng.uniform(*contrast_range)
    brightness = rng.uniform(*brightness_range)
    volume = np.clip(volume**gamma, 0.0, 1.0)
    mean = float(volume.mean())
    volume = np.clip((volume - mean) * contrast + mean + brightness, 0.0, 1.0)
    return volume


def apply_xct_domain_randomization(
    volume: np.ndarray,
    seed: int | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """Apply a random combination of XCT artifact augmentations.

    Parameters
    ----------
    intensity:
        Overall strength multiplier (0 = no artifacts, 1 = full defaults).
    """
    rng = np.random.default_rng(seed)
    v = volume.copy()
    if rng.random() < 0.5 * intensity:
        v = add_beam_hardening(v, strength=rng.uniform(0.03, 0.12) * intensity, rng=rng)
    if rng.random() < 0.5 * intensity:
        v = add_partial_volume_blur(v, sigma_voxels=rng.uniform(0.3, 1.0) * intensity)
    if rng.random() < 0.5 * intensity:
        v = add_poisson_noise(v, scale=rng.uniform(500.0, 2000.0), rng=rng)
    if rng.random() < 0.3 * intensity:
        ring_strength = rng.uniform(0.02, 0.08) * intensity
        v = add_ring_artifacts(v, n_rings=int(rng.integers(1, 4)), strength=ring_strength, rng=rng)
    if rng.random() < 0.7 * intensity:
        v = add_contrast_jitter(v, rng=rng)
    return np.asarray(np.clip(v, 0.0, 1.0))
