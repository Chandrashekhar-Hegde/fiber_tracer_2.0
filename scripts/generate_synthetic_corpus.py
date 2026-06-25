"""Generate a synthetic XCT corpus for FiberTracer-X pre-training.

The corpus mixes several fiber architectures (UD continuous, short-fiber,
woven bundles, recycled/discontinuous) and applies physics-based XCT domain
randomization.  Each sample is saved as a 3D patch with a semantic mask,
instance labels, and a local orientation tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from fiber_tracer.validation.phantoms import (
    FiberPhantom,
    apply_xct_domain_randomization,
    compute_orientation_tensor,
    generate_fiber_phantom,
    generate_recycled_fiber_phantom,
    generate_short_fiber_phantom,
    generate_woven_bundle_phantom,
    semantic_mask_from_phantom,
)

PATCH_SIZE = (64, 64, 64)
VOXEL_SPACING = (1.0, 1.0, 1.0)


def _crop_center(volume: np.ndarray, patch_size: tuple[int, int, int]) -> np.ndarray:
    """Crop the centre of a volume to *patch_size*."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    z0 = (d - pd) // 2
    y0 = (h - ph) // 2
    x0 = (w - pw) // 2
    return volume[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw]


def _local_orientation_tensor(
    patch_labels: np.ndarray,
    phantom: FiberPhantom,
) -> np.ndarray:
    """Compute the A2 tensor for the fibers present in *patch_labels*."""
    present = np.unique(patch_labels[patch_labels > 0])
    if len(present) == 0:
        return np.eye(3, dtype=float) / 3.0
    directions = []
    weights = []
    # Phantom labels are contiguous starting at 1; orientations are stored
    # in the same order.
    for label in present:
        idx = int(label) - 1
        if 0 <= idx < len(phantom.orientations):
            directions.append(phantom.orientations[idx])
            weights.append(float(phantom.lengths_um[idx]))
    if not directions:
        return np.eye(3, dtype=float) / 3.0
    return compute_orientation_tensor(np.array(directions), np.array(weights))


def _generate_sample(
    sample_id: int,
    output_dir: Path,
    patch_size: tuple[int, int, int],
) -> dict | None:
    """Generate one synthetic sample and return its registry entry."""
    rng = np.random.default_rng(sample_id)

    # Choose architecture randomly, weighted toward simple cases for stability.
    architecture = rng.choice(
        ["ud", "short", "woven", "recycled"],
        p=[0.25, 0.35, 0.20, 0.20],
    )

    # Generate a larger volume so we can crop a clean central patch.
    shape = tuple(2 * s for s in patch_size)
    diameter = rng.uniform(2.0, 8.0)

    phantom: FiberPhantom
    material = "generic"
    if architecture == "ud":
        phantom = generate_fiber_phantom(
            shape=shape,
            n_fibers=rng.integers(5, 20),
            fiber_diameter_um=diameter,
            orientation_mode=rng.choice(["aligned", "in_plane", "angle"]),
            porosity=rng.uniform(0.0, 0.005),
            seed=sample_id,
        )
    elif architecture == "short":
        phantom = generate_short_fiber_phantom(
            shape=shape,
            n_fibers=rng.integers(40, 150),
            fiber_diameter_um=diameter,
            fiber_length_um=(10.0, 60.0),
            concentration=rng.uniform(0.0, 4.0),
            porosity=rng.uniform(0.0, 0.005),
            seed=sample_id,
        )
        material = rng.choice(["gfrp", "cfrp"])
    elif architecture == "woven":
        phantom = generate_woven_bundle_phantom(
            shape=shape,
            n_bundles=rng.integers(6, 16),
            bundle_diameter_um=rng.uniform(12.0, 30.0),
            orientation_mode=rng.choice(["woven", "twill"]),
            porosity=rng.uniform(0.0, 0.003),
            seed=sample_id,
        )
        material = rng.choice(["gfrp", "cfrp"])
    else:  # recycled
        phantom = generate_recycled_fiber_phantom(
            shape=shape,
            n_fibers=rng.integers(50, 120),
            fiber_diameter_um=(2.0, 8.0),
            fiber_length_um=(8.0, 50.0),
            porosity=rng.uniform(0.0, 0.008),
            seed=sample_id,
        )
        material = "rcfrp"

    # Crop central patch.
    volume_clean = _crop_center(phantom.volume, patch_size)
    labels = _crop_center(phantom.labels, patch_size)
    semantic = _crop_center(semantic_mask_from_phantom(phantom), patch_size)
    a2 = _local_orientation_tensor(labels, phantom)

    # Apply domain randomization to the input volume only.
    volume = apply_xct_domain_randomization(volume_clean, seed=sample_id + 12345)

    sample_path = output_dir / f"sample_{sample_id:06d}.npz"
    np.savez_compressed(
        sample_path,
        volume=volume.astype(np.float32),
        volume_clean=volume_clean.astype(np.float32),
        semantic=semantic.astype(np.uint8),
        labels=labels.astype(np.int32),
        a2=a2.astype(np.float32),
    )

    return {
        "id": sample_id,
        "file": str(sample_path.relative_to(output_dir.parent)),
        "architecture": architecture,
        "material": material,
        "fiber_diameter_um": float(phantom.fiber_diameter_um),
        "n_labels": int(labels.max()),
        "void_fraction": float((semantic == 2).mean()),
        "fiber_fraction": float((semantic == 1).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic XCT corpus")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_corpus"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, nargs=3, default=list(PATCH_SIZE))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    output_dir = args.output / "patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(args.patch_size)
    registry: list[dict] = []
    for i in tqdm(range(args.n_samples), desc="synthetic corpus"):
        entry = _generate_sample(i, output_dir, patch_size)
        if entry is not None:
            registry.append(entry)

    registry_path = args.output / "corpus.json"
    with open(registry_path, "w") as f:
        json.dump(
            {
                "patch_size": patch_size,
                "voxel_spacing_um": VOXEL_SPACING,
                "n_samples": len(registry),
                "samples": registry,
            },
            f,
            indent=2,
        )
    print(f"Generated {len(registry)} samples in {args.output}")
    print(f"Registry written to {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
