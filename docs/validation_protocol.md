# Validation Protocol

This document describes how `fiber-tracer` is validated. It is intended to be transparent about what has been tested, what the acceptance criteria are, and what remains the responsibility of the user.

## Synthetic phantom benchmark

The primary automated benchmark generates deterministic 3D phantoms of straight fibers with known orientations and runs the resolved-regime pipeline. The script is `scripts/benchmark_phantoms.py`.

### Phantom generation

Phantoms are produced by `fiber_tracer.validation.phantoms.generate_fiber_phantom`:

- Straight cylindrical fibers drawn at random positions and orientations.
- Controlled additive Gaussian noise (`noise_std=0.02`).
- Fixed random seeds so runs are deterministic.

The benchmark uses:

- shape `(96, 96, 96)`
- 5 fibers
- `fiber_diameter_um = 6.0`
- isotropic `voxel_spacing_um = (1.0, 1.0, 1.0)`
- `regime = "resolved"`

This gives `r = 1/6 ≈ 0.167`, well inside the resolved threshold (`r <= 0.3`).

### Alignment

Predicted label IDs are remapped to ground-truth IDs by maximum voxel overlap before computing per-label metrics.

### Metrics and acceptance thresholds

| Metric                        | Target                |
|-------------------------------|-----------------------|
| Mean Dice score               | `> 0.85`              |
| Mean angular error            | `< 5°`                |

These thresholds are synthetic-benchmark targets. They do not guarantee the same accuracy on real XCT data, where partial-volume effects, noise, and touching fibers make segmentation harder.

## Public dataset validation

The GF-PA66 3D XCT dataset is used as a public reference for benchmarking:

- **Citation**: Bertoldo, J. P. C. et al. (2021). A Modular U-Net for Automated Segmentation of X-Ray Tomography Images in Composite Materials. *Frontiers in Materials*, 8, 761229. DOI:10.3389/fmats.2021.761229
- **License**: CC BY-SA 4.0
- **DOI**: 10.5281/zenodo.4587827

The dataset is used for benchmarking only. It is not redistributed with this package. Users who create and redistribute derived works based on this dataset must comply with the CC BY-SA 4.0 share-alike terms.

An automatic downloader is provided in `scripts/download_gfpa66.py`. List available files:

```bash
python scripts/download_gfpa66.py --list
```

Download the GF-PA66 volume file (accepting the CC BY-SA 4.0 license):

```bash
python scripts/download_gfpa66.py --file pa66_volumes.h5 --output-dir data/ --accept-license
```

## Metric definitions

### Dice score

For binary masks `P` (predicted) and `T` (true):

```
Dice(P, T) = 2 |P ∩ T| / (|P| + |T|)
```

`mean_dice_score` averages this value over all foreground labels in the ground truth.

### Angular error

For two direction vectors `p` and `t`:

```
angle(p, t) = arccos(|p · t|) * 180 / π
```

The absolute dot product makes the error invariant to the sign of the direction (fibers have no inherent orientation).

### Orientation tensor error

For predicted and ground-truth `A2` tensors:

```
error = ||A2_pred - A2_true||_F
```

where `||·||_F` is the Frobenius norm.

### Fractional anisotropy

From the eigenvalues `λi` of `A2`:

```
FA = sqrt(1.5 * Σ(λi - λ̄)^2 / Σλi^2)
```

`FA` ranges from `0` (isotropic) to `1` (perfectly aligned).

## Reproducibility

- Phantoms are deterministic when a seed is supplied.
- The benchmark script fixes the random seed.
- Dependency versions are constrained in `pyproject.toml` (lower bounds and a scikit-image upper bound).
- For full reproducibility, install the package in a clean virtual environment with `pip install -e ".[dev]"` and run the benchmark.

## How to run

```bash
python scripts/benchmark_phantoms.py
```

The script prints a JSON report and asserts the acceptance thresholds. A passing run exits with code `0`.

## Caveats

- The phantom benchmark tests idealized straight fibers. Real composites may contain curvature, varying diameter, and contact points that reduce accuracy.
- Public dataset benchmarking requires downloading the dataset separately and respecting its license.
- Acceptance thresholds are project targets, not universal accuracy claims.
