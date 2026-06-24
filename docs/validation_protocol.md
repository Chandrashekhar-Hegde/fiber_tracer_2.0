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

Run the validation helper:

```bash
python scripts/validate_gfpa66.py --data data/pa66_volumes.h5 --output results/gfpa66/
```

`validate_gfpa66.py` converts the HDF5 volume to a temporary TIFF stack and runs the RAFA pipeline. The HDF5 dataset is auto-detected when the file contains a single dataset or a common name (`data`, `image`, `volume`, `XCT`). Use `--dataset <name>` only if auto-detection fails.

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

## U-Net model validation

A separate validation stream is used for the optional 3D U-Net segmentation backend (`segmentation.method: "unet"`).

### Training validation

During training we report three validation metrics:

- **Soft Dice** — Dice computed from sigmoid probabilities; useful for monitoring but dominated by background-only patches.
- **Hard Dice** — Dice computed after thresholding probabilities at 0.5; more interpretable but still affected by the high background fraction.
- **Foreground hard Dice** — Hard Dice averaged only over validation patches containing ≥0.1% foreground voxels. This is the primary metric because it measures segmentation quality where fibers actually exist.

The production checkpoint (`fiber_unet_v2_full.pt`) achieved a foreground hard Dice of **0.948** on its training-validation split.

### Held-out GF-PA66 validation

GF-PA66 (`pa66_volumes.h5`) was held out from training and used as a blind-like test. Sliding-window inference (64³ patches, stride 32) was run on the central 128 axial slices and compared against the `ground_truth` labels (classes > 0 treated as fiber):

| Metric | Value |
|--------|-------|
| Target fiber voxels | 30,117,323 |
| Predicted fiber voxels | 24,565,325 |
| Dice | 0.895 |
| IoU | 0.811 |
| Pixel accuracy | 0.967 |

### U-Net caveats

- The GF-PA66 result is a single-reference benchmark. It does not prove the model generalizes to all glass/carbon fiber systems, resolutions, or contrast conditions.
- Real training volumes other than GF-PA66 use Otsu pseudo-labels, so the model learns to approximate Otsu behavior on those volumes.
- Reported accuracy is high mainly because background voxels dominate; Dice and IoU are the meaningful segmentation metrics.
- The model was trained and validated on an Apple M5 Pro with MPS + CPU fallback; numerical results should be identical on CUDA/CPU for the same checkpoint, but throughput differs.

See [`docs/MODEL_CARD.md`](MODEL_CARD.md) for a full model card, including intended use, known failure modes, and retraining instructions.

## Caveats

- The phantom benchmark tests idealized straight fibers. Real composites may contain curvature, varying diameter, and contact points that reduce accuracy.
- Public dataset benchmarking requires downloading the dataset separately and respecting its license.
- Acceptance thresholds are project targets, not universal accuracy claims.
- Machine-learning results depend on the similarity between the training distribution and the target data. Always validate the U-Net on your own volumes before reporting conclusions.
