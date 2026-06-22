# Architecture

This document describes the high-level architecture of `fiber-tracer`, a regime-aware fiber analysis (RAFA) toolkit for 3D X-ray CT of fiber-reinforced composites.

## Overview

RAFA selects analysis algorithms based on the physical relationship between the voxel size and the expected fiber diameter. The ratio

```
r = min(voxel_spacing_z, voxel_spacing_y, voxel_spacing_x) / fiber_diameter_um
```

drives the choice of one of three regimes:

| Regime   | Threshold        | Physical interpretation                         |
|----------|------------------|-------------------------------------------------|
| resolved | `r <= 0.3`       | fiber diameter is much larger than a voxel      |
| marginal | `0.3 < r <= 3.0` | fiber diameter is comparable to a voxel         |
| subvoxel | `r > 3.0`        | many fibers fit inside a single voxel           |

The core package remains small and uses only permissive dependencies. Optional heavy backends (PyTorch, GUDHI) are isolated in adapter modules and lazy-imported so they are never required for core functionality.

## Module map

| Package | Purpose |
|---------|---------|
| `fiber_tracer.config` | `Config` dataclass, nested option dataclasses (`VoxelSpacing`, `ProcessingConfig`, `SegmentationConfig`, `OrientationConfig`, `AnalysisConfig`), validation, and YAML/JSON loading. |
| `fiber_tracer.cli` | `argparse` CLI with subcommands `run`, `analyze`, `view`, `report-viz`, and `batch`, plus backward-compatible top-level flags. |
| `fiber_tracer.exceptions` | Shared exception hierarchy: `FiberTracerError`, `ConfigError`, `DataError`, `BackendNotAvailableError`, `ValidationError`. |
| `fiber_tracer.io` | TIFF stack loading and saving (single file or directory of slices), volume metadata helpers, and dtype conversion for TIFF writing. |
| `fiber_tracer.preprocess` | Intensity normalization, 3D Gaussian denoising with physical sigma, and optional resampling to isotropic voxels. |
| `fiber_tracer.segmentation` | Classical 3D segmentation: Otsu thresholding, connected-components labeling, and marker-controlled watershed on the distance transform. |
| `fiber_tracer.centerline` | Per-label skeletonization with scikit-image and an optional `skan`-based skeleton graph adapter. |
| `fiber_tracer.orientation` | Per-fiber PCA orientation, gradient structure-tensor orientation field, and Advani–Tucker second-order orientation tensor (`A2`) computation. |
| `fiber_tracer.analysis` | Fiber morphometry: equivalent spherical diameter, ordered path length, tortuosity, and per-fiber volume counts. |
| `fiber_tracer.regime` | `detect_regime` and regime validation utilities. |
| `fiber_tracer.pipeline` | `FiberAnalysisPipeline` orchestrator that dispatches to the regime-specific pipeline and writes all outputs. |
| `fiber_tracer.reporting` | JSON, CSV, and HTML report writers plus shared citation and caveat metadata. |
| `fiber_tracer.viz` | Optional visualization helpers: napari viewer integration and Plotly interactive reports. |
| `fiber_tracer.validation` | Synthetic phantom generation, benchmark scripts, alignment and metrics. |
| `fiber_tracer.backends` | Optional backend adapters for ML segmentation (`MLSegmentationBackend`) and topological data analysis (`gudhi`). |
| `fiber_tracer.batch` | YAML/JSON batch configuration loader and multi-volume processing. |
| `fiber_tracer.chunked` | zarr-based chunked helpers for out-of-core normalization, denoising, and generic per-chunk processing. |

## Data flow

```
raw TIFF stack
    ↓
I/O loader (fiber_tracer.io)
    ↓
Preprocessing (normalize, denoise, resample)
    ↓
Regime selector (resolved / marginal / subvoxel)
    ↓
Regime-specific pipeline
    ↓
Outputs: summary.json, report.csv, report.html, labels.tif, skeleton.tif, a2_map.npy
```

The pipeline always writes `summary.json`, `report.csv`, and `report.html`. Regime-specific files are produced only when relevant:

- `labels.tif` and `skeleton.tif` are produced in the resolved regime.
- `a2_map.npy` and `a2_centers.npy` are produced in the marginal regime.
- `normalized_input.tif` is written for all regimes.

## Regime-specific pipelines

### Resolved regime

Used when individual fibers are several voxels wide.

1. Optional denoising and intensity normalization.
2. Foreground detection with global 3D Otsu thresholding, or a U-Net backend when `segmentation.method = "unet"`.
3. Binary morphological opening to remove small spurious foreground voxels.
4. Labeling:
   - `segmentation.method = "otsu"`: connected-components labeling.
   - `segmentation.method = "watershed"`: marker-controlled watershed on the distance transform.
5. Per-label skeletonization to avoid bridging separate fibers.
6. Per-fiber PCA orientation on voxel coordinates.
7. Per-fiber equivalent spherical diameter from label volume.

### Marginal regime

Used when the fiber diameter is comparable to the voxel size.

1. Optional denoising and intensity normalization.
2. Gradient structure tensor with inner scale `sigma_um` and outer integration scale `rho_um`.
3. Foreground masking with Otsu thresholding.
4. Local orientation field from the eigenvector of the smallest structure-tensor eigenvalue.
5. Windowed `A2` tensor field across the volume, producing `a2_map.npy`.
6. Global `A2` and fractional anisotropy reported in `summary.json`.

### Subvoxel regime

Used when many fibers fit inside one voxel and only population statistics are meaningful.

1. Optional denoising and intensity normalization.
2. Gradient structure tensor with an integration scale enlarged to at least `3 * min(voxel_spacing)`.
3. Foreground masking with Otsu thresholding.
4. Global `A2` tensor and fractional anisotropy over all foreground voxels.
5. Orientation distribution histogram relative to the principal axis.

## Optional backends

The `backends/` package isolates heavy or license-sensitive dependencies:

- `MLSegmentationBackend` lazy-imports `torch`. It ships without a trained model; users must load a checkpoint or subclass it before calling `segment()`. Selecting `segmentation.method = "unet"` in the resolved pipeline routes the volume through this backend.
- `tda_gudhi` lazy-imports `gudhi` and provides `betti_numbers()` and `persistence_summary()`. Setting `analysis.compute_tda_descriptors = True` in the resolved regime adds these descriptors to `summary.json`.
- `centerline/graph.py` lazy-imports `skan` and raises `BackendNotAvailableError` if the skeleton extra is not installed.
- `orientation/structure_tensor.py` can use the optional `structure-tensor` package, but the pipeline uses a scipy-based fallback by default.

No optional backend is required for core functionality. The resolved-regime classical pipeline, marginal analysis, and subvoxel analysis all run with only the core dependencies.

## Chunked / out-of-core processing

`fiber_tracer.chunked` provides zarr-based helpers for volumes that do not fit in RAM:

- `load_zarr`, `save_zarr`, `tiff_to_zarr` — read and write chunked arrays.
- `process_chunks` — apply a function to overlapping chunks and write only the central region back.
- `normalize_intensity_chunked` — two-pass min–max normalization with one chunk in memory at a time.
- `gaussian_denoise_chunked` — Gaussian smoothing with overlap padding to avoid boundary artifacts.

The `parallel` extra installs `dask` for higher-level distributed workflows, but it is not required by these helpers. See `docs/parameter_guide.md` for an example.

## Extension points

### Adding a new regime

1. Add a regime identifier to `VALID_REGIMES` in `fiber_tracer/config.py` if needed.
2. Update `detect_regime` in `fiber_tracer/regime.py` with the threshold logic.
3. Add a `_run_<regime>` method in `fiber_tracer/pipeline.py`.
4. Add a caveat string in `fiber_tracer/reporting/citations.py`.
5. Add tests in `tests/test_regime.py` and `tests/test_pipeline_<regime>.py`.

### Adding a new backend

1. Create a module under `src/fiber_tracer/backends/`.
2. Lazy-import the third-party library inside the constructor or function.
3. Raise `BackendNotAvailableError` with an installation hint if the dependency is missing.
4. Add tests in `tests/test_<backend>_backend.py` that verify both availability and the error path when the dependency is missing.
5. Wire the config flag or `segmentation.method` value in `fiber_tracer/pipeline.py`.

### Adding a new visualization helper

1. Add a helper function to `src/fiber_tracer/viz/napari_viewer.py` or `src/fiber_tracer/viz/plotly_plots.py`.
2. Export it from `src/fiber_tracer/viz/__init__.py`.
3. If it needs CLI access, add a subcommand or flag in `fiber_tracer/cli.py`.
4. Add tests in `tests/test_napari_viewer.py` or `tests/test_plotly_plots.py`.
