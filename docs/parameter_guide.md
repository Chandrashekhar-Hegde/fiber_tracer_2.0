# Parameter Guide

This guide explains the main configuration options for `fiber-tracer`. All physical lengths are in micrometres (`µm`) unless otherwise noted.

## CLI subcommands

`fiber-tracer` provides the following subcommands:

- `run` (default) — Run the RAFA pipeline on a single volume. Equivalent to calling with no subcommand for backward compatibility.
- `analyze` — Alias for `run`.
- `view` — Launch a napari viewer with the raw volume and pipeline outputs (`labels.tif`, `skeleton.tif`, `summary.json`). Requires the `viz` extra.
- `report-viz` — Generate an interactive HTML report from a `summary.json` file using Plotly. Requires the `viz` extra.
- `batch` — Process multiple volumes described by a YAML or JSON batch config.

Top-level flags such as `--data`, `--output`, `--config`, `--voxel-spacing`, `--fiber-diameter`, and `--regime` are still accepted for backward compatibility.

## `viz` extra

Interactive visualizations require the `viz` optional dependency group:

```bash
pip install -e ".[viz]"
```

This installs napari (for the `view` subcommand) and Plotly (for `report-viz`). The `viz` extra is not required for batch processing or core analysis.

## Global parameters

### `voxel_spacing_um`

Physical size of one voxel in `(z, y, x)` order. This is required for every run because it converts voxel measurements to physical units and drives regime selection.

- Must be positive for all axes.
- May be supplied as a dict `{z: ..., y: ..., x: ...}` or a 3-element list/tuple `[z, y, x]`. Both forms are equivalent after loading.
- Anisotropic spacing is supported, but accuracy degrades for large anisotropy ratios. Consider resampling to isotropic voxels when ratios exceed ~2–3.

### `fiber_diameter_um`

Expected fiber diameter. Used for:

- Regime selection (`r = min(spacing) / fiber_diameter_um`).
- Default integration scale for the structure tensor (`rho_um = fiber_diameter_um / 2`).
- Default window size for the marginal-regime `A2` map.

### `regime`

Analysis regime. Allowed values: `"auto"`, `"resolved"`, `"marginal"`, `"subvoxel"`.

- `"auto"` selects the regime from the voxel/fiber ratio using `detect_regime`.
- Use `"resolved"` only when fibers are clearly wider than several voxels.
- Use `"marginal"` when the fiber diameter is comparable to the voxel size.
- Use `"subvoxel"` when many fibers fit inside a single voxel and only population statistics are meaningful.

## Processing parameters (`processing`)

### `denoise_sigma`

Physical standard deviation of a 3D Gaussian smoothing kernel applied before analysis. Set to `None` to disable.

- Typical starting value: `0.5–1.0` µm for resolved fibers.
- Use smaller values or disable for marginal/subvoxel work to avoid blurring fine orientation signals.

### `normalize`

Boolean. Defaults to `True`. If `True`, the input volume is min–max normalized to `[0, 1]` before segmentation; if `False`, the raw volume is cast to `np.float32` and divided by its maximum so the pipeline still receives a float array in a known range. The pipeline respects this flag.

## Chunked / out-of-core processing

For volumes that do not fit in RAM, `fiber_tracer.chunked` provides helpers built on `zarr`. The `parallel` extra installs `dask` for higher-level distributed workflows, but it is not required by these helpers. The module includes:

- `load_zarr` / `save_zarr` / `tiff_to_zarr` – read and write chunked `zarr` arrays from TIFF stacks.
- `process_chunks` – apply a function to overlapping chunks and write only the central region back, avoiding boundary artifacts.
- `normalize_intensity_chunked` – two-pass min–max normalization that keeps only one chunk in memory at a time.
- `gaussian_denoise_chunked` – Gaussian smoothing with overlap padding for seamless chunk-wise denoising.

These helpers are intended for programmatic use; a future CLI flag `--chunk-size` will expose chunked processing directly from the command line. For now, convert a TIFF stack to zarr and call the chunked normalization/denoising functions before passing the result to the pipeline:

```python
from fiber_tracer.chunked import tiff_to_zarr, normalize_intensity_chunked
import zarr

input_zarr = tiff_to_zarr("large_stack.tif", "input.zarr", chunks=(64, 64, 64))
output_zarr = zarr.open_array("normalized.zarr", mode="w", shape=input_zarr.shape,
                               chunks=(64, 64, 64), dtype="float32")
normalize_intensity_chunked(input_zarr, output_zarr, chunk_shape=(64, 64, 64))
```

See `docs/architecture.md` and `src/fiber_tracer/chunked.py` for the full API and design notes.

## Segmentation parameters (`segmentation`)

### `method`

- `"otsu"` (default): threshold with Otsu, then label connected components. Best when fibers are already well separated.
- `"watershed"`: marker-controlled watershed on the distance transform. Useful for touching fibers, but can over-segment elongated objects.
- `"unet"`: lightweight custom 3D U-Net backend. Requires the `ml` extra and a trained checkpoint via `model_path`.

### `model_path`

Path to a PyTorch checkpoint produced by `scripts/train_unet_phantoms.py`. Required when `method` is `"unet"`; ignored otherwise.

### `min_fiber_diameter_um` / `max_fiber_diameter_um`

Bounds used by advanced segmentation methods if implemented. The classical pipeline currently relies on morphological cleanup and connected-components/watershed labeling.

### `watershed_seed_sigma_um`

Optional smoothing applied to the distance transform before peak detection. Not used by the default `min_distance=3` peak-finding implementation.

## Orientation parameters (`orientation`)

### `sigma_um`

Inner scale of the gradient structure tensor (derivative Gaussian sigma). Defaults to the minimum voxel spacing.

- Resolved regime: not used for orientation (PCA is used instead).
- Marginal/subvoxel: start with `sigma_um ≈ min(voxel_spacing)`.

### `rho_um`

Outer scale of the structure tensor (integration Gaussian sigma). Defaults to `fiber_diameter_um / 2`.

- Marginal: choose `rho_um` comparable to the fiber radius.
- Subvoxel: the pipeline automatically enlarges `rho_um` to at least `3 * min(voxel_spacing)` to average over many fibers.

### `window_size_um`

Size of the cubic window used to compute the marginal-regime `A2` map. Defaults to `fiber_diameter_um`. Converted to an odd number of voxels internally.

- Smaller windows give higher spatial resolution but noisier tensors.
- Larger windows give smoother, more reliable tensors but blur local variations.
- The window must fit inside the volume; choose a size smaller than the smallest spatial dimension in physical units.

## Analysis parameters (`analysis`)

### `compute_morphometry`

Boolean. Defaults to `True`. If `True`, the resolved-regime pipeline computes per-fiber geometric descriptors such as equivalent diameter. If `False`, the label image and skeleton are still produced, but per-fiber diameter values are omitted.

### `compute_orientation_tensor`

Boolean. Defaults to `True`. If `True`, the resolved-regime pipeline computes per-fiber PCA orientation, and the marginal/subvoxel pipelines compute the Advani–Tucker `A2` tensor, fractional anisotropy, and orientation distributions. If `False`, these analyses are skipped and a note is recorded in the summary.

### `compute_tda_descriptors`

Boolean. Defaults to `False`. If `True`, the resolved-regime pipeline computes Betti numbers and a persistence summary on the cleaned binary mask using the optional `gudhi` backend and includes them in `summary.json`. Requires the `tda` extra.

## Practical guidance by regime

| Regime    | `regime` value | Typical `sigma_um` | Typical `rho_um` | Typical `window_size_um` | Notes |
|-----------|----------------|--------------------|------------------|--------------------------|-------|
| Resolved  | `"resolved"`   | —                  | —                | —                        | Use `"otsu"` for separated fibers; `"watershed"` for touching fibers; `"unet"` with a trained checkpoint for learned segmentation. Ensure `fiber_diameter_um` is correct. |
| Marginal  | `"marginal"`   | `min(spacing)`     | `fiber_diameter / 2` | `fiber_diameter`     | Window size trades resolution versus noise. |
| Subvoxel  | `"subvoxel"`   | `min(spacing)`     | auto-enlarged    | —                        | Only population-level orientation statistics are reliable. |

## Batch configuration

The `batch` subcommand processes multiple volumes from a single YAML or JSON config. The config has a `common` block shared by all entries and a `volumes` list with per-volume overrides.

```yaml
common:
  voxel_spacing_um: [1.0, 1.0, 1.0]
  fiber_diameter_um: 4.0
  regime: auto
  processing:
    denoise_sigma: 0.8
    normalize: true
  segmentation:
    method: otsu
  analysis:
    compute_morphometry: true
    compute_orientation_tensor: true

volumes:
  - data_path: sample_a.tif
    output_dir: results/sample_a
  - data_path: sample_b.tif
    output_dir: results/sample_b
    regime: resolved
    fiber_diameter_um: 6.0
```

Run the batch:

```bash
fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv
```

The aggregate CSV contains one row per volume with `data_path`, `output_dir`, `regime`, `n_labels`, and `elapsed_s`.

## Example configuration

```yaml
data_path: "input.tif"
output_dir: "output"
voxel_spacing_um:
  z: 1.0
  y: 1.0
  x: 1.0
fiber_diameter_um: 4.0
regime: "auto"
processing:
  denoise_sigma: 0.8
  normalize: true
segmentation:
  method: "otsu"
orientation:
  sigma_um: 1.0
  rho_um: 2.0
  window_size_um: 4.0
analysis:
  compute_morphometry: true
  compute_orientation_tensor: true
  compute_tda_descriptors: false
```

Save this as `config.yaml` and run:

```bash
fiber-tracer --config config.yaml --data path/to/stack.tif --output results/
```

> **Note:** `--data` and `--output` may be provided on the command line or in the configuration file. If both are provided, the command-line value overrides the config file value.
>
> A ready-to-use single-volume example is in [`config_example.yaml`](../config_example.yaml) and a batch example is in [`batch_example.yaml`](../batch_example.yaml).
