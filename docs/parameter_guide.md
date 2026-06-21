# Parameter Guide

This guide explains the main configuration options for `fiber-tracer`. All physical lengths are in micrometres (`µm`) unless otherwise noted.

## Global parameters

### `voxel_spacing_um`

Physical size of one voxel in `(z, y, x)` order. This is required for every run because it converts voxel measurements to physical units and drives regime selection.

- Must be positive for all axes.
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

Boolean. If `True` (default), the input volume is min–max normalized to `[0, 1]` before segmentation. Recommended unless the input is already normalized.

## Segmentation parameters (`segmentation`)

### `method`

- `"otsu"` (default): threshold with Otsu, then label connected components. Best when fibers are already well separated.
- `"watershed"`: marker-controlled watershed on the distance transform. Useful for touching fibers, but can over-segment elongated objects.

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

If `True`, compute per-fiber geometric descriptors such as equivalent diameter. Used mainly in the resolved regime.

### `compute_orientation_tensor`

If `True`, compute the Advani–Tucker `A2` tensor and fractional anisotropy where applicable (marginal/subvoxel regimes).

### `compute_tda_descriptors`

If `True`, compute optional topological data analysis descriptors. Requires the `tda` extras (`gudhi`, `ripser`).

## Practical guidance by regime

| Regime    | `regime` value | Typical `sigma_um` | Typical `rho_um` | Typical `window_size_um` | Notes |
|-----------|----------------|--------------------|------------------|--------------------------|-------|
| Resolved  | `"resolved"`   | —                  | —                | —                        | Use `"otsu"` for separated fibers; `"watershed"` for touching fibers. Ensure `fiber_diameter_um` is correct. |
| Marginal  | `"marginal"`   | `min(spacing)`     | `fiber_diameter / 2` | `fiber_diameter`     | Window size trades resolution versus noise. |
| Subvoxel  | `"subvoxel"`   | `min(spacing)`     | auto-enlarged    | —                        | Only population-level orientation statistics are reliable. |

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
fiber-tracer --config config.yaml
```
