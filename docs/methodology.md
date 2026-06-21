# Methodology

`fiber-tracer` is an image-analysis toolkit for 3D X-ray CT of fiber-reinforced composites. It selects algorithms based on the physical relationship between the voxel size and the expected fiber diameter. The implementation uses classical image processing and tensor methods; it does not solve differential equations and does not perform streamline integration.

## Regime-aware design

The analysis regime is chosen from the ratio

```
r = min(voxel_spacing_z, voxel_spacing_y, voxel_spacing_x) / fiber_diameter_um
```

using the thresholds in `fiber_tracer.regime.detect_regime`:

| Regime     | Threshold            | Physical interpretation                  |
|------------|----------------------|------------------------------------------|
| resolved   | `r <= 0.3`           | fiber diameter is much larger than a voxel |
| marginal   | `0.3 < r <= 3.0`     | fiber diameter is comparable to a voxel   |
| subvoxel   | `r > 3.0`            | fiber diameter is much smaller than a voxel |

The minimum spacing is used so that anisotropic datasets are classified conservatively. The regime can also be set explicitly with the `regime` configuration option.

## Resolved regime pipeline

When fibers are resolved, the tool extracts individual fibers with a conventional image-processing pipeline:

1. **Intensity normalization** — min–max scaling to `[0, 1]` (`normalize_intensity`).
2. **Optional denoising** — 3D Gaussian smoothing with a physical sigma (`processing.denoise_sigma`).
3. **Foreground detection** — global 3D Otsu thresholding (`skimage.filters.threshold_otsu`).
4. **Morphological cleanup** — binary opening to remove small spurious foreground voxels.
5. **Labeling**
   - `segmentation.method = "otsu"` (default): connected-components labeling.
   - `segmentation.method = "watershed"`: marker-controlled watershed on the distance transform with `peak_local_max` markers and `min_distance=3` voxels.
6. **Skeletonization** — each labeled fiber is skeletonized independently with `skimage.morphology.skeletonize_3d` to avoid bridging separate fibers.
7. **Per-fiber orientation** — principal-component analysis (PCA) on the voxel coordinates of each label (`fiber_tracer.orientation.pca`).
8. **Morphometry** — equivalent spherical diameter from the label volume; the skeleton is saved for visualization; path-length and tortuosity measures are implemented in `analysis.morphometry` but are not yet integrated into the default pipeline output.

## Marginal and subvoxel pipeline

When individual fibers are not reliably separated, the tool estimates a local orientation field and summarizes it with the Advani–Tucker second-order orientation tensor.

### Gradient structure tensor

A 3D gradient structure tensor is computed at each voxel. The fallback implementation in `fiber_tracer.orientation.structure_tensor.compute_local_orientation_field`:

1. Computes Gaussian derivatives along each axis.
2. Divides derivatives by the physical voxel spacing so the tensor is consistent under anisotropic sampling.
3. Forms the outer product of the gradient vector with itself.
4. Smooths each tensor component with a Gaussian of outer scale `rho`.

The optional `structure-tensor` package backend is exposed in `orientation/structure_tensor.py` but the pipeline currently uses the scipy-based `compute_local_orientation_field` fallback.

Scales default to:

- inner (derivative) scale `sigma_um = min(voxel_spacing)` if not set
- outer (integration) scale `rho_um = fiber_diameter_um / 2` if not set

For the subvoxel regime the integration scale is enlarged to at least `3 * min(voxel_spacing)` because the gradient field must be averaged over many fibers.

The fiber direction is taken as the eigenvector of the **smallest** eigenvalue of the structure tensor.

### Advani–Tucker orientation tensor

For a set of unit fiber directions `p_i`, the second-order orientation tensor is

```
A2 = <p p^T> = (1 / N) Σ_i p_i p_i^T
```

In the marginal regime a spatially windowed `A2` map is produced (`fiber_tracer.orientation.tensor.windowed_orientation_tensor_field`). In the subvoxel regime a single global `A2` is computed from all foreground directions.

### Fractional anisotropy

From the eigenvalues `λ1, λ2, λ3` of `A2`:

```
λ̄ = (λ1 + λ2 + λ3) / 3
FA = sqrt(1.5 * Σ_i (λi - λ̄)^2 / Σ_i λi^2)
```

`FA` is `0` for an isotropic orientation distribution and approaches `1` for perfectly aligned fibers.

## Why Runge–Kutta and Poincaré–Hopf are not used

`fiber-tracer` is an image-analysis tool, not a differential-equation solver. Fiber extraction in the resolved regime relies on thresholding, morphological operations, and skeletonization. The marginal and subvoxel regimes use gradient structure tensors and orientation tensors. Runge–Kutta integration and Poincaré–Hopf index tracking are therefore not part of the segmentation pipeline.

## Limitations and caveats

- **Resolved regime** assumes fibers are separable by thresholding and/or watershed. Touching or overlapping fibers can be over- or under-segmented.
- **Watershed** separation depends on the distance transform and can fail for densely packed or irregularly shaped fibers.
- **Marginal regime** mixes single-fiber and population-level information; results degrade as the fiber diameter approaches the voxel size.
- **Subvoxel regime** produces population statistics only. Individual fiber measurements are not meaningful.
- **Anisotropic voxel spacing** reduces orientation and morphometry accuracy unless derivatives are scaled by physical spacing. Resampling to isotropic voxels is recommended when the spacing ratios are large.
- **Phantom validation** is performed on ideal straight fibers with controlled noise. Performance on real composites will vary with contrast, noise, partial-volume effects, and fiber contacts.

## Citations

- Bigün, J., & Granlund, G. H. (1987). Optimal orientation detection of linear symmetry. *Proceedings of the First International Conference on Computer Vision (ICCV)*.
- Jeppesen, N., Mikkelsen, L. P., Dahl, A. B., Christensen, A. N., & Dahl, V. A. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. *Composites Part A*, 149, 106541. DOI:10.1016/j.compositesa.2021.106541
- Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. *Journal of Rheology*, 31(8), 751–784. DOI:10.1122/1.549945
- van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: Image processing in Python. *PeerJ*, 2, e453.

See also `CITATIONS.md` and `THIRD_PARTY_LICENSES.md` for software and dataset attributions.
