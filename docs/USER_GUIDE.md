# User Guide: Running the RAFA Pipeline

This guide walks you through analyzing 3D X-ray computed tomography (XCT) volumes with `fiber-tracer`, a Python toolkit for regime-aware fiber analysis (RAFA) of fiber-reinforced composites. It is written for users who have a TIFF stack and want to go from raw data to fiber orientation and morphometry results.

> **Project status:** Beta. The public API and CLI are stabilizing. Treat results as experimental and validate them on your own data before drawing conclusions. See [`docs/validation_protocol.md`](validation_protocol.md) for what has been benchmarked and [`docs/parameter_guide.md`](parameter_guide.md) for detailed configuration options.

---

## Table of contents

- [The three analysis regimes](#the-three-analysis-regimes)
- [Input data formats](#input-data-formats)
- [First analysis: synthetic phantom](#first-analysis-synthetic-phantom)
- [Resolved-regime workflow](#resolved-regime-workflow)
- [Marginal-regime workflow](#marginal-regime-workflow)
- [Subvoxel-regime workflow](#subvoxel-regime-workflow)
- [Configuration walkthrough](#configuration-walkthrough)
- [Anisotropic voxel spacing](#anisotropic-voxel-spacing)
- [Batch processing](#batch-processing)
- [Visualization](#visualization)
- [Next steps](#next-steps)

---

## The three analysis regimes

`fiber-tracer` selects a pipeline from the physical ratio

```text
r = min(voxel_spacing_z, voxel_spacing_y, voxel_spacing_x) / fiber_diameter_um
```

The minimum voxel spacing is used so that anisotropic datasets are classified conservatively.

| Regime   | Threshold        | When to use it                                                                  |
|----------|------------------|---------------------------------------------------------------------------------|
| `resolved` | `r <= 0.3`       | Fiber diameter is much larger than a voxel. Individual fibers can be segmented. |
| `marginal` | `0.3 < r <= 3.0` | Fiber diameter is comparable to a voxel. Individual segmentation is unreliable. |
| `subvoxel` | `r > 3.0`        | Many fibers fit inside one voxel. Only population statistics are meaningful.    |
| `auto`     | —                | Select from `r` automatically using the thresholds above.                       |

To choose a regime manually:

1. Measure or look up the smallest voxel spacing of your scan (in µm).
2. Divide it by the expected fiber diameter.
3. Pick `resolved`, `marginal`, or `subvoxel` based on the table, or use `auto` and inspect `summary.json` to see which regime was selected.

Detailed algorithm discussion is in [`docs/methodology.md`](methodology.md).

---

## Input data formats

`fiber-tracer` reads:

| Format                      | How to specify it on the CLI            | Notes |
|-----------------------------|------------------------------------------|-------|
| Single multi-page TIFF stack | `--data path/to/stack.tif`              | Standard output from most XCT reconstruction software. |
| Directory of TIFF slices    | `--data path/to/slices/`                | Slices are read in sorted filename order. |
| HDF5 (helper script)        | Use `scripts/download_gfpa66.py` or load in Python and save as TIFF. | The GF-PA66 reference dataset is distributed as HDF5; helper scripts convert or validate it. |

Two pieces of metadata are mandatory for every run:

1. **`voxel_spacing_um`** in `(z, y, x)` order. This converts voxel measurements to physical units and drives regime selection.
2. **`fiber_diameter_um`**, the expected fiber diameter.

Optional preprocessing flags such as `--denoise-sigma` are supplied through a configuration file (see [Configuration walkthrough](#configuration-walkthrough)).

---

## First analysis: synthetic phantom

If you are new to `fiber-tracer`, start with the deterministic phantom from the README so you can compare your outputs against a known example.

### 1. Install the package

```bash
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[structure,skeleton,dev]"
```

Add interactive visualization later with:

```bash
pip install -e ".[viz]"
```

### 2. Generate the phantom

```bash
python - <<'PY'
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.validation.phantoms import generate_fiber_phantom

phantom = generate_fiber_phantom(
    shape=(64, 64, 64),
    n_fibers=3,
    fiber_diameter_um=4.0,
    voxel_spacing_um=(1.0, 1.0, 1.0),
    seed=42,
)
save_tiff_stack("phantom.tif", phantom.volume)
PY
```

### 3. Run the resolved pipeline

```bash
fiber-tracer \
  --data phantom.tif \
  --output output/ \
  --voxel-spacing 1.0 1.0 1.0 \
  --fiber-diameter 4.0 \
  --regime resolved
```

Here `r = 1.0 / 4.0 = 0.25`, which is below the `resolved` threshold of `0.3`.

### 4. Inspect the outputs

```text
output/
├── labels.tif
├── normalized_input.tif
├── report.csv
├── report.html
├── skeleton.tif
└── summary.json
```

```bash
python - <<'PY'
import json
with open("output/summary.json") as f:
    summary = json.load(f)
print("Regime:", summary["regime"])
print("Labels:", summary["n_labels"])
PY
```

You should see `regime: "resolved"` and three labels, matching the requested number of fibers.

---

## Resolved-regime workflow

Use `resolved` when individual fibers are clearly wider than several voxels (`r <= 0.3`). This is the only regime that produces per-fiber measurements.

### Pipeline summary

1. Intensity normalization and optional Gaussian denoising.
2. Foreground detection with 3D Otsu thresholding.
3. Binary morphological cleanup.
4. Labeling via connected components or marker-controlled watershed.
5. Per-label skeletonization.
6. Per-fiber PCA orientation.
7. Per-fiber equivalent spherical diameter.

### Expected outputs

| File                   | Description                                            |
|------------------------|--------------------------------------------------------|
| `labels.tif`           | Label image of segmented fibers.                       |
| `skeleton.tif`         | Per-fiber skeleton image.                              |
| `normalized_input.tif` | Intensity-normalized input volume.                     |
| `report.csv`           | One row per fiber with orientation and morphometry.    |
| `report.html`          | Human-readable HTML report.                            |
| `summary.json`         | JSON summary with regime, metrics, and configuration.  |

### Choosing a segmentation method

Set the method in your configuration file under `segmentation.method`:

| Method       | When to use it                                      | Caveat |
|--------------|-----------------------------------------------------|--------|
| `otsu`       | Fibers are already well separated.                  | Touching fibers will be merged into one label. |
| `watershed`  | Fibers touch and you want to split them.            | Can over-segment elongated or densely packed fibers. |

Example configuration snippet:

```yaml
segmentation:
  method: watershed
```

Run with:

```bash
fiber-tracer --config config.yaml --data sample.tif --output results/
```

### Limitations

- Fibers must be separable by thresholding and/or watershed.
- Touching or overlapping fibers can be over- or under-segmented.
- Results depend on a correct `fiber_diameter_um` for morphometry.

---

## Marginal-regime workflow

Use `marginal` when the fiber diameter is comparable to the voxel size (`0.3 < r <= 3.0`). Individual fiber segmentation is unreliable, so the pipeline estimates a local orientation field and summarizes it with the Advani–Tucker second-order orientation tensor `A2`.

### Pipeline summary

1. Intensity normalization and optional denoising.
2. Gradient structure tensor with inner scale `sigma_um` and outer integration scale `rho_um`.
3. Foreground masking with Otsu thresholding.
4. Local orientation field from the eigenvector of the smallest structure-tensor eigenvalue.
5. Windowed `A2` tensor field across the volume.
6. Global `A2` and fractional anisotropy reported in `summary.json`.

### Expected outputs

| File               | Description                                                            |
|--------------------|------------------------------------------------------------------------|
| `a2_map.npy`       | Windowed Advani–Tucker `A2` tensor field (`N_windows × 3 × 3`).        |
| `a2_centers.npy`   | Physical `(z, y, x)` coordinates of the window centers.                |
| `report.csv`       | Global metrics; not per-fiber rows.                                    |
| `report.html`      | Human-readable HTML report.                                            |
| `summary.json`     | JSON summary including global `A2`, fractional anisotropy, and caveats.|

### Choosing structure-tensor scales

Set these under `orientation` in your configuration:

| Parameter        | Default                        | Guidance |
|------------------|--------------------------------|----------|
| `sigma_um`       | `min(voxel_spacing)`           | Inner (derivative) scale. Start with the smallest voxel spacing. |
| `rho_um`         | `fiber_diameter_um / 2`        | Outer (integration) scale. Choose a value comparable to the fiber radius. |
| `window_size_um` | `fiber_diameter_um`            | Size of the cubic window for the `A2` map. Smaller windows give higher spatial resolution but noisier tensors; larger windows are smoother but blur local variation. |

Example configuration:

```yaml
voxel_spacing_um: [1.0, 1.0, 1.0]
fiber_diameter_um: 2.0
regime: marginal
orientation:
  sigma_um: 1.0
  rho_um: 1.0
  window_size_um: 2.0
```

### Limitations

- Results mix single-fiber and population-level information.
- Accuracy degrades as the fiber diameter approaches the voxel size.
- The window must fit inside the volume; choose `window_size_um` smaller than the smallest physical dimension.

---

## Subvoxel-regime workflow

Use `subvoxel` when many fibers fit inside a single voxel (`r > 3.0`). Only population-level orientation statistics are meaningful.

### Pipeline summary

1. Intensity normalization and optional denoising.
2. Gradient structure tensor with an integration scale enlarged to at least `3 * min(voxel_spacing)`.
3. Foreground masking with Otsu thresholding.
4. Global `A2` tensor and fractional anisotropy over all foreground voxels.
5. Orientation distribution histogram relative to the principal axis.

### Expected outputs

| File           | Description                                                            |
|----------------|------------------------------------------------------------------------|
| `report.csv`   | Global metrics only.                                                   |
| `report.html`  | Human-readable HTML report.                                            |
| `summary.json` | Global `A2`, fractional anisotropy, and orientation distribution data. |

The normalized volume is kept in memory but is **not** saved by default.

### Limitations

- Individual fiber measurements are not meaningful.
- Only population orientation statistics such as `A2`, fractional anisotropy, and orientation distributions are reported.
- Large integration scales blur spatial detail.

---

## Configuration walkthrough

Most analysis options are supplied through a YAML or JSON file. Command-line values take precedence over file values.

### Minimal configuration

```yaml
data_path: "input.tif"
output_dir: "output"
voxel_spacing_um: [1.0, 1.0, 1.0]
fiber_diameter_um: 4.0
regime: "auto"
```

Run it:

```bash
fiber-tracer --config config.yaml --data input.tif --output output/
```

### Dict-style vs. list-style `voxel_spacing_um`

`Config.from_dict` accepts two forms. The list form `[z, y, x]` is convenient for batch files and the CLI:

```yaml
# List style (z, y, x)
voxel_spacing_um: [1.0, 1.0, 1.0]
```

```yaml
# Dict style
voxel_spacing_um:
  z: 1.0
  y: 1.0
  x: 1.0
```

Both are equivalent. The list form is expanded to a `VoxelSpacing(z, y, x)` internally.

### Full example

```yaml
data_path: "input.tif"
output_dir: "output"
voxel_spacing_um: [1.0, 1.0, 1.0]
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

For a complete reference of every option, see [`docs/parameter_guide.md`](parameter_guide.md).

---

## Anisotropic voxel spacing

Anisotropic voxels are supported. Provide the actual physical spacing in `(z, y, x)` order:

```bash
fiber-tracer \
  --data sample_b.tif \
  --output results/sample_b/ \
  --voxel-spacing 2.0 1.0 1.0 \
  --fiber-diameter 8.0 \
  --regime auto
```

Because regime selection uses `min(voxel_spacing)`, this example is classified conservatively: `r = 1.0 / 8.0 = 0.125`, so `auto` selects `resolved`.

### When to resample

Orientation and morphometry accuracy degrade for large anisotropy ratios. Consider resampling to isotropic voxels when the ratio between the largest and smallest spacing exceeds approximately 2–3. You can do this before analysis with your favorite image-processing tool, or programmatically with `fiber_tracer.preprocess` helpers.

---

## Batch processing

The `batch` subcommand processes multiple volumes from a single YAML or JSON file. A `common` block supplies shared settings; each entry in `volumes` can override them.

Create `batch.yaml`:

```yaml
common:
  voxel_spacing_um: [1.0, 1.0, 1.0]
  fiber_diameter_um: 6.0
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
    fiber_diameter_um: 4.0
    regime: resolved
```

Run the batch:

```bash
fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv
```

The aggregate CSV contains one row per volume with `data_path`, `output_dir`, `regime`, `n_labels`, and `elapsed_s`.

---

## Visualization

Interactive visualization requires the `viz` extra:

```bash
pip install -e ".[viz]"
```

### View raw data and results in napari

```bash
fiber-tracer view --data phantom.tif --output output/
```

The viewer loads the raw volume, label image, skeleton, and orientation vector layers when available.

### Generate an interactive HTML report

```bash
fiber-tracer report-viz --summary output/summary.json --output report.html
```

Open `report.html` in any modern web browser. The report is self-contained and does not require a running server.

---

## Next steps

- **Tune parameters:** [`docs/parameter_guide.md`](parameter_guide.md) contains the full configuration reference, CLI flags, and practical guidance by regime.
- **Command-line details:** [`docs/CLI_REFERENCE.md`](CLI_REFERENCE.md) lists every `fiber-tracer` subcommand and option.
- **Troubleshoot problems:** [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) covers common errors and how to fix them.
- **Validate your setup:** [`docs/validation_protocol.md`](validation_protocol.md) explains the phantom benchmarks, the GF-PA66 public dataset, and what the acceptance thresholds mean.
- **Understand the algorithms:** [`docs/methodology.md`](methodology.md) describes the structure tensor, Advani–Tucker tensor, and regime selection in detail.
- **Extend the tool:** [`docs/architecture.md`](architecture.md) maps the package modules and explains how to add backends or regimes.

---

## License and citations

`fiber-tracer` is released under the MIT License. If you use it in your research, please cite the methods listed in [`CITATIONS.md`](../CITATIONS.md) and [`docs/methodology.md`](methodology.md).
