# `fiber-tracer` CLI Reference

Complete command-line reference for **Fiber Tracer (RAFA) v3.2.0**, generated from the actual CLI help text and source code.

## Synopsis

```text
fiber-tracer [-h] [--version] [--log-level LOG_LEVEL]
             [--data DATA] [--output OUTPUT] [--config CONFIG]
             [--voxel-spacing Z Y X] [--fiber-diameter FIBER_DIAMETER]
             [--regime {auto,resolved,marginal,subvoxel}]
             [--segmentation-method {otsu,watershed,unet}]
             [--model-path MODEL_PATH] [--batch-size BATCH_SIZE]
             {run,analyze,view,report-viz,batch,model,experiment,train} ...
```

Top-level flags are retained for backward compatibility. Calling `fiber-tracer --data ... --output ...` without a subcommand is equivalent to `fiber-tracer run --data ... --output ...`.

## Installation reminder

The core package is installed with:

```bash
pip install -e .
```

Interactive visualization requires the `viz` extra:

```bash
pip install -e ".[viz]"
```

Other useful extras: `structure`, `skeleton`, `ml`, `unet`, `tda`, `parallel`, `dev`, `all`. See [`pyproject.toml`](../pyproject.toml) and [`README.md`](../README.md) for details.

## Subcommands

| Subcommand | Alias | Purpose |
|------------|-------|---------|
| `run` | — | Run the RAFA pipeline on a single volume. |
| `analyze` | `run` | Alias for `run`. |
| `view` | — | Visualize results in napari (requires `viz`). |
| `report-viz` | — | Generate an interactive Plotly HTML report from `summary.json` (requires `viz`). |
| `batch` | — | Process multiple volumes from a YAML/JSON batch config. |
| `model` | — | Manage registered segmentation models. |
| `experiment` | — | Manage and compare training experiments. |
| `train` | — | Train a 3D U-Net from a dataset directory. |

---

## `fiber-tracer run` / `fiber-tracer analyze`

Run the Regime-Aware Fiber Analysis (RAFA) pipeline on one 3D volume.

### Flags

| Flag | Required | Accepted values | Default | Description |
|------|----------|-----------------|---------|-------------|
| `--data` | Yes* | file or directory path | from config, if provided | Path to a multi-page TIFF stack or directory of TIFF slices. |
| `--output` | Yes* | directory path | from config, if provided | Output directory for `summary.json`, `report.csv`, `report.html`, and regime-specific images. |
| `--config` | No | YAML/JSON file path | — | Config file whose values are used for any flag not supplied on the command line. |
| `--voxel-spacing` | No | three positive floats (`Z Y X`) | `[1.0, 1.0, 1.0]` | Physical voxel spacing in micrometres. |
| `--fiber-diameter` | No | positive float | `10.0` | Expected fiber diameter in micrometres. |
| `--regime` | No | `auto`, `resolved`, `marginal`, `subvoxel` | `auto` | Analysis regime. `auto` selects from the voxel/fiber ratio. |
| `--segmentation-method` | No | `otsu`, `watershed`, `unet` | `otsu` | Segmentation backend. Use `unet` for the 3D U-Net model. |
| `--model-path` | No* | file path or registered model ID | `models/fiber_unet_v2_full.pt` | Path to a PyTorch checkpoint for `unet`, or a model ID from the registry. |
| `--batch-size` | No | positive integer | `1` | U-Net inference batch size. Increase for higher throughput if memory allows. |

\* `--data` and `--output` are not individually marked required by the parser, but the run fails if either is not provided by CLI or config.  
\*\* `--model-path` is required only when `--segmentation-method unet` is used.

### Examples

```bash
# Basic resolved-regime analysis
fiber-tracer run \
  --data sample_a.tif \
  --output results/sample_a/ \
  --voxel-spacing 1.0 1.0 1.0 \
  --fiber-diameter 6.0 \
  --regime resolved

# Backward-compatible top-level invocation
fiber-tracer \
  --data sample_b.tif \
  --output results/sample_b/ \
  --voxel-spacing 2.0 1.0 1.0 \
  --fiber-diameter 8.0 \
  --regime auto

# Use the pre-trained 3D U-Net backend
fiber-tracer run \
  --data sample_c.tif \
  --output results/sample_c_unet/ \
  --voxel-spacing 1.0 1.0 1.0 \
  --fiber-diameter 6.0 \
  --segmentation-method unet \
  --model-path models/fiber_unet_v2_full.pt

# Use a config file and override the input data and output directory
fiber-tracer run \
  --config config.yaml \
  --data sample_d.tif \
  --output results/sample_d/
```

---

## `fiber-tracer view`

Launch a napari viewer showing the raw volume and pipeline outputs (`labels.tif`, `skeleton.tif`, etc.). Requires the `viz` extra.

### Flags

| Flag | Required | Accepted values | Description |
|------|----------|-----------------|-------------|
| `--data` | Yes | file or directory path | Path to the original TIFF stack or directory. |
| `--output` | Yes | directory path | Output directory produced by `run`/`analyze`. |

### Example

```bash
fiber-tracer view --data sample_a.tif --output results/sample_a/
```

---

## `fiber-tracer report-viz`

Generate a self-contained interactive HTML report from a `summary.json` file using Plotly. Requires the `viz` extra.

### Flags

| Flag | Required | Accepted values | Description |
|------|----------|-----------------|-------------|
| `--summary` | Yes | `summary.json` path | JSON summary written by the pipeline. |
| `--output` | Yes | HTML file path | Destination HTML report. |

### Example

```bash
fiber-tracer report-viz \
  --summary results/sample_a/summary.json \
  --output report.html
```

---

## `fiber-tracer batch`

Process multiple volumes described by a single YAML or JSON batch config. Each volume can share settings from a `common` block and override them individually.

### Flags

| Flag | Required | Accepted values | Default | Description |
|------|----------|-----------------|---------|-------------|
| `--config` | Yes | YAML/JSON file path | — | Batch configuration file. |
| `--aggregate-csv` | No | CSV file path | `batch_summary.csv` | Aggregate output path with one row per volume. |

### Example

```bash
fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv
```

---

## YAML/JSON configuration files

Any CLI flag has a corresponding config-file key. The full set of keys follows the dataclass structure in `src/fiber_tracer/config.py`:

| Config key | CLI equivalent | Type / example |
|------------|----------------|----------------|
| `data_path` | `--data` | string |
| `output_dir` | `--output` | string |
| `voxel_spacing_um` | `--voxel-spacing` | dict `{z, y, x}` or list `[z, y, x]` of positive floats |
| `fiber_diameter_um` | `--fiber-diameter` | positive float |
| `regime` | `--regime` | `"auto"`, `"resolved"`, `"marginal"`, `"subvoxel"` |
| `processing.denoise_sigma` | — | float or `null` |
| `processing.normalize` | — | boolean |
| `processing.anisotropic_spacing` | — | dict `{z, y, x}` or `null` |
| `segmentation.method` | — | `"otsu"`, `"watershed"`, `"unet"` |
| `segmentation.model_path` | `--model-path` | string (path to `.pt` checkpoint) |
| `segmentation.batch_size` | `--batch-size` | positive integer |
| `segmentation.min_fiber_diameter_um` | — | float |
| `segmentation.max_fiber_diameter_um` | — | float |
| `segmentation.watershed_seed_sigma_um` | — | float or `null` |
| `orientation.method` | — | `"structure_tensor"`, `"pca"` |
| `orientation.sigma_um` | — | float or `null` |
| `orientation.rho_um` | — | float or `null` |
| `orientation.window_size_um` | — | float or `null` |
| `analysis.compute_morphometry` | — | boolean |
| `analysis.compute_orientation_tensor` | — | boolean |
| `analysis.compute_tda_descriptors` | — | boolean |

### `voxel_spacing_um` format

`voxel_spacing_um` is always expressed in `(z, y, x)` order and may be written as either a dict or a 3-element sequence:

```yaml
voxel_spacing_um:
  z: 2.0
  y: 1.0
  x: 1.0

# equivalent
voxel_spacing_um: [2.0, 1.0, 1.0]
```

### Command-line precedence

Command-line flags override values read from a config file. In particular:

* `--data` overrides `data_path`
* `--output` overrides `output_dir`
* `--voxel-spacing` overrides `voxel_spacing_um`
* `--fiber-diameter` overrides `fiber_diameter_um`
* `--regime` overrides `regime`

### Single-volume config example

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

Run it with:

```bash
fiber-tracer --config config.yaml --data path/to/stack.tif --output results/
```

### Batch config example

A batch config has a `common` block shared by every entry and a `volumes` list of per-volume overrides. Each entry must provide at least `data_path` and `output_dir` (either in `common` or in the entry).

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

The aggregate CSV contains one row per volume with `data_path`, `output_dir`, `regime`, `n_labels`, and `elapsed_s`.

---

## Exit codes and logging

* `--log-level` sets the Python logging level. Default: `INFO`. Accepted values include `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` (case-insensitive).
* Exit code `0` indicates success.
* Missing required inputs or runtime errors raise exceptions and produce a non-zero exit code.

Example:

```bash
fiber-tracer --log-level DEBUG run --data sample.tif --output results/
```

---

## `fiber-tracer model`

Manage the local segmentation model registry. Registry data is stored in `~/.config/fiber-tracer/models.json`.

### `fiber-tracer model list`

List registered models.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--json` | No | `false` | Output as JSON. |

```bash
fiber-tracer model list
fiber-tracer model list --json
```

### `fiber-tracer model add`

Add a checkpoint to the registry.

| Flag | Required | Description |
|------|----------|-------------|
| `--model-id` | Yes | Short unique identifier. |
| `--name` | Yes | Human-readable name. |
| `--path` | Yes | Path to the `.pt` checkpoint. |
| `--architecture` | No | Architecture, e.g. `UNet3D`. |
| `--version` | No | Version string. |
| `--description` | No | Free-text description. |

```bash
fiber-tracer model add \
  --model-id fiber_unet_v2_full \
  --name "Production 3D U-Net" \
  --path models/fiber_unet_v2_full.pt \
  --architecture UNet3D \
  --version 3.2.0 \
  --description "Trained on 2,152 mixed synthetic + XCT patches"
```

### `fiber-tracer model set-default`

Set the default model used by the TUI and by `run` when `--model-path` is omitted.

```bash
fiber-tracer model set-default fiber_unet_v2_full
```

### `fiber-tracer model remove`

Remove a registry entry. The checkpoint file on disk is not deleted.

```bash
fiber-tracer model remove fiber_unet_v2_full
```

---

## `fiber-tracer experiment`

Manage training experiment records. Experiment data is stored in `~/.config/fiber-tracer/experiments.jsonl`.

### `fiber-tracer experiment list`

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--json` | No | `false` | Output as JSON. |

```bash
fiber-tracer experiment list
fiber-tracer experiment list --json
```

### `fiber-tracer experiment show`

Show details for one experiment.

```bash
fiber-tracer experiment show exp-20260624-abc123
```

### `fiber-tracer experiment compare`

Compare two or more experiments by a metric.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--metric` | No | `val_loss` | Metric to compare. |

```bash
fiber-tracer experiment compare exp-001 exp-002 --metric val_dice
```

---

## `fiber-tracer train`

Train a 3D U-Net from a dataset directory containing `images/` and `masks/` subfolders. The run is recorded as an experiment and, when `--model-id` is supplied, the resulting checkpoint is added to the model registry.

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dataset-dir` | Yes | — | Training dataset directory. |
| `--output-dir` | Yes | — | Output directory for checkpoints and logs. |
| `--model-id` | No | generated | Registry ID for the trained model. |
| `--name` | No | generated | Experiment name. |
| `--epochs` | No | `20` | Training epochs. |
| `--batch-size` | No | `4` | Batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--val-fraction` | No | `0.2` | Validation split fraction. |
| `--device` | No | `auto` | `cpu`, `cuda`, `mps`, or `auto`. |
| `--features` | No | — | Optional feature flags. |

### Example

```bash
fiber-tracer train \
  --dataset-dir data/processed/training/ \
  --output-dir models/experiments/exp-001/ \
  --model-id fiber_unet_v3 \
  --name "v3 mixed training" \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --device auto
```

### JSON progress

`train` emits compact JSON progress lines to stdout, for example:

```json
{"epoch": 1, "train_loss": 0.421, "val_loss": 0.398, "val_dice": 0.72}
```

These lines are parsed by the TUI Training screen and can be consumed by external orchestrators.

---

## See also

* [`docs/USER_GUIDE.md`](USER_GUIDE.md) — High-level workflow guidance.
* [`docs/parameter_guide.md`](parameter_guide.md) — Full parameter reference and regime-specific guidance.
* [`docs/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) — Common problems and fixes.
* [`docs/methodology.md`](methodology.md) — Algorithm and regime details.
* [`docs/architecture.md`](architecture.md) — Package architecture and module map.
