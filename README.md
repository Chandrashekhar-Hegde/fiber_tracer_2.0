# Fiber Tracer (RAFA) v3.2.0

[![Version](https://img.shields.io/badge/version-3.2.0-blue)](https://github.com/llMr-Sweetll/fiber_tracer_2.0)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/github/license/llMr-Sweetll/fiber_tracer_2.0)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/llMr-Sweetll/fiber_tracer_2.0/ci.yml?branch=main&label=CI)](https://github.com/llMr-Sweetll/fiber_tracer_2.0/actions/workflows/ci.yml)

**Regime-aware 3D fiber analysis for X-ray CT of fiber-reinforced composites.**

Fiber Tracer is a Python toolkit that analyzes fiber-reinforced polymer composites from 3D X-ray computed tomography (XCT) volumes. It selects analysis algorithms automatically from the physical voxel-size to fiber-diameter ratio—resolved, marginal, or subvoxel regimes—and reports per-fiber or population-level orientation and morphometry using classical image processing and tensor methods.

> **Project status:** Beta. The public API and CLI are stabilizing. Results should be treated as experimental and validated on your own data before drawing conclusions.

## Table of contents

- [What this tool does](#what-this-tool-does)
- [What this tool does not do](#what-this-tool-does-not-do)
- [Installation](#installation)
- [Quick start](#quick-start)
- [First analysis](#first-analysis)
- [CLI overview](#cli-overview)
- [Model Registry, Experiments, and Training](#model-registry-experiments-and-training)
- [Terminal UI (TUI)](#terminal-ui-tui)
- [Working with real data](#working-with-real-data)
- [Visualization](#visualization)
- [Batch processing](#batch-processing)
- [Validation and benchmarking](#validation-and-benchmarking)
- [Output files reference](#output-files-reference)
- [Configuration overview](#configuration-overview)
- [Documentation index](#documentation-index)
- [Development and testing](#development-and-testing)
- [Citations and license](#citations-and-license)
- [Contact](#contact)

## What this tool does

Fiber Tracer implements **Regime-Aware Fiber Analysis (RAFA)**. The analysis pipeline is selected from the ratio

```
r = min(voxel_spacing_z, voxel_spacing_y, voxel_spacing_x) / fiber_diameter_um
```

using the minimum voxel spacing so that anisotropic datasets are classified conservatively:

| Regime   | Threshold        | Physical situation                              | Pipeline summary                                                                 |
|----------|------------------|-------------------------------------------------|----------------------------------------------------------------------------------|
| resolved | `r <= 0.3`       | Fiber diameter is much larger than a voxel      | Segmentation, labeling, skeletonization, per-fiber PCA orientation, and equivalent diameter. |
| marginal | `0.3 < r <= 3.0` | Fiber diameter is comparable to a voxel         | Gradient structure-tensor orientation field, windowed Advani–Tucker `A2` tensor map, fractional anisotropy. |
| subvoxel | `r > 3.0`        | Many fibers fit inside a single voxel           | Gradient structure tensor with enlarged integration scale, global `A2`, fractional anisotropy, orientation distribution. |
| auto     | —                | Select from `r` automatically                   | Uses the same thresholds as above.                                               |

Across all regimes the tool writes machine-readable JSON, CSV, and HTML reports with configuration metadata, citations, and regime-specific caveats. Detailed algorithm discussion is provided in [`docs/methodology.md`](docs/methodology.md).

## What this tool does not do

- It does **not** claim peer-reviewed validation, universal accuracy, or certification for any specific material.
- It does **not** solve differential equations; Runge–Kutta integration and Poincaré–Hopf index tracking are not part of the pipeline.
- It ships an optional pre-trained 3D U-Net for fiber segmentation, but the model should be validated on your own data before drawing conclusions.
- It does **not** guarantee segmentation quality on noisy, low-contrast, or heavily touching fibers without parameter tuning.
- It does **not** perform out-of-core analysis from the CLI; large volumes require programmatic use of the `fiber_tracer.chunked` helpers (see [`docs/architecture.md`](docs/architecture.md)).

## Installation

Fiber Tracer requires Python ≥3.10 and is installed as `fiber-tracer`. The source repository is at `https://github.com/llMr-Sweetll/fiber_tracer_2.0`.

```bash
# Clone the repository
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install options:

```bash
# Core package only
pip install -e .

# Recommended for tests and documentation examples
pip install -e ".[structure,skeleton,dev]"

# Add interactive visualization (napari and plotly)
pip install -e ".[viz]"

# All optional backends (Linux-only nnUNet extra is skipped on other platforms)
pip install -e ".[all]"
```

Optional extras:

| Extra        | Purpose                                                             |
|--------------|---------------------------------------------------------------------|
| `structure`  | Optional `structure-tensor` backend for orientation estimation.     |
| `skeleton`   | `skan`-based skeleton-graph adapter.                                |
| `ml`         | PyTorch/scikit-learn support; enables the custom 3D U-Net backend (`unet` segmentation method). |
| `unet`       | `nnunetv2` backend (Linux only).                                    |
| `tda`        | `gudhi` backend for Betti numbers and persistence summaries.        |
| `viz`        | `napari` viewer and `plotly` HTML reports.                          |
| `parallel`   | `zarr` and `dask` for chunked and distributed workflows.            |
| `dev`        | `pytest`, `black`, `ruff`, `mypy`, `h5py`, `hypothesis`, `requests`. |
| `all`        | All of the above.                                                   |

Verify the installation:

```bash
fiber-tracer --help
```

## Quick start

Install and launch the terminal UI in one step:

```bash
curl -fsSL https://raw.githubusercontent.com/llMr-Sweetll/fiber_tracer_2.0/main/scripts/install.sh | bash
cd fiber_tracer_2.0
source .venv/bin/activate
cd tui && bun run dev
```

For a manual install or Windows PowerShell, see [`docs/INSTALL.md`](docs/INSTALL.md).

## First analysis

Generate a deterministic synthetic phantom and analyze it in the resolved regime:

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

fiber-tracer \
  --data phantom.tif \
  --output output/ \
  --voxel-spacing 1.0 1.0 1.0 \
  --fiber-diameter 4.0 \
  --regime resolved
```

Expected output directory contents:

```text
output/
├── labels.tif
├── normalized_input.tif
├── report.csv
├── report.html
├── skeleton.tif
└── summary.json
```

Inspect the JSON summary:

```bash
python - <<'PY'
import json
with open("output/summary.json") as f:
    summary = json.load(f)
print("Regime:", summary["regime"])
print("Labels:", summary["n_labels"])
PY
```

## CLI overview

`fiber-tracer` exposes eight subcommands. Top-level flags such as `--data`, `--output`, `--config`, `--voxel-spacing`, `--fiber-diameter`, and `--regime` are accepted for backward compatibility and are equivalent to `fiber-tracer run ...`.

| Subcommand         | Alias | Purpose                                               | Example one-liner                                                                 |
|--------------------|-------|-------------------------------------------------------|-------------------------------------------------------------------------------------|
| `run`              | —     | Run the RAFA pipeline on a single volume (default).   | `fiber-tracer --data stack.tif --output results/ --voxel-spacing 1.0 1.0 1.0 --fiber-diameter 6.0` |
| `analyze`          | `run` | Alias for `run`.                                      | `fiber-tracer analyze --data stack.tif --output results/ ...`                       |
| `view`             | —     | Open raw data and results in napari.                  | `fiber-tracer view --data stack.tif --output results/`                              |
| `report-viz`       | —     | Generate an interactive Plotly HTML report.           | `fiber-tracer report-viz --summary output/summary.json --output report.html`       |
| `batch`            | —     | Process multiple volumes from a YAML/JSON config.     | `fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv`         |
| `model`            | —     | Manage registered segmentation models.                | `fiber-tracer model list`                                                           |
| `experiment`       | —     | Manage and compare training experiments.              | `fiber-tracer experiment list`                                                      |
| `train`            | —     | Train a 3D U-Net from a dataset directory.            | `fiber-tracer train --dataset-dir data/training/ --output-dir models/exp-001/`    |

Run `fiber-tracer <subcommand> --help` for detailed options.

## Model Registry, Experiments, and Training

Fiber Tracer includes local, file-based management for segmentation models and training experiments. Data is stored under `~/.config/fiber-tracer/` and surfaced in both the CLI and the TUI.

### Model Registry

Register local checkpoints so they can be referenced by ID:

```bash
fiber-tracer model add \
  --model-id fiber_unet_v2_full \
  --name "Production 3D U-Net" \
  --path models/fiber_unet_v2_full.pt

fiber-tracer model list
fiber-tracer model set-default fiber_unet_v2_full
```

### Experiments

Training runs are recorded as experiments with hyper-parameters and metrics:

```bash
fiber-tracer experiment list
fiber-tracer experiment show exp-20260624-abc123
fiber-tracer experiment compare exp-001 exp-002 --metric val_dice
```

### Training

Train a 3D U-Net from a prepared dataset directory:

```bash
fiber-tracer train \
  --dataset-dir data/processed/training/ \
  --output-dir models/experiments/exp-001/ \
  --model-id fiber_unet_v3 \
  --name "v3 mixed training" \
  --epochs 20 \
  --batch-size 4 \
  --device auto
```

**Note:** Training requires the `ml` extra: `pip install -e ".[ml]"`.

See [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) for the full guide, and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for model limitations and retraining advice.

## Terminal UI (TUI)

A guided, keyboard-driven terminal UI is available in the `tui/` directory. It provides a wizard for new analyses, a results dashboard, run history, logs, and settings with customizable themes.

Requirements: [Bun](https://bun.sh/) ≥ 1.2 and [Node.js](https://nodejs.org/) ≥ 18.

```bash
cd tui
bun install
bun run dev
```

Use the number keys to switch sections, `←`/`→` or `p`/`n` to move through the wizard, `Enter` to select, `r` to run from the Review step, and `q` to quit. See [`tui/README.md`](tui/README.md) for the full shortcut list and theme configuration.

## Working with real data

Fiber Tracer reads single multi-page TIFF stacks or directories of TIFF slices. Two pieces of metadata are mandatory:

1. **Voxel spacing** in micrometres, given in `(z, y, x)` order.
2. **Expected fiber diameter** in micrometres.

These values drive regime selection and convert voxel measurements to physical units.

```bash
fiber-tracer \
  --data sample_a.tif \
  --output results/sample_a/ \
  --voxel-spacing 1.0 1.0 1.0 \
  --fiber-diameter 6.0 \
  --regime auto
```

For anisotropic voxels, use the actual spacing values:

```bash
fiber-tracer \
  --data sample_b.tif \
  --output results/sample_b/ \
  --voxel-spacing 2.0 1.0 1.0 \
  --fiber-diameter 8.0 \
  --regime auto
```

Anisotropic spacing is supported, but accuracy degrades for large anisotropy ratios. Consider resampling to isotropic voxels when ratios exceed approximately 2–3. See [`docs/parameter_guide.md`](docs/parameter_guide.md) for guidance on denoising, segmentation methods, and structure-tensor scales.

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

## Batch processing

The `batch` subcommand processes multiple volumes from a single YAML or JSON configuration. A `common` block supplies shared settings; each entry in `volumes` can override them.

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

A ready-to-use batch example is provided in [`batch_example.yaml`](batch_example.yaml).

## Validation and benchmarking

### Synthetic phantom benchmark

The primary automated benchmark uses deterministic straight-fiber phantoms with known orientations:

```bash
python scripts/benchmark_phantoms.py
```

A passing run reports a resolved-regime Dice score above 0.85 and a mean angular error below 5°. Typical observed values on the default phantom are approximately 0.98 and 0.09°, respectively. The script exits with code `0` when acceptance thresholds are met.

### GF-PA66 public dataset

The GF-PA66 XCT dataset is used as an external reference for benchmarking. It is not redistributed with this package.

- **Citation:** Bertoldo, J. P. C. et al. (2021). A Modular U-Net for Automated Segmentation of X-Ray Tomography Images in Composite Materials. *Frontiers in Materials*, 8, 761229. DOI:10.3389/fmats.2021.761229
- **License:** CC BY-SA 4.0
- **DOI:** 10.5281/zenodo.4587827
- **File:** `pa66_volumes.h5`

List, download, and validate:

```bash
python scripts/download_gfpa66.py --list
python scripts/download_gfpa66.py --file pa66_volumes.h5 --output-dir data/ --accept-license
python scripts/validate_gfpa66.py --data data/pa66_volumes.h5 --output results/gfpa66/
```

`validate_gfpa66.py` auto-detects the HDF5 dataset when the file contains a single dataset or a common name (`data`, `image`, `volume`, `XCT`). Use `--dataset <name>` only when the file contains multiple datasets and auto-detection fails.

Users who create and redistribute derived works based on the GF-PA66 dataset must comply with the CC BY-SA 4.0 share-alike terms.

## Output files reference

The pipeline always writes `summary.json`, `report.csv`, and `report.html`. Regime-specific files are produced only when relevant.

| File                  | Regime(s)              | Description                                                                 |
|-----------------------|------------------------|-------------------------------------------------------------------------------|
| `summary.json`        | all                    | Complete JSON summary including regime, metrics, configuration, citations, and caveats. |
| `report.csv`          | all                    | Tabular summary; per-fiber rows in resolved regime, global metrics in marginal/subvoxel regimes. |
| `report.html`         | all                    | Human-readable HTML report.                                                   |
| `labels.tif`          | resolved               | Label image of segmented fibers.                                              |
| `skeleton.tif`        | resolved               | Per-fiber skeleton image.                                                     |
| `normalized_input.tif`| resolved               | Intensity-normalized input volume.                                            |
| `a2_map.npy`          | marginal               | Windowed Advani–Tucker `A2` tensor field.                                     |
| `a2_centers.npy`      | marginal               | Physical coordinates of the window centers for `a2_map.npy`.                  |

In the marginal and subvoxel regimes the normalized volume is kept in memory but is not saved by default. See [`docs/architecture.md`](docs/architecture.md) for the data-flow diagram and module responsibilities.

## Configuration overview

Configuration can be supplied as YAML or JSON and overridden from the command line. Command-line values take precedence over configuration-file values. The full option set is documented in [`docs/parameter_guide.md`](docs/parameter_guide.md); key groups include:

- `processing`: denoising sigma, normalization.
- `segmentation`: method (`otsu`, `watershed`, or `unet`), plus `model_path` for the custom U-Net backend.
- `orientation`: structure-tensor inner scale `sigma_um`, integration scale `rho_um`, marginal window size.
- `analysis`: morphometry, orientation tensor, and optional topological descriptors.

Example `config.yaml`:

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

Run with:

```bash
fiber-tracer --config config.yaml --data path/to/stack.tif --output results/
```

A ready-to-use copy of this example is provided in [`config_example.yaml`](config_example.yaml). The batch equivalent is in [`batch_example.yaml`](batch_example.yaml).

### Optional backends

Set `segmentation.method: "unet"` to use the lightweight 3D U-Net implemented in `fiber_tracer.backends.unet3d`. This requires the `ml` extra and a trained checkpoint:

```bash
pip install -e ".[ml]"
# Use the pre-trained production model (download from GitHub Releases)
# https://github.com/llMr-Sweetll/fiber_tracer_2.0/releases/tag/v3.2.0-unet-v2
fiber-tracer --data stack.tif --output results/ \
  --segmentation-method unet --model-path models/fiber_unet_v2_full.pt

# Or train your own on mixed synthetic + open XCT data
python scripts/download_datasets.py
python scripts/prepare_training_data.py --n-synthetic 1000 --n-patches-per-volume 64
python scripts/train_unet_mixed.py --epochs 100 --output models/fiber_unet_v2.pt
```

The production model (`fiber_unet_v2_full.pt`) was trained on 2,152 mixed patches covering glass/carbon, UD/woven/short/broken fibers, and failure cases from Henry Royce Institute, DTU, and IVW open XCT datasets. Held-out validation on the GF-PA66 ground truth achieves **Dice ≈ 0.90 / IoU ≈ 0.81** on central slices.

> **Important:** These validation numbers are a single-reference benchmark, not a universal accuracy guarantee. The model was trained on a specific distribution of open datasets and synthetic phantoms; it may fail on out-of-distribution data (different contrast, resolution, fiber type, or artifact levels). Always validate on your own volumes before reporting results. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for a rigorous discussion of limitations, failure modes, and retraining instructions.

Set `analysis.compute_tda_descriptors: true` to compute Betti numbers and persistence summaries with `gudhi` (requires the `tda` extra).

## Documentation index

| Document | Description |
|----------|-------------|
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Beginner-to-intermediate walkthrough: regimes, first analysis, configuration, batching, visualization. |
| [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) | Complete `fiber-tracer` command-line reference generated from the actual CLI. |
| [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) | Model registry, experiment tracking, and the `fiber-tracer train` quick-start guide. |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Detailed installation instructions, extras, platform notes, and verification. |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Common errors, causes, and fixes. |
| [`docs/parameter_guide.md`](docs/parameter_guide.md) | Complete configuration reference, CLI flags, and practical guidance by regime. |
| [`docs/methodology.md`](docs/methodology.md) | Algorithms, regime selection, structure-tensor formulation, Advani–Tucker tensor, and limitations. |
| [`docs/architecture.md`](docs/architecture.md) | Package architecture, module map, data flow, and extension points. |
| [`docs/validation_protocol.md`](docs/validation_protocol.md) | Phantom benchmarks, GF-PA66 dataset instructions, metrics, and reproducibility notes. |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Detailed model card for the production 3D U-Net: training data, validation, limitations, and retraining. |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Tuning guidance for runtime, memory, and U-Net batch size. |
| [`docs/developer_guide.md`](docs/developer_guide.md) | Development setup, testing, linting, type checking, and contribution workflow. |
| [`docs/RAFA_IMPLEMENTATION_PLAN.md`](docs/RAFA_IMPLEMENTATION_PLAN.md) | Original redesign plan for the RAFA pipeline. |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history and release notes. |
| [`ROADMAP.md`](ROADMAP.md) | Short-, medium-, and long-term project roadmap. |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to set up a dev environment, run tests, and submit PRs. |
| [`CITATIONS.md`](CITATIONS.md) | Academic and software citations. |
| [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) | Dependency and dataset licenses. |

## Development and testing

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run code-quality checks:

```bash
ruff check .
black --check .
mypy src/fiber_tracer
```

Full contributor guidance, including how to add backends, regimes, and visualizations, is in [`docs/developer_guide.md`](docs/developer_guide.md).

## Citations and license

Fiber Tracer is released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

If you use this software in your research, please cite the relevant methods:

- Bigün, J., & Granlund, G. H. (1987). Optimal orientation detection of linear symmetry. *Proceedings of the First International Conference on Computer Vision (ICCV)*.
- Jeppesen, N., Mikkelsen, L. P., Dahl, A. B., Christensen, A. N., & Dahl, V. A. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. *Composites Part A*, 149, 106541. DOI:10.1016/j.compositesa.2021.106541
- Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. *Journal of Rheology*, 31(8), 751–784. DOI:10.1122/1.549945
- van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: Image processing in Python. *PeerJ*, 2, e453.

Additional software and dataset attributions are listed in [`CITATIONS.md`](CITATIONS.md) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Contact

- **Author:** Chandrashekhar Hegde
- **Email:** [hegde.g.chandrashekhar@gmail.com](mailto:hegde.g.chandrashekhar@gmail.com)
- **Repository:** [https://github.com/llMr-Sweetll/fiber_tracer_2.0](https://github.com/llMr-Sweetll/fiber_tracer_2.0)
- **Issues:** [https://github.com/llMr-Sweetll/fiber_tracer_2.0/issues](https://github.com/llMr-Sweetll/fiber_tracer_2.0/issues)
