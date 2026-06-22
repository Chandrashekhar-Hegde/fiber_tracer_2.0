# Fiber Tracer (RAFA) v3.2.0

**Regime-aware 3D fiber analysis for X-ray CT of fiber-reinforced composites, with optional structure-tensor and skeleton backends.**

Fiber Tracer is a Python toolkit for analyzing fiber-reinforced polymer composites from 3D X-ray computed tomography (XCT) images. It selects analysis strategies automatically from the physical voxel-size to fiber-diameter ratio (resolved, marginal, or subvoxel regimes) and supports classical image-processing methods with optional structure-tensor and skeleton backends.

> **Status:** Beta. The public API and CLI are still stabilizing. Results should be treated as experimental and validated on your own data before drawing conclusions.

## What the tool does

- Loads 3D XCT volumes from TIFF stacks or directories of TIFF slices.
- Classifies each dataset into a **resolved**, **marginal**, or **subvoxel** analysis regime.
- In the **resolved regime**: normalizes the image, thresholds the foreground, labels fibers, skeletonizes them, and reports per-fiber orientation and equivalent diameter.
- In the **marginal** and **subvoxel** regimes: computes a gradient structure-tensor orientation field, aggregates it into the Advani–Tucker second-order orientation tensor `A2`, and reports fractional anisotropy (marginal and subvoxel) and orientation distributions (subvoxel).
- Exports results as JSON, CSV, HTML, and TIFF files.
- Provides reproducible synthetic-phantom benchmarks and documented validation protocols.

## What the tool does not do

- It does **not** claim peer-reviewed validation, universal accuracy, or certification for any specific material.
- It does **not** solve differential equations; Runge–Kutta integration and Poincaré–Hopf index tracking are not part of the pipeline.
- It does **not** ship trained machine-learning models or benchmark datasets.
- It does **not** guarantee segmentation quality on noisy, low-contrast, or heavily touching fibers without parameter tuning.

## Installation

The package requires Python ≥3.9.

```bash
# Core package only
pip install -e .

# Core development
pip install -e ".[structure,skeleton,dev]"

# + napari / plotly visualizations
pip install -e ".[viz]"

# All optional backends
pip install -e ".[all]"
```

The `structure` and `skeleton` extras install optional backends used by the test suite and documentation examples. The `viz` extra installs napari and plotly for interactive viewing and HTML reporting. Other extras are defined in `pyproject.toml` but are not required for the current tests.

## Quick start

Generate a synthetic phantom and analyze it in the resolved regime:

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

### Visualize results in napari

```bash
fiber-tracer view --data path/to/stack.tif --output results/
```

### Generate interactive HTML report

```bash
fiber-tracer report-viz --summary results/summary.json --output report.html
```

### Batch processing

Create `batch.yaml`:

```yaml
common:
  voxel_spacing_um: [1.0, 1.0, 1.0]
  fiber_diameter_um: 6.0
  regime: auto
volumes:
  - data_path: sample_a.tif
    output_dir: results/sample_a
  - data_path: sample_b.tif
    output_dir: results/sample_b
    fiber_diameter_um: 4.0
```

Run:

```bash
fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv
```

### Download GF-PA66 validation dataset

```bash
python scripts/download_gfpa66.py --list
python scripts/download_gfpa66.py --file pa66_volumes.h5 --output-dir data/ --accept-license
python scripts/validate_gfpa66.py --data data/pa66_volumes.h5 --output results/gfpa66/ --dataset <name>
```

### CLI commands

- `fiber-tracer run` / `fiber-tracer analyze` — analyze a single volume.
- `fiber-tracer view` — open raw data and results in napari.
- `fiber-tracer report-viz` — generate an interactive Plotly HTML report from `summary.json`.
- `fiber-tracer batch` — process multiple volumes from a batch config.

Run `fiber-tracer --help` for full usage details.

## Example output files

After a run, the output directory contains some or all of the following:

| File                | Description                                                       |
|---------------------|-------------------------------------------------------------------|
| `summary.json`      | Complete JSON summary including regime, metrics, config, and caveats |
| `report.csv`        | Tabular summary for spreadsheet import                            |
| `report.html`       | Human-readable HTML report                                        |
| `labels.tif`        | Label image of segmented fibers (resolved regime)                 |
| `skeleton.tif`      | Per-fiber skeleton image (resolved regime)                        |
| `a2_map.npy`        | Windowed `A2` tensor field (marginal regime)                      |
| `normalized_input.tif` | Intensity-normalized input volume                              |

Not every file is produced in every regime; for example, `a2_map.npy` is written only for marginal analyses.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — Package architecture and module responsibilities.
- [`docs/developer_guide.md`](docs/developer_guide.md) — Development workflow, testing, and contribution guidelines.
- [`docs/methodology.md`](docs/methodology.md) — Algorithms, regime selection, and limitations.
- [`docs/validation_protocol.md`](docs/validation_protocol.md) — Phantom benchmarks, public datasets, metrics, and reproducibility.
- [`docs/parameter_guide.md`](docs/parameter_guide.md) — Configuration options and practical guidance.
- [`docs/RAFA_IMPLEMENTATION_PLAN.md`](docs/RAFA_IMPLEMENTATION_PLAN.md) — Original redesign plan.

## Visualization

Interactive visualization is available via the `viz` extra:

- **napari** (`fiber-tracer view`) — explore raw data, fiber labels, skeletons, and orientation vector layers.
- **plotly** (`fiber-tracer report-viz`) — generate self-contained interactive HTML reports.

Install with:

```bash
pip install -e ".[viz]"
```

## Running the benchmark

```bash
python scripts/benchmark_phantoms.py
```

The script reports Dice score and mean angular error on a deterministic synthetic phantom and asserts the project acceptance thresholds.

## Running the test suite

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

## Citations

If you use this software in your research, please cite the relevant methods:

- Bigün, J., & Granlund, G. H. (1987). Optimal orientation detection of linear symmetry. *ICCV*.
- Jeppesen, N., Mikkelsen, L. P., Dahl, A. B., Christensen, A. N., & Dahl, V. A. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. *Composites Part A*, 149, 106541. DOI:10.1016/j.compositesa.2021.106541
- Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. *Journal of Rheology*, 31(8), 751–784. DOI:10.1122/1.549945
- van der Walt, S., et al. (2014). scikit-image: Image processing in Python. *PeerJ*, 2, e453.

See [`CITATIONS.md`](CITATIONS.md) and [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for additional software and dataset attributions.

## Contact

- **Author:** Chandrashekhar Hegde
- **Email:** <hegde.g.chandrashekhar@gmail.com>
- **Repository:** <https://github.com/llMr-Sweetll/fiber_tracer_2.0>
