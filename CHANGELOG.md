# Changelog

All notable changes to the Fiber Tracer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-06-21

### Added
- Hypothesis property-based tests for metrics and phantom generation.
- Phantom benchmark regression test (`tests/test_benchmark_regression.py`) and CI integration.
- zarr-based chunked processing helpers (`src/fiber_tracer/chunked.py`) for out-of-core normalization and Gaussian denoising.
- Memory profiler script (`scripts/profile_pipeline_memory.py`).

### Changed
- CI workflow runs the benchmark regression test once per matrix job and uploads `benchmark_results.json`.
- `.gitignore` now excludes `benchmark_results/` and `profile_tmp/`.

### Fixed
- Clarified that chunked helpers are zarr-backed; `dask` remains available via the `parallel` extra.

## [3.0.0] - 2026-06-21

### Added

- Regime-aware analysis pipeline (`resolved`, `marginal`, `subvoxel`) selected from the physical voxel/fiber ratio.
- New modular package layout under `src/fiber_tracer/`:
  - `config.py` — dataclass-based configuration with validation
  - `cli.py` — command-line interface
  - `pipeline.py` — pipeline orchestrator
  - `preprocess.py` — normalization and physical-unit denoising
  - `segmentation/classical.py` — Otsu thresholding, connected components, watershed
  - `centerline/skeleton.py` and `centerline/graph.py` — skeletonization helpers
  - `orientation/structure_tensor.py`, `orientation/pca.py`, `orientation/tensor.py` — orientation estimation and Advani–Tucker `A2`
  - `analysis/morphometry.py` — equivalent diameter and centerline measures
  - `reporting/json.py`, `reporting/csv.py`, `reporting/html.py` — honest report exporters with caveats
  - `validation/phantoms.py` and `validation/metrics.py` — synthetic phantoms and validation metrics
  - `io.py`, `regime.py`, `exceptions.py`
- Synthetic straight-fiber phantom generator with ground-truth orientations.
- Validation metrics: Dice score, mean angular error, orientation tensor error, fractional anisotropy.
- `scripts/benchmark_phantoms.py` with documented acceptance thresholds.
- Documentation: `docs/methodology.md`, `docs/validation_protocol.md`, `docs/parameter_guide.md`.
- `pyproject.toml` with Python ≥3.9 support and optional dependency groups.
- `CITATIONS.md` and `THIRD_PARTY_LICENSES.md` for academic and software attributions.

### Changed

- Package layout moved to `src/fiber_tracer/`.
- Minimum Python version set to 3.9.
- Project metadata rewritten to be honest and defensible (no peer-review or universal-accuracy claims).
- CLI rewritten as `fiber-tracer` entry point.
- Reporting outputs now include configuration, citations, and regime-specific caveats.

### Removed

- Dead code, monolithic scripts, and unused modules from earlier versions.
- False or unsupported claims (peer-review badges, "official algorithms", "pure math").
- GPL-incompatible dependencies from the default dependency set.
- Deprecated `requirements.txt`, `environment.yml`, and old configuration examples.

---

## [2.0.0] - 2025-01-08

### Added
- **Modular Architecture**: Complete refactoring into separate modules
  - `preprocessing.py`: Image loading and preprocessing
  - `segmentation.py`: Fiber segmentation algorithms
  - `analysis.py`: Fiber property analysis
  - `visualization.py`: Visualization generation
  - `core.py`: Pipeline orchestration
  - `config.py`: Configuration management
  - `utils.py`: Utility functions and warning suppression

- **Configuration Management**
  - YAML/JSON configuration file support
  - Dataclass-based configuration with validation
  - Example configuration file (`config_example.yaml`)

- **Enhanced Analysis**
  - Fiber connectivity analysis
  - Automatic fiber classification by orientation
  - Extended fiber properties (surface area, aspect ratio)
  - Fiber bundle detection

- **Improved Visualizations**
  - Interactive 3D plots with Plotly
  - Correlation analysis plots
  - Orientation distribution plots
  - Comprehensive summary dashboard

- **Performance Improvements**
  - Parallel processing with configurable workers
  - Memory-mapped arrays for large datasets
  - Chunked processing for memory efficiency
  - Optimized segmentation algorithms

- **Developer Tools**
  - Comprehensive test suite (`test_fiber_tracer.py`)
  - Synthetic data generation for testing
  - Progress logging utilities
  - Dependency checking

- **Documentation**
  - Detailed setup guide (`SETUP_GUIDE.md`)
  - Enhanced README with API documentation
  - Inline code documentation
  - Performance benchmarks

- **Environment Support**
  - Conda environment file (`environment.yml`)
  - Updated requirements with version constraints
  - Platform-specific installation notes

### Changed
- **File Structure**: Reorganized into proper Python package
- **CLI Interface**: New argument structure with `fiber_tracer_v2.py`
- **Logging**: Enhanced logging with file and console output
- **Error Handling**: Improved error messages and recovery
- **Dependencies**: Updated to latest stable versions with constraints

### Fixed
- Memory leaks in large dataset processing
- Segmentation accuracy for low-contrast images
- Fiber tracking across slices
- Visualization generation for edge cases

### Deprecated
- Original monolithic script (`fiber_tracer_cli.py`) - maintained for backward compatibility

## [1.0.0] - 2024-09-15

### Initial Release
- Basic fiber tracing functionality
- TIFF image processing
- 3D volume reconstruction
- Adaptive thresholding segmentation
- Fiber property extraction (length, diameter, volume, orientation, tortuosity)
- Volume fraction calculation
- Basic visualizations (heatmap, histogram)
- Mayavi 3D visualization support
- Command-line interface
- Comprehensive README documentation

### Features
- Process X-ray CT images of GFRP composites
- Handle large datasets (>15GB)
- Parallel processing support
- Memory-efficient chunk processing
- CSV output of fiber properties
- Fiber classification by orientation
