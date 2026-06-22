# Installation Guide

This guide covers installing `fiber-tracer` from source. For everyday use we recommend an editable install inside a Python virtual environment so updates and configuration changes are picked up immediately.

## System requirements

- **Python:** `>=3.9` (Python 3.9, 3.10, 3.11, and 3.12 are supported).
- **Operating systems:** Linux, macOS, and Windows.
- **RAM:** At least 4 GB is recommended; 8 GB or more is preferred for 3D volumes.
- **Disk:** A few hundred megabytes for the source checkout and virtual environment. Working volumes and intermediate outputs can require several gigabytes.
- **Build tools:** A recent `pip` (see the troubleshooting section if wheel builds fail).

## Editable install from source

Run these commands in a terminal. The example uses `.venv` for the virtual environment.

```bash
# 1. Clone the repository
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
#    Linux / macOS:
source .venv/bin/activate
#    Windows (PowerShell):
# .venv\Scripts\Activate.ps1
#    Windows (cmd):
# .venv\Scripts\activate.bat

# 4. Install the package in editable mode
pip install -e .
```

After installation, the `fiber-tracer` CLI command is available inside the active environment.

## Optional dependency groups

Optional features are installed using `pip install -e ".[<extra>]"`. The available extras are defined in `pyproject.toml`.

| Extra       | What it enables                                                                 | Install command                              |
|-------------|---------------------------------------------------------------------------------|----------------------------------------------|
| `structure` | Optional `structure-tensor` backend for orientation estimation.                 | `pip install -e ".[structure]"`             |
| `skeleton`  | `skan`-based skeleton-graph adapter for resolved-regime analysis.              | `pip install -e ".[skeleton]"`              |
| `ml`        | PyTorch, torchvision, and scikit-learn for the ML segmentation backend.        | `pip install -e ".[ml]"`                    |
| `unet`      | `nnunetv2` backend for U-Net segmentation. **Linux only.**                     | `pip install -e ".[unet]"`                  |
| `tda`       | `gudhi` backend for Betti numbers and persistence summaries.                   | `pip install -e ".[tda]"`                   |
| `viz`       | `napari` viewer and `plotly` for interactive visualization and HTML reports.   | `pip install -e ".[viz]"`                   |
| `parallel`  | `zarr` and `dask` for chunked and distributed workflows.                       | `pip install -e ".[parallel]"`              |
| `dev`       | `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`, `h5py`, `hypothesis`, `requests`. | `pip install -e ".[dev]"`                 |
| `all`       | Installs all extras listed above (the `unet` extra is skipped on non-Linux).   | `pip install -e ".[all]"`                   |

> **Note:** The `unet` extra depends on `nnunetv2`, which currently builds cleanly on Linux only. On macOS and Windows this extra is excluded automatically because it is declared with a Linux platform marker in `pyproject.toml`.

Multiple extras can be combined in one command:

```bash
pip install -e ".[structure,skeleton,viz,dev]"
```

## Recommended installs for common use cases

### Basic analysis (core only)

Core dependencies are sufficient for running the default resolved, marginal, and subvoxel pipelines without optional backends.

```bash
pip install -e .
```

### Running tests and documentation examples

Many tests and documented examples use the structure-tensor and skeleton-graph backends.

```bash
pip install -e ".[structure,skeleton,dev]"
```

### Interactive visualization

Required for `fiber-tracer view` and `fiber-tracer report-viz`.

```bash
pip install -e ".[viz]"
```

### Large-scale / HPC processing

Install the parallel extra for `zarr` and `dask` support when processing volumes that do not fit in memory or need distributed execution.

```bash
pip install -e ".[parallel]"
```

### Everything (Linux)

Installs all backends and development tools. On Linux this includes the `unet` extra.

```bash
pip install -e ".[all]"
```

## Verification commands

After installation, verify that the package and CLI are working.

### Check the CLI help

```bash
fiber-tracer --help
```

### Check the installed version

```bash
python -c "import fiber_tracer; print(fiber_tracer.__version__)"
```

Expected output starts with the current version, for example:

```text
3.2.0
```

### Run the test suite

```bash
pytest tests/ -q
```

A successful run reports no failures. If tests are skipped, it usually means an optional extra is not installed.

## Platform-specific notes

### macOS (including Apple Silicon)

No special steps are required. Use either native Apple Silicon Python or a Rosetta-based Python interpreter; both work with the source install. If you use Homebrew Python, ensure it is Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[structure,skeleton,dev]"
```

### Windows

Use Git Bash, PowerShell, or the Windows Command Prompt. Path separators differ by shell:

- PowerShell:

  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  pip install -e ".[structure,skeleton,dev]"
  ```

- Command Prompt:

  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -e ".[structure,skeleton,dev]"
  ```

If PowerShell script execution is restricted, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux

All extras are available on Linux, including the `unet` extra. Make sure the system has Python 3.9+ and the standard build tools for compiling any wheels:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential
```

Then install as usual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Troubleshooting

### pip is too old

If installation fails with metadata or build errors, upgrade `pip`, `setuptools`, and `wheel` first:

```bash
pip install --upgrade pip setuptools wheel
```

### Missing build tools

Some optional dependencies compile C/C++ extensions. On Debian/Ubuntu install:

```bash
sudo apt install -y build-essential python3-dev
```

On Fedora/RHEL/CentOS:

```bash
sudo dnf install -y gcc python3-devel
```

On macOS, the Xcode Command Line Tools usually provide the required compilers:

```bash
xcode-select --install
```

### Missing system libraries

- **`numpy`/`scipy` build failures:** Install BLAS/LAPACK development headers (`libopenblas-dev`, `liblapack-dev` on Debian/Ubuntu).
- **`nnunetv2` / `unet` extra fails:** This extra is supported on Linux only. On macOS or Windows it is excluded automatically; do not request `unet` directly on those platforms.
- **`gudhi` / `tda` extra fails:** Install a C++17-capable compiler and `cmake`. See the [GUDHI documentation](https://gudhi.inria.fr/python/latest/installation.html) for platform-specific notes.
- **`napari` / `viz` extra fails:** Make sure a Qt backend is available (`pip install "napari[all]"` inside the `viz` install if needed).

### Import or version mismatch

If `fiber-tracer --help` is not found, confirm the virtual environment is activated and the install completed without errors. If an old version is reported, reinstall with:

```bash
pip install -e .
```

## Contributor setup

If you plan to modify `fiber-tracer`, use the development install and run the test suite and lint checks before committing. Full contributor guidance is in [`docs/developer_guide.md`](developer_guide.md).

```bash
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -q
ruff check .
black --check .
mypy src/fiber_tracer
```
