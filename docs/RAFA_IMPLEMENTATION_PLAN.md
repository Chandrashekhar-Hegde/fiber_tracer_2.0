# RAFA Implementation Plan — Final

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `fiber_tracer_2.0` as a cross-platform, honest, regime-aware fiber-analysis tool with classical + optional ML backends, optional TDA descriptors, and reproducible validation on public data.

**Architecture:** A plugin-based core (`fiber_tracer/`) stays MIT-licensed and imports only permissive dependencies. Heavy/optional backends (PyTorch/nnU-Net, gudhi) are lazy-loaded through adapter modules. Three analysis regimes (resolved / marginal / sub-voxel) select algorithms automatically from physical voxel/fiber ratio. All outputs are validated against synthetic phantoms and public datasets.

**Tech Stack:** Python ≥3.9; `numpy`, `scipy`, `scikit-image` (BSD), `pandas`, `tifffile`, `pyyaml`; optional `structure-tensor` (MIT), `skan` (BSD-3), `torch`/`torchvision`/`scikit-learn` (BSD-3), `gudhi` ≥3.9 (MIT Python modules). **Excluded:** `ripser` and `kimimaro` (GPL-3, license-incompatible).

---

## 0. License & Compliance Guardrails (applies to every task)

- The project remains **MIT licensed**.
- **Never import GPL/AGPL/SSPL code into the core package.** Optional adapters may depend on copyleft libraries only if they are isolated plugins and never required by default.
- All third-party code, data, and models get attribution in `THIRD_PARTY_LICENSES.md`.
- Public datasets are used for benchmarking only unless their share-alike terms are explicitly handled.
- No vendoring of upstream source. Depend on PyPI wheels and cite them.
- Before shipping any model trained on CC BY-SA data, flag for legal review.

---

## Phase 0 — Surgical Demolition & Legal Hygiene

### Task 0.1: Audit and freeze the existing repo state

**Files:**
- Read: entire repo
- Create: `ARCHITECTURE_DECISIONS.md` (initial entry)

- [ ] **Step 1: Record current commit hash and create a safety branch**

```bash
git branch pre-rafa-archived
git log --oneline -1 > ARCHIVED_COMMIT.txt
```

- [ ] **Step 2: List files to remove and files to keep**

Remove:
- `fiber_tracer/ascii_art.py`
- `fiber_tracer/premium_report_generator.py`
- `fiber_tracer/run_wizard.py` (if it exists at root)
- `run_wizard.py`
- `fiber_tracer_cli.py`
- `fiber_tracer_v2.py`
- `benchmark_ht3.py`
- `test_fiber_tracer.py`
- `DISSERTATION_THEORY.md`
- `PEER_REVIEW.md`
- `README_old.md`
- `SETUP_GUIDE.md` (to be rewritten)
- `CONTRIBUTING.md` (to be rewritten)
- `CONTRIBUTORS.md` (to be rewritten)
- `config_example.yaml` (to be replaced)
- `requirements.txt` (to be replaced)
- `environment.yml` (to be replaced)
- `pyproject.toml` (to be replaced)
- `docs/images/` (keep only if genuinely used)

Keep (to be rewritten/refactored):
- `README.md`
- `LICENSE`
- `CHANGELOG.md`
- `fiber_tracer/__init__.py` (rewrite)
- `fiber_tracer/config.py` (rewrite)
- `fiber_tracer/core.py` (rewrite into `pipeline.py`)
- `fiber_tracer/preprocessing.py` (rewrite)
- `fiber_tracer/segmentation.py` (split and rewrite)
- `fiber_tracer/analysis.py` (split into `orientation/`, `analysis/`, `centerline/`)
- `fiber_tracer/visualization.py` (rewrite)
- `fiber_tracer/utils.py` (rewrite)

- [ ] **Step 3: Commit the demolition list**

```bash
git add ARCHIVED_COMMIT.txt ARCHITECTURE_DECISIONS.md
git commit -m "docs: record pre-RAFA state and demolition plan"
```

---

### Task 0.2: Replace project metadata and legal notices

**Files:**
- Create: `pyproject.toml`, `THIRD_PARTY_LICENSES.md`, `CITATIONS.md`
- Modify: `README.md` (header only in this task)

- [ ] **Step 1: Write `pyproject.toml` with correct metadata and optional deps**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fiber-tracer"
version = "3.0.0"
description = "Regime-aware 3D fiber analysis for X-ray CT of fiber-reinforced composites"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Chandrashekhar Hegde", email = "hegde.g.chandrashekhar@gmail.com"}
]
requires-python = ">=3.10"
keywords = [
    "fiber analysis",
    "composite materials",
    "X-ray CT",
    "image processing",
    "structure tensor",
    "fiber orientation"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-image>=0.21.0,<0.27.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "tifffile>=2023.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
]

[project.optional-dependencies]
structure = ["structure-tensor>=0.3.0"]
skeleton = ["skan>=0.13.0"]
ml = ["torch>=2.0.0", "torchvision>=0.15.0", "scikit-learn>=1.3.0"]
unet = ["nnunetv2>=2.4.0"]
tda = ["gudhi>=3.9.0"]
viz = ["plotly>=5.18.0", "napari>=0.4.18"]
parallel = ["zarr>=2.16.0", "dask>=2023.0.0"]
dev = ["pytest>=7.4.0", "pytest-cov", "black", "ruff", "mypy"]
all = ["fiber-tracer[structure,skeleton,ml,unet,tda,viz,parallel,dev]"]

[project.scripts]
fiber-tracer = "fiber_tracer.cli:main"

[project.urls]
Homepage = "https://github.com/llMr-Sweetll/fiber_tracer_2.0"
Documentation = "https://github.com/llMr-Sweetll/fiber_tracer_2.0#readme"
Repository = "https://github.com/llMr-Sweetll/fiber_tracer_2.0.git"
Issues = "https://github.com/llMr-Sweetll/fiber_tracer_2.0/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
fiber_tracer = ["*.yaml", "*.yml", "*.json"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-ra -q --strict-markers"
```

- [ ] **Step 2: Write `THIRD_PARTY_LICENSES.md`**

```markdown
# Third-Party Licenses and Attributions

## Core dependencies

- **NumPy** — BSD-3-Clause — https://numpy.org/
- **SciPy** — BSD-3-Clause — https://scipy.org/
- **scikit-image** — BSD-3-Clause — https://scikit-image.org/
- **pandas** — BSD-3-Clause — https://pandas.pydata.org/
- **Matplotlib** — PSF-based license — https://matplotlib.org/
- **tifffile** — BSD-3-Clause — https://pypi.org/project/tifffile/
- **PyYAML** — MIT — https://pyyaml.org/

## Optional dependencies

- **structure-tensor** — MIT — Copyright Vedrana Andersen Dahl and Niels Jeppesen — https://github.com/Skielex/structure-tensor
- **skan** — BSD-3-Clause — Copyright Juan Nunez-Iglesias — https://github.com/jni/skan
- **PyTorch** — BSD-3-Clause — Copyright PyTorch contributors — https://pytorch.org/
- **nnU-Net** — Apache-2.0 — Copyright DKFZ — https://github.com/MIC-DKFZ/nnUNet
- **GUDHI** — MIT (Python modules ≥3.9.0) — https://gudhi.inria.fr/


## Datasets used for validation

- **GF-PA66 3D XCT** — CC BY-SA 4.0 — DOI:10.5281/zenodo.4587827 — Bertoldo et al., Front. Mater. 2021.

See `CITATIONS.md` for academic citations.
```

- [ ] **Step 3: Write `CITATIONS.md`**

```markdown
# Academic Citations

## Algorithms

- Bigün, J., & Granlund, G. H. (1987). Optimal orientation detection of linear symmetry. *ICCV*.
- Jeppesen, N., Mikkelsen, L. P., Dahl, A. B., Christensen, A. N., & Dahl, V. A. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. *Composites Part A*, 149, 106541. DOI:10.1016/j.compositesa.2021.106541
- Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. *Journal of Rheology*, 31(8), 751–784. DOI:10.1122/1.549945

## Software

- van der Walt et al. (2014). scikit-image: Image processing in Python. *PeerJ*, 2, e453.
- Nunez-Iglesias, J. & skan contributors. skan: skeleton analysis in Python.
- Vedrana Andersen Dahl and Niels Jeppesen. structure-tensor.

## Validation data

- Bertoldo J.P.C. et al. (2021). A Modular U-Net for Automated Segmentation of X-Ray Tomography Images in Composite Materials. *Front. Mater.* 8:761229. DOI:10.3389/fmats.2021.761229
```

- [ ] **Step 4: Update `README.md` header to remove false claims**

Remove badges: "Validated-Peer Review Level", "Official Algorithms", "Pure Math". Replace with honest description.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml THIRD_PARTY_LICENSES.md CITATIONS.md README.md
git commit -m "chore: replace project metadata and third-party attributions"
```

---

## Phase 1 — Core Foundation

### Task 1.1: Create the new package skeleton

**Files:**
- Create: `src/fiber_tracer/__init__.py`, `src/fiber_tracer/cli.py`, `src/fiber_tracer/config.py`, `src/fiber_tracer/exceptions.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Write `src/fiber_tracer/__init__.py`**

```python
"""RAFA: Regime-Aware Fiber Analysis for X-ray CT."""

__version__ = "3.0.0"
__author__ = "Chandrashekhar Hegde"
__email__ = "hegde.g.chandrashekhar@gmail.com"
__license__ = "MIT"

from fiber_tracer.config import Config
from fiber_tracer.pipeline import FiberAnalysisPipeline

__all__ = ["Config", "FiberAnalysisPipeline"]
```

- [ ] **Step 2: Write `src/fiber_tracer/exceptions.py`**

```python
"""Custom exceptions for fiber_tracer."""


class FiberTracerError(Exception):
    """Base exception."""


class ConfigError(FiberTracerError):
    """Invalid configuration."""


class DataError(FiberTracerError):
    """Data loading or validation error."""


class BackendNotAvailableError(FiberTracerError):
    """Optional backend dependency is missing."""


class ValidationError(FiberTracerError):
    """Validation metric or phantom generation failed."""
```

- [ ] **Step 3: Write `src/fiber_tracer/config.py`**

```python
"""Configuration management with validation and units."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import os
import json
import yaml


@dataclass
class VoxelSpacing:
    z: float
    y: float
    x: float

    def is_isotropic(self, tol: float = 1e-3) -> bool:
        return max(abs(self.z - self.y), abs(self.z - self.x), abs(self.y - self.x)) < tol


@dataclass
class ProcessingConfig:
    denoise_sigma: Optional[float] = None
    normalize: bool = True
    anisotropic_spacing: Optional[VoxelSpacing] = None


@dataclass
class SegmentationConfig:
    method: str = "otsu"  # otsu, watershed, adaptive, unet
    min_fiber_diameter_um: float = 10.0
    max_fiber_diameter_um: float = 50.0
    watershed_seed_sigma_um: Optional[float] = None


@dataclass
class OrientationConfig:
    method: str = "structure_tensor"  # structure_tensor, pca
    sigma_um: Optional[float] = None
    rho_um: Optional[float] = None
    window_size_um: Optional[float] = None


@dataclass
class AnalysisConfig:
    compute_morphometry: bool = True
    compute_orientation_tensor: bool = True
    compute_tda_descriptors: bool = False


@dataclass
class Config:
    data_path: str = ""
    output_dir: str = ""
    voxel_spacing_um: VoxelSpacing = field(default_factory=lambda: VoxelSpacing(1.0, 1.0, 1.0))
    fiber_diameter_um: float = 10.0
    regime: str = "auto"  # auto, resolved, marginal, subvoxel
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def validate(self) -> None:
        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError(f"data_path does not exist: {self.data_path}")
        if self.fiber_diameter_um <= 0:
            raise ValueError("fiber_diameter_um must be positive")
        for s in (self.voxel_spacing_um.z, self.voxel_spacing_um.y, self.voxel_spacing_um.x):
            if s <= 0:
                raise ValueError("voxel spacing must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            if path.endswith((".yaml", ".yml")):
                yaml.safe_dump(self.to_dict(), f)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)
```

- [ ] **Step 4: Write `src/fiber_tracer/cli.py` skeleton**

```python
"""Command-line interface for fiber_tracer."""

import argparse
import sys
from pathlib import Path

from fiber_tracer.config import Config
from fiber_tracer.pipeline import FiberAnalysisPipeline


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="RAFA fiber analysis")
    parser.add_argument("--data", required=True, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Path to YAML/JSON config")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, metavar=("Z", "Y", "X"))
    parser.add_argument("--fiber-diameter", type=float)
    parser.add_argument("--regime", choices=["auto", "resolved", "marginal", "subvoxel"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config(data_path=args.data, output_dir=args.output)

    if args.voxel_spacing:
        config.voxel_spacing_um = VoxelSpacing(*args.voxel_spacing)
    if args.fiber_diameter:
        config.fiber_diameter_um = args.fiber_diameter
    if args.regime:
        config.regime = args.regime

    config.validate()
    pipeline = FiberAnalysisPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write a smoke test**

```python
# tests/test_config.py
import pytest
from fiber_tracer.config import Config, VoxelSpacing


def test_default_config_validates_with_existing_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    cfg = Config(data_path=str(data_dir), output_dir=str(tmp_path / "out"))
    cfg.validate()


def test_invalid_voxel_spacing_raises():
    cfg = Config()
    cfg.voxel_spacing_um = VoxelSpacing(-1, 1, 1)
    with pytest.raises(ValueError):
        cfg.validate()
```

- [ ] **Step 6: Run test and commit**

```bash
pytest tests/test_config.py -v
git add src/fiber_tracer tests pyproject.toml
git commit -m "feat: add core config, cli skeleton, and smoke tests"
```

---

### Task 1.2: I/O and metadata module

**Files:**
- Create: `src/fiber_tracer/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Implement volume reader**

```python
"""I/O for TIFF stacks, HDF5, and metadata."""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import tifffile
import logging

logger = logging.getLogger(__name__)


def load_tiff_stack(path: Union[str, Path]) -> np.ndarray:
    """Load a TIFF stack from a file or directory of TIFFs."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.tif*"))
        if not files:
            raise FileNotFoundError(f"No TIFF files found in {path}")
        logger.info(f"Loading {len(files)} TIFF slices from {path}")
        return tifffile.imread(files)
    return tifffile.imread(path)


def estimate_volume_fraction(volume: np.ndarray, threshold: float = 0.5) -> float:
    """Quick estimate of foreground volume fraction from normalized volume."""
    return float(np.mean(volume > threshold))


def get_shape_info(volume: np.ndarray, voxel_spacing: Tuple[float, float, float]) -> dict:
    """Return human-readable shape and physical size info."""
    dz, dy, dx = voxel_spacing
    z, y, x = volume.shape
    return {
        "shape_voxels": (z, y, x),
        "shape_um": (z * dz, y * dy, x * dx),
        "voxel_spacing_um": (dz, dy, dx),
        "dtype": str(volume.dtype),
        "size_gb": volume.nbytes / (1024**3),
    }
```

- [ ] **Step 2: Write tests**

```python
# tests/test_io.py
import numpy as np
import tifffile
from fiber_tracer.io import load_tiff_stack, get_shape_info


def test_load_tiff_stack_from_dir(tmp_path):
    volume = np.random.randint(0, 65535, (5, 8, 8), dtype=np.uint16)
    for i, img in enumerate(volume):
        tifffile.imwrite(tmp_path / f"slice_{i:04d}.tif", img)
    loaded = load_tiff_stack(tmp_path)
    assert loaded.shape == volume.shape
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_io.py -v
git add src/fiber_tracer/io.py tests/test_io.py
git commit -m "feat: add tiff stack loader and shape info"
```

---

### Task 1.3: Synthetic phantom generator

**Files:**
- Create: `src/fiber_tracer/validation/phantoms.py`
- Test: `tests/test_phantoms.py`

- [ ] **Step 1: Implement phantom generator**

```python
"""Synthetic fiber phantoms with ground truth."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage


@dataclass
class FiberPhantom:
    volume: np.ndarray
    labels: np.ndarray
    orientations: np.ndarray  # Nx3 unit vectors
    diameters_um: np.ndarray
    lengths_um: np.ndarray
    voxel_spacing_um: Tuple[float, float, float]


def generate_straight_fiber(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    direction: np.ndarray,
    radius_voxels: float,
    intensity: float = 1.0,
) -> np.ndarray:
    """Draw a single straight cylinder in a binary volume."""
    direction = direction / np.linalg.norm(direction)
    z, y, x = np.indices(shape, dtype=float)
    coords = np.stack([z, y, x], axis=-1)
    center_vec = np.array(center)
    to_center = coords - center_vec
    projection = np.dot(to_center, direction)
    perpendicular = to_center - projection[:, :, :, None] * direction
    distance = np.linalg.norm(perpendicular, axis=-1)
    volume = np.zeros(shape, dtype=float)
    volume[distance <= radius_voxels] = intensity
    return volume


def generate_fiber_phantom(
    shape: Tuple[int, int, int] = (64, 64, 64),
    n_fibers: int = 10,
    fiber_diameter_um: float = 4.0,
    voxel_spacing_um: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_std: float = 0.02,
    seed: Optional[int] = None,
) -> FiberPhantom:
    """Generate a phantom with straight, non-touching fibers."""
    rng = np.random.default_rng(seed)
    radius_voxels = 0.5 * fiber_diameter_um / min(voxel_spacing_um)
    volume = np.zeros(shape, dtype=float)
    labels = np.zeros(shape, dtype=np.int32)
    orientations = []
    diameters = []
    lengths = []

    for i in range(n_fibers):
        center = rng.uniform(radius_voxels * 2, np.array(shape) - radius_voxels * 2)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        fiber = generate_straight_fiber(shape, tuple(center), direction, radius_voxels)
        mask = fiber > 0
        if np.any(labels[mask] > 0):
            continue
        labels[mask] = i + 1
        volume += fiber
        orientations.append(direction)
        diameters.append(fiber_diameter_um)
        lengths.append(min(shape) * min(voxel_spacing_um))

    volume = np.clip(volume + rng.normal(0, noise_std, shape), 0, 1)
    return FiberPhantom(
        volume=volume,
        labels=labels,
        orientations=np.array(orientations),
        diameters_um=np.array(diameters),
        lengths_um=np.array(lengths),
        voxel_spacing_um=voxel_spacing_um,
    )
```

- [ ] **Step 2: Write tests**

```python
# tests/test_phantoms.py
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def test_phantom_has_expected_fibers():
    phantom = generate_fiber_phantom(
        shape=(32, 32, 32), n_fibers=3, fiber_diameter_um=2.0, seed=42
    )
    assert phantom.volume.shape == (32, 32, 32)
    assert len(phantom.orientations) >= 1
    assert phantom.volume.max() <= 1.0
```

- [ ] **Step 3: Run tests and commit**

```bash
pytest tests/test_phantoms.py -v
git add src/fiber_tracer/validation tests/test_phantoms.py
git commit -m "feat: add synthetic fiber phantom generator"
```

---

## Phase 2 — Honest Conventional Pipeline (Phase 1 of RAFA)

### Task 2.1: Preprocessing module

**Files:**
- Create: `src/fiber_tracer/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Implement 3D-aware preprocessing**

```python
"""Preprocessing: denoising, normalization, anisotropy handling."""

from typing import Optional, Tuple
import numpy as np
from scipy import ndimage
from fiber_tracer.config import VoxelSpacing


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = volume.min(), volume.max()
    if vmax == vmin:
        return np.zeros_like(volume, dtype=float)
    return ((volume - vmin) / (vmax - vmin)).astype(np.float32)


def gaussian_denoise(volume: np.ndarray, sigma_um: float, voxel_spacing: VoxelSpacing) -> np.ndarray:
    """3D Gaussian denoising with physical sigma in micrometers."""
    sigma_voxels = (
        sigma_um / voxel_spacing.z,
        sigma_um / voxel_spacing.y,
        sigma_um / voxel_spacing.x,
    )
    return ndimage.gaussian_filter(volume, sigma=sigma_voxels)


def resample_to_isotropic(
    volume: np.ndarray,
    voxel_spacing: VoxelSpacing,
    order: int = 1,
) -> Tuple[np.ndarray, VoxelSpacing]:
    """Resample anisotropic volume to isotropic voxels at the smallest spacing."""
    target = min(voxel_spacing.z, voxel_spacing.y, voxel_spacing.x)
    zoom = (
        voxel_spacing.z / target,
        voxel_spacing.y / target,
        voxel_spacing.x / target,
    )
    resampled = ndimage.zoom(volume, zoom, order=order)
    return resampled, VoxelSpacing(target, target, target)
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_preprocess.py -v
git add src/fiber_tracer/preprocess.py tests/test_preprocess.py
git commit -m "feat: add 3D preprocessing with physical units"
```

---

### Task 2.2: 3D segmentation module

**Files:**
- Create: `src/fiber_tracer/segmentation/classical.py`
- Test: `tests/test_segmentation_classical.py`

- [ ] **Step 1: Implement Otsu + 3D watershed**

```python
"""Classical segmentation: thresholding and watershed separation."""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure


def segment_otsu_3d(volume: np.ndarray) -> np.ndarray:
    """Global 3D Otsu thresholding."""
    threshold = filters.threshold_otsu(volume)
    return volume > threshold


def segment_watershed_3d(
    volume: np.ndarray,
    foreground_mask: np.ndarray,
    min_distance_voxels: int = 3,
) -> np.ndarray:
    """3D marker-controlled watershed on distance transform."""
    distance = ndimage.distance_transform_edt(foreground_mask)
    local_max = morphology.local_maxima(distance, min_distance=min_distance_voxels)
    markers = ndimage.label(local_max)[0]
    labels = morphology.watershed(-distance, markers, mask=foreground_mask)
    return labels


def remove_small_objects(labels: np.ndarray, min_size_voxels: int) -> np.ndarray:
    """Remove connected components below minimum size."""
    return morphology.remove_small_objects(labels, min_size=min_size_voxels)
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_segmentation_classical.py -v
git add src/fiber_tracer/segmentation/classical.py tests/test_segmentation_classical.py
git commit -m "feat: add 3D Otsu and watershed segmentation"
```

---

### Task 2.3: Centerline extraction using skan/scikit-image

**Files:**
- Create: `src/fiber_tracer/centerline/skeleton.py`, `src/fiber_tracer/centerline/graph.py`
- Test: `tests/test_centerline.py`

- [ ] **Step 1: Implement skeletonization wrapper**

```python
"""Skeletonization and graph analysis using scikit-image and skan."""

from typing import List
import numpy as np
from skimage.morphology import skeletonize_3d


def skeletonize_label_volume(labels: np.ndarray) -> np.ndarray:
    """Skeletonize each labeled fiber separately to avoid bridging."""
    skeleton = np.zeros_like(labels, dtype=bool)
    for label in np.unique(labels)[1:]:
        mask = labels == label
        skel = skeletonize_3d(mask)
        skeleton |= skel
    return skeleton


def try_skeletonize_with_skan(skeleton: np.ndarray):
    """Return skan Skeleton object for graph analysis."""
    try:
        from skan import Skeleton
        return Skeleton(skeleton)
    except ImportError as e:
        raise BackendNotAvailableError("Install skeleton extra: pip install fiber-tracer[skeleton]") from e
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_centerline.py -v
git add src/fiber_tracer/centerline tests/test_centerline.py
git commit -m "feat: add skan/scikit-image skeletonization"
```

---

### Task 2.4: Orientation module using structure-tensor package

**Files:**
- Create: `src/fiber_tracer/orientation/structure_tensor.py`, `src/fiber_tracer/orientation/pca.py`
- Test: `tests/test_orientation.py`

- [ ] **Step 1: Implement structure-tensor orientation**

```python
"""Orientation estimation using the structure-tensor package."""

import numpy as np
from fiber_tracer.exceptions import BackendNotAvailableError


def compute_structure_tensor_field(
    volume: np.ndarray,
    sigma: float,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of the 3D structure tensor."""
    try:
        from structure_tensor import structure_tensor_3d
    except ImportError as e:
        raise BackendNotAvailableError("Install structure extra: pip install fiber-tracer[structure]") from e

    eigenvalues, eigenvectors = structure_tensor_3d(volume, sigma, rho, truncate=4.0)
    return eigenvalues, eigenvectors


def orientation_from_smallest_eigenvector(eigenvectors: np.ndarray) -> np.ndarray:
    """Eigenvector of smallest eigenvalue points along the fiber."""
    # eigenvectors shape: (3, 3, D, H, W) where first dim is eigenvalue index (0=smallest)
    direction = eigenvectors[0]  # (3, D, H, W)
    return direction
```

- [ ] **Step 2: Implement per-fiber PCA orientation (fallback)**

```python
"""PCA-based orientation from voxel coordinates."""

import numpy as np
from scipy import linalg


def pca_orientation(coords: np.ndarray) -> np.ndarray:
    """Return principal axis from voxel coordinates."""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    return axis / np.linalg.norm(axis)
```

- [ ] **Step 3: Tests and commit**

```bash
pytest tests/test_orientation.py -v
git add src/fiber_tracer/orientation tests/test_orientation.py
git commit -m "feat: add structure tensor and PCA orientation backends"
```

---

### Task 2.5: Morphometry from centerlines

**Files:**
- Create: `src/fiber_tracer/analysis/morphometry.py`
- Test: `tests/test_morphometry.py`

- [ ] **Step 1: Implement length, diameter, tortuosity**

```python
"""Fiber morphometry computed from ordered centerlines."""

import numpy as np
from scipy import ndimage


def ordered_path_length(path: np.ndarray, voxel_spacing: tuple) -> float:
    """Sum Euclidean distances along an ordered path in physical units."""
    scaled = path * np.array(voxel_spacing)
    return float(np.sum(np.linalg.norm(np.diff(scaled, axis=0), axis=1)))


def tortuosity(path: np.ndarray, voxel_spacing: tuple) -> float:
    """Arc length divided by endpoint Euclidean distance."""
    if len(path) < 2:
        return 1.0
    arc_length = ordered_path_length(path, voxel_spacing)
    scaled = path * np.array(voxel_spacing)
    chord = np.linalg.norm(scaled[-1] - scaled[0])
    if chord == 0:
        return 1.0
    return arc_length / chord


def equivalent_diameter_from_volume(n_voxels: int, voxel_spacing: tuple) -> float:
    """Diameter of a sphere with equivalent volume."""
    volume_um3 = n_voxels * np.prod(voxel_spacing)
    return 2.0 * (3.0 * volume_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_morphometry.py -v
git add src/fiber_tracer/analysis/morphometry.py tests/test_morphometry.py
git commit -m "feat: add centerline-based morphometry"
```

---

### Task 2.6: Regime selector

**Files:**
- Create: `src/fiber_tracer/regime.py`
- Test: `tests/test_regime.py`

- [ ] **Step 1: Implement regime detection**

```python
"""Select analysis regime from physical voxel/fiber ratio."""

from fiber_tracer.config import Config


def detect_regime(config: Config) -> str:
    """Return one of: resolved, marginal, subvoxel."""
    spacing = min(config.voxel_spacing_um.z, config.voxel_spacing_um.y, config.voxel_spacing_um.x)
    ratio = spacing / config.fiber_diameter_um
    if ratio <= 0.3:
        return "resolved"
    elif ratio <= 3.0:
        return "marginal"
    return "subvoxel"
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_regime.py -v
git add src/fiber_tracer/regime.py tests/test_regime.py
git commit -m "feat: add regime selector from voxel/fiber ratio"
```

---

### Task 2.7: Pipeline orchestrator

**Files:**
- Create: `src/fiber_tracer/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Implement basic orchestrator**

```python
"""Pipeline orchestrator for RAFA."""

import logging
from pathlib import Path
import numpy as np

from fiber_tracer.config import Config
from fiber_tracer.io import load_tiff_stack, get_shape_info
from fiber_tracer.preprocess import normalize_intensity, gaussian_denoise
from fiber_tracer.regime import detect_regime
from fiber_tracer.segmentation.classical import segment_otsu_3d, segment_watershed_3d
from fiber_tracer.centerline.skeleton import skeletonize_label_volume

logger = logging.getLogger(__name__)


class FiberAnalysisPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.volume: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def run(self) -> dict:
        self.config.validate()
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        raw = load_tiff_stack(self.config.data_path)
        logger.info(get_shape_info(raw, self.config.voxel_spacing_um))

        volume = normalize_intensity(raw)
        if self.config.processing.denoise_sigma:
            volume = gaussian_denoise(volume, self.config.processing.denoise_sigma, self.config.voxel_spacing_um)

        regime = self.config.regime if self.config.regime != "auto" else detect_regime(self.config)
        logger.info(f"Selected regime: {regime}")

        if regime == "resolved":
            mask = segment_otsu_3d(volume)
            labels = segment_watershed_3d(volume, mask)
            skeleton = skeletonize_label_volume(labels)
            # TODO: extract per-fiber properties in Task 2.8
        elif regime == "marginal":
            # TODO: structure tensor field in Phase 2
            raise NotImplementedError("Marginal regime in Phase 2")
        else:
            # TODO: subvoxel orientation tensor in Phase 2
            raise NotImplementedError("Subvoxel regime in Phase 2")

        return {"regime": regime, "n_labels": int(np.max(labels))}
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_pipeline.py -v
git add src/fiber_tracer/pipeline.py tests/test_pipeline.py
git commit -m "feat: add resolved-regime pipeline orchestrator"
```

---

## Phase 3 — Regime-Aware Engine

### Task 3.1: Orientation tensor (Advani–Tucker A₂)

**Files:**
- Create: `src/fiber_tracer/orientation/tensor.py`
- Test: `tests/test_orientation_tensor.py`

- [ ] **Step 1: Implement A₂ computation**

```python
"""Advani-Tucker second-order orientation tensor."""

import numpy as np


def direction_tensor(directions: np.ndarray) -> np.ndarray:
    """Compute A2 = <p p^T> for an array of unit directions."""
    directions = np.atleast_2d(directions)
    return np.mean(np.einsum('bi,bj->bij', directions, directions), axis=0)


def fractional_anisotropy(tensor: np.ndarray) -> float:
    """Scalar anisotropy measure from A2 eigenvalues."""
    evals = np.linalg.eigvalsh(tensor)
    mean = evals.mean()
    if mean == 0:
        return 0.0
    return np.sqrt(1.5 * np.sum((evals - mean) ** 2) / np.sum(evals ** 2))
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_orientation_tensor.py -v
git add src/fiber_tracer/orientation/tensor.py tests/test_orientation_tensor.py
git commit -m "feat: add Advani-Tucker orientation tensor and anisotropy"
```

---

### Task 3.2: Marginal and subvoxel pipelines

**Files:**
- Modify: `src/fiber_tracer/pipeline.py`
- Test: `tests/test_pipeline_marginal.py`, `tests/test_pipeline_subvoxel.py`

- [ ] **Step 1: Implement marginal regime**

Use structure-tensor field on normalized volume, compute local orientation per voxel, aggregate into windows, output A₂ map.

- [ ] **Step 2: Implement subvoxel regime**

Same as marginal but with larger integration window; output global A₂ and orientation distribution.

- [ ] **Step 3: Tests and commit**

```bash
pytest tests/test_pipeline_marginal.py tests/test_pipeline_subvoxel.py -v
git add src/fiber_tracer/pipeline.py tests/
git commit -m "feat: add marginal and subvoxel regime pipelines"
```

---

## Phase 4 — Optional Backends

### Task 4.1: ML segmentation backend (lazy import)

**Files:**
- Create: `src/fiber_tracer/backends/__init__.py`, `src/fiber_tracer/backends/base.py`, `src/fiber_tracer/backends/ml_segmentation.py`
- Test: `tests/test_ml_backend.py`

- [ ] **Step 1: Define backend ABC and optional torch wrapper**

```python
"""Optional ML segmentation backend."""

from fiber_tracer.backends.base import SegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError


class MLSegmentationBackend(SegmentationBackend):
    def __init__(self, model_path: str | None = None):
        try:
            import torch
        except ImportError as e:
            raise BackendNotAvailableError("Install ml extra: pip install fiber-tracer[ml]") from e
        self.torch = torch
        self.model_path = model_path

    def segment(self, volume):
        raise NotImplementedError("Train or load a model before segmenting")
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_ml_backend.py -v
git add src/fiber_tracer/backends tests/test_ml_backend.py
git commit -m "feat: add optional ML segmentation backend with lazy imports"
```

---

### Task 4.2: TDA descriptor backend (lazy import)

**Files:**
- Create: `src/fiber_tracer/backends/tda_gudhi.py`
- Test: `tests/test_tda_backend.py`

- [ ] **Step 1: Implement Betti numbers and persistence summaries**

```python
"""Optional TDA descriptors using gudhi."""

from fiber_tracer.exceptions import BackendNotAvailableError


def betti_numbers(binary_volume: bytes):
    try:
        import gudhi
    except ImportError as e:
        raise BackendNotAvailableError("Install tda extra: pip install fiber-tracer[tda]") from e
    cc = gudhi.CubicalComplex(dimensions=binary_volume.shape, top_dimensional_cells=1 - binary_volume.flatten())
    persistence = cc.persistence()
    return persistence
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_tda_backend.py -v
git add src/fiber_tracer/backends/tda_*.py tests/test_tda_backend.py
git commit -m "feat: add optional TDA descriptor backends"
```

---

## Phase 5 — Validation & Benchmarking

### Task 5.1: Metrics module

**Files:**
- Create: `src/fiber_tracer/validation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Implement validation metrics**

```python
"""Validation metrics against ground truth."""

import numpy as np
from scipy.spatial.distance import cdist


def angular_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Smallest angle between two directions in degrees."""
    pred = pred / np.linalg.norm(pred)
    true = true / np.linalg.norm(true)
    dot = np.clip(np.abs(np.dot(pred, true)), 0, 1)
    return float(np.degrees(np.arccos(dot)))


def orientation_tensor_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Frobenius norm of A2 difference."""
    return float(np.linalg.norm(pred - true, ord='fro'))


def dice_score(pred: np.ndarray, true: np.ndarray) -> float:
    intersection = np.sum(pred & true)
    return 2.0 * intersection / (np.sum(pred) + np.sum(true))
```

- [ ] **Step 2: Tests and commit**

```bash
pytest tests/test_metrics.py -v
git add src/fiber_tracer/validation/metrics.py tests/test_metrics.py
git commit -m "feat: add validation metrics"
```

---

### Task 5.2: Benchmark harness

**Files:**
- Create: `src/fiber_tracer/validation/benchmark.py`
- Create: `scripts/benchmark_phantoms.py`

- [ ] **Step 1: Implement benchmark script**

```python
# scripts/benchmark_phantoms.py
"""Run deterministic benchmarks on synthetic phantoms."""

from fiber_tracer.validation.phantoms import generate_fiber_phantom
from fiber_tracer.validation.metrics import angular_error


def main():
    phantom = generate_fiber_phantom(shape=(64, 64, 64), n_fibers=5, fiber_diameter_um=4.0, seed=42)
    print(f"Generated phantom with {len(phantom.orientations)} fibers")
    # TODO: run pipeline and report metrics


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add src/fiber_tracer/validation/benchmark.py scripts/benchmark_phantoms.py
git commit -m "feat: add phantom benchmark harness"
```

---

### Task 5.3: Public dataset validation harness

**Files:**
- Create: `scripts/download_gfpa66.py`
- Create: `scripts/validate_gfpa66.py`

- [ ] **Step 1: Document dataset provenance and download helper**

```python
# scripts/download_gfpa66.py
"""Download GF-PA66 validation dataset with attribution.

License: CC BY-SA 4.0
DOI: 10.5281/zenodo.4587827
Citation: Bertoldo et al., Front. Mater. 2021.
"""

ZENODO_URL = "https://zenodo.org/records/4587827/files/GF-PA66_3D_XCT.h5"


def main():
    print("Please download GF-PA66_3D_XCT.h5 from")
    print("https://zenodo.org/records/4587827")
    print("and cite: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/download_gfpa66.py scripts/validate_gfpa66.py
git commit -m "docs: add GF-PA66 validation harness with license attribution"
```

---

## Phase 6 — Reporting, Documentation & CI

### Task 6.1: Honest reporting module

**Files:**
- Create: `src/fiber_tracer/reporting/csv.py`, `src/fiber_tracer/reporting/html.py`, `src/fiber_tracer/reporting/json.py`

- [ ] **Step 1: Implement CSV/JSON/HTML exporters**

Write exporters that include:
- Config used
- Regime selected
- Per-fiber or per-window properties
- Uncertainty/regime caveats
- Citations

- [ ] **Step 2: Commit**

```bash
git add src/fiber_tracer/reporting
git commit -m "feat: add honest CSV/JSON/HTML reporting with caveats"
```

---

### Task 6.2: Rewrite documentation

**Files:**
- Create: `docs/methodology.md`, `docs/validation_protocol.md`, `docs/parameter_guide.md`
- Rewrite: `README.md`, `CHANGELOG.md`

- [ ] **Step 1: Write honest methodology doc**

Explain:
- Regime-aware design
- Structure tensor math with correct citations
- Why RK4/PH are not used for segmentation
- Limitations and caveats

- [ ] **Step 2: Rewrite README**

Remove false claims. Include:
- What the tool does
- Installation instructions
- Quick start with phantom
- License and citations

- [ ] **Step 3: Commit**

```bash
git add docs README.md CHANGELOG.md
git commit -m "docs: rewrite methodology and README with honest claims"
```

---

### Task 6.3: CI/CD

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[structure,skeleton,dev]"
      - name: Test
        run: pytest tests/ -v
      - name: License check
        run: |
          pip install pip-licenses
          pip-licenses --format=json --output-file=licenses.json
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add cross-platform pytest workflow and license check"
```

---

## 7. Final Acceptance Criteria

Before declaring RAFA complete, verify:

1. `pytest tests/` passes on Linux, macOS, Windows.
2. `pip install -e .` installs core package with no GPL dependencies.
3. `pip install -e ".[all]"` installs optional backends without breaking core.
4. Synthetic phantom benchmark reports angular error < 5° and Dice > 0.85 on resolved fibers.
5. `THIRD_PARTY_LICENSES.md` and `CITATIONS.md` are complete.
6. README contains no false claims.
7. `fiber-tracer --help` works.
8. License check passes (no GPL/AGPL/SSPL in default install).

---

## 8. Open Questions Flagged for User

1. **Model training data:** Do you have permission/labels to train on GF-PA66, or should we generate synthetic training data only?
2. **Public dataset access:** Should the download scripts attempt automatic Zenodo download, or only document links?
3. **Minimum Python:** Is Python 3.10 acceptable, or do you need 3.9 support? (3.10 strongly recommended for modern scikit-image.)
4. **GUI:** Is a napari viewer plugin desired, or CLI-only for Phase 1?
5. **Performance target:** What is the largest volume you need to process in one run? (Affects zarr/dask priorities.)

---

## 9. Recommended First Execution

Start with **Phase 0** and **Phase 1** only. Those deliver a working, honest, conventional pipeline with tests. Then iterate through Phases 2–6. Do not attempt all phases in one pass.