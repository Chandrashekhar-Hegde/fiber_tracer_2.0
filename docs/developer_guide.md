# Developer Guide

This guide is for contributors working on `fiber-tracer`. It covers setup, testing, code quality, and common extension tasks.

## Getting started

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/llMr-Sweetll/fiber_tracer_2.0.git
cd fiber_tracer_2.0
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

For the recommended set of optional backends used by the test suite and documentation examples:

```bash
pip install -e ".[structure,skeleton,dev]"
```

## Running tests

Run the full test suite:

```bash
pytest tests/ -q
```

Run a single test module:

```bash
pytest tests/test_config.py -q
```

Run with coverage:

```bash
pytest tests/ --cov=fiber_tracer --cov-report=term-missing
```

## Linting and formatting

The project uses `ruff` for linting, `black` for formatting, and `mypy` for type checking.

```bash
ruff check .
black .
mypy src/fiber_tracer
```

Check formatting without modifying files:

```bash
black --check .
```

Configuration for these tools is in `pyproject.toml`.

## Adding a new backend

Backends are adapters for optional heavy dependencies. The core package must always work without them.

### Example: adding a new segmentation backend

1. **Create the adapter module** at `src/fiber_tracer/backends/my_segmentation.py`:

```python
"""Optional my-segmentation backend."""

from typing import Optional

import numpy as np

from fiber_tracer.backends.base import SegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError


class MySegmentationBackend(SegmentationBackend):
    """Segmentation backend that lazy-imports my_lib."""

    def __init__(self, model_path: Optional[str] = None):
        try:
            import my_lib
        except ImportError as exc:
            raise BackendNotAvailableError(
                "Install my-lib extra: pip install fiber-tracer[my-lib]"
            ) from exc
        self.my_lib = my_lib
        self.model_path = model_path
        self.model = None

    def segment(self, volume: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise NotImplementedError(
                "No model is loaded. Load a checkpoint before calling segment()."
            )
        return np.asarray(self.model(volume))
```

2. **Register it** in `src/fiber_tracer/backends/__init__.py`.
3. **Wire the config flag** in `fiber_tracer/pipeline.py`, for example by checking `self.config.segmentation.method == "my_lib"`.
4. **Add tests** in `tests/test_my_segmentation_backend.py`:

```python
import pytest

from fiber_tracer.backends.my_segmentation import MySegmentationBackend
from fiber_tracer.exceptions import BackendNotAvailableError


def test_backend_raises_when_dependency_missing(monkeypatch):
    monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: None)
    with pytest.raises(BackendNotAvailableError):
        MySegmentationBackend()
```

Always lazy-import the third-party library and raise `BackendNotAvailableError` with a clear install hint when it is missing.

## Adding a new visualization

1. Add the helper to `src/fiber_tracer/viz/napari_viewer.py` or `src/fiber_tracer/viz/plotly_plots.py`. Lazy-import `napari` or `plotly` and raise `BackendNotAvailableError` if the `viz` extra is not installed.
2. Export the helper from `src/fiber_tracer/viz/__init__.py`.
3. Add a CLI subcommand or flag in `fiber_tracer/cli.py` if users should access it from the command line.
4. Add tests in `tests/test_napari_viewer.py` or `tests/test_plotly_plots.py`. Use `monkeypatch` to avoid launching a GUI or writing large HTML files in unit tests.

## Adding a new regime

1. Add the regime identifier to `VALID_REGIMES` in `fiber_tracer/config.py` if it is not already covered by the existing identifiers.
2. Update `detect_regime` in `fiber_tracer/regime.py` with the new threshold logic.
3. Add a `_run_<regime>` method to `FiberAnalysisPipeline` in `fiber_tracer/pipeline.py`.
4. Add a caveat string for the new regime in `fiber_tracer/reporting/citations.py`.
5. Add tests:
   - Regime selection in `tests/test_regime.py`.
   - Pipeline behavior in `tests/test_pipeline_<regime>.py`.
   - End-to-end CLI behavior in `tests/test_cli.py` if the regime changes CLI output.

## Commit and review workflow

1. Run the test suite and lint checks before committing:

```bash
pytest tests/ -q
ruff check .
black --check .
mypy src/fiber_tracer
```

2. Use clear, concise commit messages in the present tense:

```
feat: add gudhi-based persistence summary backend
test: cover BackendNotAvailableError for missing torch
docs: update parameter guide with new CLI flags
```

3. Keep changes focused. If a pull request adds a backend, it should not also refactor unrelated modules.
4. Update documentation when behavior or configuration changes.

## License and citations

`fiber-tracer` is released under the MIT License. When adding a new dependency:

1. Add it to `pyproject.toml` under the correct optional-dependency group.
2. Record its license and URL in `THIRD_PARTY_LICENSES.md`.
3. Add any required academic citations to `CITATIONS.md` and `fiber_tracer/reporting/citations.py`.
4. Do not import GPL/AGPL/SSPL code into the core package. Optional adapters may depend on copyleft libraries only if they are isolated plugins and never required by default.

When adding validation datasets, document the license, DOI, and citation in `docs/validation_protocol.md`, `CITATIONS.md`, and `THIRD_PARTY_LICENSES.md`.
