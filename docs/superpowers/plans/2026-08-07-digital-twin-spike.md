# Digital Twin Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal PoC described in `docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md`: fit synthetic-phantom parameters from a resolved-regime pipeline summary, regenerate a statistically-matched "twin" volume, and compute an effective modulus via Halpin-Tsai — validated by a round-trip check against a known-parameter phantom.

**Architecture:** One script, `scripts/digital_twin_poc.py`, no pipeline/config/CLI/TUI wiring (spike, per the DVC/DIC PoC precedent).

**Tech Stack:** Python, NumPy. No new dependency — reuses `fiber_tracer.validation.phantoms`, `fiber_tracer.orientation.tensor`, `fiber_tracer.pipeline.FiberAnalysisPipeline`.

## Global Constraints

- No pipeline/config/CLI/TUI changes — standalone script only.
- Run `black --check`, `ruff check`, `mypy` before each commit (established convention from the DVC/DIC work).
- The round-trip validation is checking the *fitting logic*, not the *twin concept* — the spec is explicit that this is a narrower claim; don't overstate what the self-check proves in the script's docstring/output.

---

### Task 1: Fit twin parameters from a resolved-regime summary

**Files:**
- Create: `scripts/digital_twin_poc.py`

- [ ] **Step 1: Module docstring and imports**

```python
"""Proof-of-concept: fit synthetic-phantom parameters from a resolved-regime
pipeline summary (standing in for "a real scan" -- this repo has no real
experimental scan to fit against, so the round-trip below validates the
fitting logic against a known-parameter phantom, not the twin concept
itself against real mechanical behavior).

See docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md.

# ponytail: spike script, not part of the installed package; no
# pipeline/config/CLI/TUI wiring.
"""

from __future__ import annotations

import numpy as np

from fiber_tracer.orientation.tensor import aggregate_direction_tensor, fractional_anisotropy
from fiber_tracer.validation.phantoms import FiberPhantom, generate_fiber_phantom

ORIENTATION_MODES = ("aligned", "in_plane", "orthogonal", "woven", "twill", "random")
```

- [ ] **Step 2: `fit_twin_parameters`**

```python
def fit_twin_parameters(summary: dict) -> dict:
    """Fit phantom-generator parameters from a resolved-regime summary.

    Returns a dict with keys: n_fibers, fiber_diameter_um, orientation_mode,
    fractional_anisotropy (diagnostic, not a generate_fiber_phantom input).
    """
    fibers = summary["fibers"]
    diameters = [f["equivalent_diameter_um"] for f in fibers if "equivalent_diameter_um" in f]
    mean_diameter_um = float(np.mean(diameters)) if diameters else 0.0

    orientations = np.array([f["orientation"] for f in fibers if "orientation" in f])
    if len(orientations) > 0:
        tensor = aggregate_direction_tensor(orientations)
        fa = fractional_anisotropy(tensor)
        principal_axis = int(np.argmax(np.linalg.eigvalsh(tensor)))
    else:
        fa = 0.0
        principal_axis = 2

    # Coarse heuristic mapping fractional anisotropy -> the phantom
    # generator's categorical orientation_mode: high FA means one dominant
    # axis (aligned); low FA means near-isotropic (random). This is exactly
    # the "closest existing mode" mapping the spec flags as a limitation,
    # not a continuous-distribution fit.
    if fa > 0.6:
        orientation_mode = "aligned"
    elif fa < 0.2:
        orientation_mode = "random"
    else:
        orientation_mode = "in_plane"

    return {
        "n_fibers": summary["n_labels"],
        "fiber_diameter_um": mean_diameter_um,
        "orientation_mode": orientation_mode,
        "fractional_anisotropy": float(fa),
        "principal_axis": principal_axis,
    }
```

- [ ] **Step 3: Run manually against a hand-built summary to sanity-check**

```bash
source .venv/bin/activate
python3 -c "
from scripts.digital_twin_poc import fit_twin_parameters
summary = {
    'n_labels': 3,
    'fibers': [
        {'equivalent_diameter_um': 4.0, 'orientation': [1.0, 0.0, 0.0]},
        {'equivalent_diameter_um': 4.2, 'orientation': [0.98, 0.1, 0.0]},
        {'equivalent_diameter_um': 3.9, 'orientation': [0.99, 0.05, 0.02]},
    ],
}
print(fit_twin_parameters(summary))
"
```

Expected: `fiber_diameter_um` near 4.0, `orientation_mode` 'aligned' (all three vectors point roughly the same direction, high FA).

- [ ] **Step 4:** `black --check scripts/digital_twin_poc.py && ruff check scripts/digital_twin_poc.py && mypy scripts/digital_twin_poc.py`, then commit:

```bash
git add scripts/digital_twin_poc.py
git commit -m "Add digital twin PoC: fit_twin_parameters from a resolved-regime summary"
```

---

### Task 2: Regenerate the twin and compute effective modulus

**Files:**
- Modify: `scripts/digital_twin_poc.py`

- [ ] **Step 1: `regenerate_twin`**

```python
def regenerate_twin(
    fitted_params: dict, shape: tuple[int, int, int], voxel_spacing_um: tuple[float, float, float]
) -> FiberPhantom:
    return generate_fiber_phantom(
        shape=shape,
        n_fibers=fitted_params["n_fibers"],
        fiber_diameter_um=fitted_params["fiber_diameter_um"],
        voxel_spacing_um=voxel_spacing_um,
        orientation_mode=fitted_params["orientation_mode"],
        seed=0,
    )
```

- [ ] **Step 2: `effective_modulus_halpin_tsai`**

Halpin-Tsai (RESEARCH_FOUNDATION.md ref 67): `E = E_m * (1 + xi*eta*Vf) / (1 - eta*Vf)`, `eta = (Ef/Em - 1) / (Ef/Em + xi)`, where `xi` depends on reinforcement geometry (2*aspect_ratio for longitudinal fiber loading is a standard choice).

```python
def effective_modulus_halpin_tsai(
    volume_fraction: float,
    fiber_modulus_gpa: float,
    matrix_modulus_gpa: float,
    aspect_ratio: float,
) -> float:
    """Halpin-Tsai longitudinal effective modulus (RESEARCH_FOUNDATION.md ref 67)."""
    xi = 2.0 * aspect_ratio
    ratio = fiber_modulus_gpa / matrix_modulus_gpa
    eta = (ratio - 1.0) / (ratio + xi)
    return matrix_modulus_gpa * (1.0 + xi * eta * volume_fraction) / (1.0 - eta * volume_fraction)
```

- [ ] **Step 3:** `black --check && ruff check && mypy`, then commit:

```bash
git add scripts/digital_twin_poc.py
git commit -m "Add twin regeneration and Halpin-Tsai effective modulus to the PoC"
```

---

### Task 3: End-to-end round-trip validation and `__main__`

**Files:**
- Modify: `scripts/digital_twin_poc.py`

**Interfaces:**
- Consumes: `fit_twin_parameters`, `regenerate_twin`, `effective_modulus_halpin_tsai` from Tasks 1-2.

- [ ] **Step 1: `main`**

```python
def main() -> None:
    import tempfile
    from pathlib import Path

    from fiber_tracer.config import Config, VoxelSpacing
    from fiber_tracer.io import save_tiff_stack
    from fiber_tracer.pipeline import FiberAnalysisPipeline

    shape = (64, 64, 64)
    voxel_spacing_um = (1.0, 1.0, 1.0)
    known_n_fibers = 30
    known_diameter_um = 4.0
    known_mode = "aligned"

    known_phantom = generate_fiber_phantom(
        shape=shape,
        n_fibers=known_n_fibers,
        fiber_diameter_um=known_diameter_um,
        voxel_spacing_um=voxel_spacing_um,
        orientation_mode=known_mode,
        seed=1,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "scan.tif"
        save_tiff_stack(data_path, known_phantom.volume)

        config = Config(
            data_path=str(data_path),
            output_dir=str(tmp_path / "out"),
            voxel_spacing_um=VoxelSpacing(*voxel_spacing_um),
            fiber_diameter_um=known_diameter_um,
            regime="resolved",
        )
        summary = FiberAnalysisPipeline(config).run()

    fitted = fit_twin_parameters(summary)
    print(f"known:  n_fibers={known_n_fibers} diameter_um={known_diameter_um} mode={known_mode}")
    print(
        f"fitted: n_fibers={fitted['n_fibers']} diameter_um={fitted['fiber_diameter_um']:.2f} "
        f"mode={fitted['orientation_mode']} (FA={fitted['fractional_anisotropy']:.2f})"
    )

    twin = regenerate_twin(fitted, shape, voxel_spacing_um)
    print(f"twin volume shape: {twin.volume.shape}, twin fiber count: {len(twin.orientations)}")

    volume_fraction = fitted["n_fibers"] * np.pi * (fitted["fiber_diameter_um"] / 2) ** 2 * shape[0]
    volume_fraction /= np.prod(shape) * np.prod(voxel_spacing_um)
    volume_fraction = min(volume_fraction, 0.65)  # physically-plausible packing cap

    modulus = effective_modulus_halpin_tsai(
        volume_fraction, fiber_modulus_gpa=72.0, matrix_modulus_gpa=3.0, aspect_ratio=20.0
    )
    print(f"estimated volume fraction: {volume_fraction:.3f}")
    print(f"Halpin-Tsai effective modulus: {modulus:.2f} GPa")

    # ponytail: generous PoC tolerances -- validates the fitting logic
    # round-trips, not that the twin is mechanically representative of a
    # real specimen (no ground truth for that exists in this repo; see the
    # spec's Validation section).
    diameter_error = abs(fitted["fiber_diameter_um"] - known_diameter_um)
    assert diameter_error < 0.5, f"fitted diameter off by {diameter_error:.2f} um"
    assert fitted["n_fibers"] == known_n_fibers, "fitted fiber count does not match known count"
    assert fitted["orientation_mode"] == known_mode, (
        f"fitted mode {fitted['orientation_mode']!r} != known mode {known_mode!r}"
    )
    print("PoC self-check passed: fitted parameters round-trip within tolerance.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end to end**

```bash
source .venv/bin/activate
python3 scripts/digital_twin_poc.py
```

Expected: prints known vs. fitted parameters, twin volume info, estimated volume fraction and effective modulus, then either the pass message or an `AssertionError`. If `n_fibers` doesn't match exactly (e.g. some known fibers overlap and get skipped during rasterization, a known behavior of `generate_fiber_phantom`), loosen that assertion to a tolerance and note why in a comment rather than silently deleting the check.

- [ ] **Step 3:** `black --check && ruff check && mypy`, then commit:

```bash
git add scripts/digital_twin_poc.py
git commit -m "Wire up digital twin PoC end-to-end round-trip validation"
```

---

## Self-Review

**Spec coverage:** `fit_twin_parameters` (Task 1) → spec PoC step 1. `regenerate_twin` + `effective_modulus_halpin_tsai` (Task 2) → spec PoC steps 2-3. `main` end-to-end round-trip (Task 3) → spec PoC step 4 and Validation section.

**Placeholder scan:** no TBD; the orientation-mode heuristic (FA thresholds) is explicitly flagged as a coarse mapping, not a placeholder — matches the spec's stated limitation.

**Consistency:** `fit_twin_parameters`'s return dict keys (`n_fibers`, `fiber_diameter_um`, `orientation_mode`, `fractional_anisotropy`, `principal_axis`) are used identically by `regenerate_twin` and `main`.
