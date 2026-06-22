# Phase 4 Backends Design — RAFA Optional Backends

> **Goal:** Make the RAFA Phase 4 optional backends real, tested, and integrated:
> 1. **ML segmentation backend** — a lightweight 3D U-Net trained on synthetic fiber phantoms so that `segmentation.method = "unet"` produces a usable binary mask.
> 2. **TDA descriptor backend** — robust Betti-number and persistence-summary computation on binary fiber masks using a distance-transform filtration.

---

## Background from research

- **Bertoldo et al. (2021)** used a modular 3D U-Net (3×3×3 convolutions, 32×32×32 crops, batch norm, dropout, AdaBelief) to segment GF-PA66 glass-fiber XCT data. Their trained models and data are public (CC BY-SA).
- **Friemann et al. (2025)** showed that a segmentation model trained **only on synthetic, automatically labelled XCT data** can segment real 3D-woven carbon-fiber composites with ~88% pixel agreement. This removes the need for manual labels and makes a self-contained training pipeline feasible.
- **GUDHI** `CubicalComplex` supports image filtrations. A **distance-transform filtration** (birth = 0 at boundary, death = EDT value) is more informative for binary masks than a two-level 0/∞ filtration because it captures feature size and shape.

---

## 1. TDA backend improvements

### Current state
`src/fiber_tracer/backends/tda_gudhi.py` already exposes:
- `betti_numbers(binary_volume)`
- `persistence_summary(binary_volume)`

It uses a binary 0/∞ filtration, which mostly yields essential components and little shape information.

### Proposed changes
1. Add `persistence_diagram(binary_volume, filtration="distance_transform")` that:
   - Computes the Euclidean distance transform (EDT) inside the foreground.
   - Builds a `gudhi.CubicalComplex` with the distance values as top-dimensional-cell filtrations.
   - Returns a list of `(dimension, birth, death)` triples.
2. Keep `betti_numbers(binary_volume)` but compute it from the persistence diagram (count essential classes, i.e. death = ∞, per dimension).
3. Keep `persistence_summary(binary_volume)` but base it on the distance-transform persistence diagram:
   - finite lifetimes summarize feature sizes,
   - essential features give the count of connected components / voids.
4. Add unit tests with analytically known shapes:
   - single solid cube → b0=1, b1=0, b2=0,
   - two separated cubes → b0=2,
   - hollow sphere / torus-like shape if feasible.
5. Lazy-import `gudhi` and raise `BackendNotAvailableError` with an install hint when missing.

### Integration
No pipeline changes are required; the resolved pipeline already calls these functions when `analysis.compute_tda_descriptors = True`.

---

## 2. ML segmentation backend

### Current state
`src/fiber_tracer/backends/ml_segmentation.py` is a stub: it lazy-imports PyTorch but raises `NotImplementedError` because no model is loaded.

### Proposed architecture
Build a **lightweight 3D U-Net** (~0.5–1 M parameters) in pure PyTorch:
- 3 encoder levels with `Conv3d(3×3×3) + ReLU + MaxPool3d(2×2×2)`.
- Bottleneck.
- 3 decoder levels with `ConvTranspose3d(2×2×2) + Conv3d` and skip connections from the encoder.
- Output: single-channel sigmoid foreground probability.
- Input patch size: `32×32×32` (matches Bertoldo’s 3D crop size and keeps CPU training feasible).

### Training data
Use the existing `fiber_tracer.validation.phantoms.generate_fiber_phantom`:
- binary target = `phantom.labels > 0`,
- vary `n_fibers`, `fiber_diameter_um`, and `noise_std` per sample,
- extract random `32×32×32` patches from `64×64×64` phantoms,
- apply light augmentation: random axis flips and 90° rotations.

### Training script
Create `scripts/train_unet_phantoms.py` that:
- Generates a configurable number of phantoms (default 200).
- Samples patches (default 16 patches per phantom).
- Trains the model with a combined **BCE + Dice** loss.
- Uses Adam optimizer, learning rate 1e-3, default 30 epochs.
- Saves the best checkpoint by validation Dice to `models/fiber_unet.pt`.
- Prints training/validation metrics and exits cleanly.

### Inference
`MLSegmentationBackend.segment(volume)` will:
1. Load the checkpoint and architecture if `model_path` is provided.
2. Normalize the input volume to `[0, 1]`.
3. Run sliding-window inference with `32×32×32` patches and 50% overlap, averaging overlapping probability maps.
4. Threshold at 0.5 to return a binary `uint8` mask.
5. Raise a clear error if no checkpoint is supplied, pointing the user to `scripts/train_unet_phantoms.py`.

Add a convenience class method:
```python
backend = MLSegmentationBackend.from_checkpoint("models/fiber_unet.pt")
mask = backend.segment(volume)
```

### Integration with the pipeline
The resolved pipeline already branches on `segmentation.method == "unet"` and instantiates `MLSegmentationBackend`. We only need to ensure the backend can load a checkpoint from config. Extend `SegmentationConfig` with an optional `model_path` field that the backend reads.

### Validation
- Train a model in CI/development and verify that `fiber-tracer --regime resolved --segmentation.method unet` runs end-to-end on a phantom.
- The trained model should achieve a foreground Dice > 0.85 on a held-out phantom test set.

---

## 3. Configuration changes

Add to `SegmentationConfig`:
```python
model_path: Optional[str] = None
```

Update `VALID_SEGMENTATION_METHODS` if a new method alias is introduced; keep `"unet"` for the PyTorch backend. Document that `"unet"` requires the `ml` extra:
```bash
pip install -e ".[ml]"
```

The existing `unet` optional-dependency group in `pyproject.toml` is for nnU-Net and remains a separate, Linux-only option.

---

## 4. Documentation updates

- `docs/parameter_guide.md` — explain `segmentation.method = "unet"`, `segmentation.model_path`, and how to train the bundled model.
- `docs/architecture.md` — update the ML backend section to describe the new trainable U-Net and checkpoint workflow.
- `docs/TROUBLESHOOTING.md` — add entries for missing checkpoint and out-of-memory during training/inference.

---

## 5. Testing strategy

| Component | Test file | What it checks |
|-----------|-----------|----------------|
| TDA Betti numbers | `tests/test_tda_backend.py` | known shapes produce expected b0/b1/b2 |
| TDA persistence summary | `tests/test_tda_backend.py` | finite/essential features are counted correctly |
| U-Net architecture | `tests/test_ml_backend.py` | model forward pass on a 32³ tensor |
| Training script smoke | `tests/test_ml_backend.py` | `scripts/train_unet_phantoms.py --epochs 1` runs and writes a checkpoint |
| End-to-end inference | `tests/test_pipeline.py` or new `tests/test_pipeline_unet.py` | `segmentation.method = "unet"` with trained checkpoint yields a binary mask |

---

## 6. Files to create/modify

**Create:**
- `src/fiber_tracer/backends/unet3d.py` — U-Net model definition.
- `scripts/train_unet_phantoms.py` — training script.
- `models/.gitkeep` — directory marker for checkpoints (actual `.pt` files ignored).

**Modify:**
- `src/fiber_tracer/backends/ml_segmentation.py` — load checkpoint, run inference.
- `src/fiber_tracer/backends/tda_gudhi.py` — distance-transform persistence.
- `src/fiber_tracer/config.py` — add `model_path` to `SegmentationConfig`.
- `src/fiber_tracer/pipeline.py` — pass `model_path` to backend.
- `tests/test_ml_backend.py` — extend with architecture and training smoke tests.
- `tests/test_tda_backend.py` — extend with shape-based Betti tests.
- `docs/parameter_guide.md`, `docs/architecture.md`, `docs/TROUBLESHOOTING.md`.
- `.gitignore` — ignore `models/*.pt`.

---

## 7. Acceptance criteria

1. `pytest tests/test_tda_backend.py` passes with gudhi installed.
2. `python scripts/train_unet_phantoms.py --epochs 5` trains and saves `models/fiber_unet.pt`.
3. A trained model segments a synthetic phantom with foreground Dice > 0.85.
4. `fiber-tracer --regime resolved --segmentation-method unet --config ...` runs end-to-end when a checkpoint is supplied.
5. All existing tests, lint, and benchmark still pass.
6. Documentation accurately describes how to install extras, train, and run the backend.

---

## 8. Open decisions

- **Checkpoint distribution:** The trained `.pt` file will not be committed to git (added to `.gitignore`). Users train their own model or the project later adds a release artifact.
- **Patch size:** 32³ is a trade-off between receptive field and CPU memory. It matches the literature and keeps training feasible without a GPU.
- **Number of phantoms:** Default 200 (~3 200 patches) is enough for a proof-of-concept; users can increase for better accuracy.
