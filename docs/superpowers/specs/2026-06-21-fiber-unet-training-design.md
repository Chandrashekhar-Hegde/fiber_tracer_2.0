# Design: Improved Fiber Segmentation Models for fiber-tracer

## Context

`fiber-tracer` Phase 4 added a lightweight 3D U-Net backend and a synthetic-phantom trainer (`scripts/train_unet_phantoms.py`). The current default model is small (`features=(8,16,32)`, ~0.5M parameters) and trained only on synthetic phantoms. The user now wants the best possible model trained on an M5 Pro MacBook Pro (48GB unified RAM), using open data from the Henry Royce Institute X-ray Centre (Manchester, UK) plus other open-source XCT datasets of fiber-reinforced composites.

## Goals

1. Collect, organize, and clean all usable open XCT fiber-composite datasets.
2. Generate reliable voxel-level labels for unlabeled scans using classical segmentation / existing annotations.
3. Train a materially diverse, higher-capacity 3D U-Net that improves on the synthetic-only baseline.
4. Fine-tune a Henry-specific variant on the Henry Royce glass-fiber datasets.
5. Verify models with Dice > 0.85 on held-out real/synthetic data and run the full lint/test/benchmark suite.
6. Upload trained model checkpoints to GitHub via Releases (large binary assets).

## Hardware constraints (M5 Pro, 48 GB)

- PyTorch MPS backend can accelerate 3D convolutions on Apple Silicon, but some training ops (e.g., certain batch-norm/instancenorm reductions, float16 AMP edge cases) may fall back to CPU or raise errors.
- Unified memory means the GPU can access the full 48GB, but training speed is typically 3-4× slower than a comparable NVIDIA GPU. A 30M-parameter 3D U-Net with 64³ patches and batch size 1 fits comfortably; batch size 2 may fit depending on patch size.
- Recommended training mode: MPS with automatic mixed precision (AMP) if stable; otherwise CPU with float32.

## Candidate data sources

| Dataset | Source / DOI | Material | Labels? | License |
|---------|--------------|----------|---------|---------|
| Henry Royce – non-crimp glass-epoxy fatigue | Zenodo 10.5281/zenodo.4541235 | Glass-epoxy NCF | No | Open/CC0 or CC-BY (Zenodo) |
| Henry Royce – UD glass-epoxy compression | Zenodo 10.5281/zenodo.2597498 | Glass-epoxy UD | No | Open/CC0 or CC-BY |
| GF-PA66 | Zenodo 10.5281/zenodo.4587827 | Glass-PA66 | Partial/ground-truth available | CC BY-SA 4.0 |
| DTU UD650 glass fiber | orbit.dtu.dk dataset | Glass-epoxy UD | Segmentation notebook | Check dataset terms |
| Bristol CFRP laminate voids | research-information.bris.ac.uk | Carbon-epoxy UD | No | Check dataset terms |
| Non-crimp fabric dataset | Data in Brief / Mendeley | Glass NCF | No | CC-BY |
| Synthetic phantoms | `fiber_tracer.validation.phantoms` | Generic | Yes (perfect) | Project MIT |

For unlabeled datasets we will generate pseudo-labels with the existing classical pipeline (Otsu + watershed) and manual quality filtering where automatic labels fail.

## Approaches considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| A. Single general model on mixed data | Simplest; one artifact to upload; broad applicability | May be slightly less accurate on Henry-specific contrast | **Primary deliverable** |
| B. Material-specific models + general fallback | Best per-material accuracy | More training time; more files; harder to benchmark | **Secondary: Henry glass + carbon variants if time permits** |
| C. General base + Henry fine-tune | Best Henry accuracy | Requires reliable Henry labels; risk of overfitting small Henry volume | **Try if Henry labels are clean** |

**Chosen approach:** Build a single high-capacity general model first (A). If Henry labels are clean enough, derive a Henry-tuned variant (C). If other materials have clear contrast, optionally add material-specific checkpoints (B). All variants use the same architecture so the backend can load any of them.

## Model architecture

- Keep the custom 3D U-Net in `fiber_tracer.backends.unet3d` but add a deeper/larger variant:
  - `features=(16, 32, 64, 128)` (~7–8M parameters) for the main model.
  - Retain the smaller `(8,16,32)` model as a fast-CPU fallback.
  - Add optional `dropout` and instance normalization support.
- Input: 1-channel grayscale 3D patches (64³ recommended for real data).
- Output: single-channel foreground probability.

## Preprocessing & labeling pipeline

1. **Download** each dataset into `data/raw/<source>/`.
2. **Normalize** to [0,1] per volume (or per dataset if stable).
3. **Generate labels**
   - Synthetic phantoms: exact binary masks.
   - GF-PA66: use provided labels if available.
   - Henry/other real scans: run existing `segment_otsu_3d` / `segment_watershed_3d` to create pseudo-labels; visually inspect sample slices and discard failed volumes.
4. **Patch extraction**
   - Random 64³ patches from regions containing fibers.
   - Balanced sampling: at least 50% patches contain foreground.
   - Augmentation: random flips, 90° rotations, intensity gamma scaling, additive Gaussian noise.
5. **Dataset registry** (`data/datasets.json`) tracks source, voxel spacing, fiber diameter, label type (manual/pseudo), and train/val split.

## Training plan

- Loss: BCEDice + optional boundary-aware term (contour loss) if labels are good.
- Optimizer: AdamW, cosine annealing with warmup.
- Scheduler: ReduceLROnPlateau or CosineAnnealingLR.
- Early stopping on validation Dice.
- Mixed precision with `torch.cuda.amp` equivalent if MPS supports `torch.autocast("mps")`; otherwise full float32.
- Validation: 10% random patch split; also report whole-volume Dice on 1–2 held-out real scans.
- Expected training time per model: several hours to overnight on M5 Pro CPU/MPS.

## Verification

- Unit tests: continue using `tests/test_ml_backend.py` and `tests/test_pipeline_backends.py`.
- Smoke test: train 1 epoch on a tiny subset and load via `MLSegmentationBackend`.
- Quality gate: foreground Dice ≥ 0.85 on a held-out synthetic phantom and on at least one held-out real patch set.
- Benchmark: `scripts/benchmark_phantoms.py` must still pass (classical pipeline unaffected).
- Lint/test: `ruff check .`, `black --check .`, `mypy src/fiber_tracer`, `pytest tests/`.

## Deliverables & GitHub upload

- `models/fiber_unet_v2.pt` — larger general model.
- `models/fiber_unet_henry.pt` — Henry-tuned model (if training succeeds).
- Updated `scripts/train_unet_phantoms.py` or a new `scripts/train_unet_real.py` that ingests the dataset registry.
- Updated docs (`README.md`, `docs/parameter_guide.md`, `CHANGELOG.md`).
- Model checkpoints uploaded as **GitHub Release assets** (not Git LFS) to avoid LFS bandwidth/storage quotas. Each checkpoint is expected to be < 100 MB for the smaller model and < 200 MB for the larger model.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| MPS backend crashes on 3D ops | Fall back to CPU; use smaller patch/batch |
| Real datasets too large for disk | Keep only extracted patches and downsampled previews; delete raw TIFFs after patch extraction if needed |
| Pseudo-labels are noisy | Use conservative Otsu/watershed, filter out low-confidence volumes, weight synthetic data higher |
| Training time exceeds session limits | Use background training jobs; save checkpoints every epoch |
| GitHub file size limits | Use Release assets (up to 2 GB per file) |

## Success criteria

- At least one improved model is trained, passes the 0.85 Dice gate, and is uploaded to GitHub Releases.
- All existing tests and lint checks continue to pass.
- Documentation accurately describes how to download/use the new checkpoints.
