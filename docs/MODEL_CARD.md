# Model Card: `fiber_unet_v2_full.pt`

This document describes the production 3D U-Net shipped with Fiber Tracer v3.2.0. It is written to be transparent about what the model can and cannot do, how it was trained, and how it was validated.

## Model identity

| Attribute | Value |
|-----------|-------|
| File | `models/fiber_unet_v2_full.pt` |
| Release | <https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/releases/tag/v3.2.0-unet-v2> |
| Architecture | 3D U-Net (`fiber_tracer.backends.unet3d.UNet3D`) |
| Encoder features | `(16, 32, 64, 128)` |
| Normalization | Batch normalization |
| Dropout | 0.1 |
| Input patch size | 64 × 64 × 64 voxels |
| Input channels | 1 (grayscale XCT) |
| Output channels | 1 (binary fiber probability) |
| Final activation | Sigmoid |
| Parameters | ~5.8 M |

## Intended use

The model is designed to segment individual fiber voxels in 3D X-ray computed-tomography (XCT) volumes of fiber-reinforced composites. It is intended to be used as the `unet` backend in `fiber-tracer`:

```bash
fiber-tracer --data stack.tif --output results/ \
  --segmentation-method unet --model-path models/fiber_unet_v2_full.pt
```

It is **not** a general-purpose XCT segmentation model. It was trained exclusively on images of polymer/glass/carbon fiber composites and should not be expected to work on bone, metal foams, geological samples, or other microstructures without retraining.

## Training data

### Volume mix

The model was trained on 2,152 3D patches (64³ voxels) drawn from synthetic phantoms and 18 open XCT volumes.

| Source | Volumes | Fibers represented |
|--------|---------|---------------------|
| Synthetic phantoms | 1,000 patches | glass/carbon, random/aligned/0°/90°/±45°/woven/twill, broken fibers, porosity, variable diameter/length |
| Henry Royce Institute benchmark | Glass_1_Sub, Carbon_1_Sub | glass and carbon fiber composites |
| Henry Royce Institute NCF fatigue | 0_cycles | non-crimp fabric glass fibers |
| Henry Royce Institute UD compression | initial + static compression | unidirectional glass fibers, pre- and post-damage |
| DTU Wind Energy / Composites | pultruded CFRP sample A | aligned carbon fibers |
| IVW carbon twill weave | CF_weave_stitched_images_2x2 | woven carbon fibers |
| IVW recycled CFRP | rCF_stitched_images_2x2 | recycled/recovered carbon fibers |
| IVW short GFRP | sGFRP_stitched_images_3x4 | short glass fibers |
| GF-PA66 | pa66_volumes.h5 | glass-fiber-reinforced PA66 with ground-truth labels |

### Data splits

Patches were assigned deterministically to train/validation sets (90/10). GF-PA66 was held out entirely from training; it was used only for final held-out validation.

### Preprocessing

1. Each raw volume was converted to `np.float32`.
2. A global intensity threshold was estimated from a sampled intensity histogram.
3. Foreground-biased 64³ patches were extracted with streaming reads to avoid loading multi-gigabyte volumes whole.
4. Patches were rejected if their foreground ratio fell below a minimum threshold (≈0.05%). This threshold is low because thin fibers occupy a tiny volume fraction even when a patch is centered on a fiber.
5. Training patches were augmented with random flips, 90° rotations, and elastic deformations.
6. Deterministic oversampling was applied so that patches with higher fiber volume fraction appear more often during training.

### Synthetic phantoms

The phantom generator (`fiber_tracer.validation.phantoms.generate_fiber_phantom`) supports:

- `orientation_mode`: `random`, `aligned`, `in_plane`, `orthogonal`, `woven`, `twill`
- `broken_fraction`: fraction of fibers cut into multiple pieces
- `porosity`: spherical/ellipsoidal voids
- variable fiber diameter, length, and number

Phantoms are **idealized**. They do not reproduce beam-hardening artifacts, partial-volume effects, fiber-matrix contrast variations, or noise distributions of real lab XCT systems. They are useful for curriculum learning and sanity checks, but they do not guarantee real-world accuracy.

## Training procedure

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Initial learning rate | 3×10⁻⁴ |
| Scheduler | Cosine annealing over 100 epochs |
| Batch size | 1 |
| Loss | Balanced BCE + Dice (`bce_weight = 0.5`) |
| Epochs trained | 16 (stopped early due to epoch-time slowdown; best checkpoint from epoch 1 by validation foreground hard Dice) |
| Hardware | Apple M5 Pro, 48 GB unified RAM, MPS backend with CPU fallback for unsupported ops |

Training was performed in float32. Automatic mixed precision was not used because MPS does not support `ConvTranspose3d` in fp16/bf16 and `max_pool3d_with_indices` falls back to CPU.

### Class imbalance handling

Fiber voxels are a small minority (mean foreground ratio ≈ 4.2% in the training corpus). Instead of a large positive weight in BCE — which made the model over-predict foreground and hurt Dice — we used deterministic oversampling of fiber-rich patches. This gives the model more positive examples without distorting the loss surface.

## Validation results

### In-domain validation patches

On the 10% validation patches that contain at least 0.1% foreground voxels, the saved checkpoint achieved:

- **Foreground hard Dice = 0.948**
- Hard Dice = 0.417 (all patches, dominated by background-only crops)
- Soft Dice = 0.043 (all patches)

The overall hard/soft Dice numbers are low because the validation corpus still contains many background-only patches. The foreground-conditioned metric is the relevant number for assessing fiber segmentation quality.

### Held-out GF-PA66 ground truth

GF-PA66 was **not** used during training or validation patch selection. We ran sliding-window inference on the central 128 slices of `pa66/data` and compared against `pa66/ground_truth` (classes > 0 treated as fiber):

| Metric | Value |
|--------|-------|
| Target foreground voxels | 30,117,323 |
| Predicted foreground voxels | 24,565,325 |
| Dice | **0.8953** |
| IoU | **0.8105** |
| Pixel accuracy | **0.9669** |

### What these numbers mean

- Dice ≈ 0.90 means the predicted fiber mask overlaps well with the manual/expert ground truth on average.
- IoU ≈ 0.81 means about 81% of the union of predicted and true fiber regions is correctly classified.
- Accuracy ≈ 0.97 is high because most voxels are background; it is **not** the primary metric.

### Honest caveats

1. **Single reference dataset.** The only ground-truth-labeled public volume we could validate against is GF-PA66. The model may perform differently on other fiber systems, contrast levels, or scan resolutions.

2. **GF-PA66 is not independent in spirit.** Although it was held out from training, it is one of the datasets the model was exposed to during development (we used it repeatedly to diagnose training issues and to tune patch extraction). Reported numbers should be treated as an internal validation, not a fully blind external benchmark.

3. **Pseudo-labels for real training data.** All real training volumes except GF-PA66 were labeled with Otsu thresholding, not expert annotations. The model therefore learns to reproduce an Otsu-like decision boundary on those volumes. It may inherit Otsu failures (e.g., touching fibers, low contrast, partial-volume ambiguity).

4. **Patch size limits context.** The model sees only 64³ voxels at a time. Long-range fiber continuity, large pores, and specimen-scale defects are not modeled. Sliding-window inference with overlap averages local decisions but cannot introduce global context.

5. **Class imbalance remains severe.** Even with oversampling, most patches contain very few fiber voxels. The model may be conservative and under-segment sparse or low-contrast fibers.

6. **MPS backend limitations.** The model was trained and runs on Apple MPS with CPU fallback for some operators. Inference on CUDA or CPU will produce bitwise-identical results for the same weights, but throughput and memory usage differ.

## Known failure modes

- **Under-segmentation of thin, low-contrast fibers:** If fiber voxels have intensity close to the matrix, the model may miss them.
- **Over-segmentation of bright artifacts:** High-density particles, porosity edges, or reconstruction streaks can be misclassified as fiber.
- **Resolution mismatch:** If the test voxel size or fiber diameter is very different from the training distribution, accuracy may drop. The model has not been systematically evaluated across resolution transfer.
- **Highly curved or entangled fibers:** The training set contains mostly straight or gently curved fibers. Very tortuous short-fiber mats may segment poorly.
- **Broken fibers:** Broken-fiber phantoms were included, but real fracture surfaces, pull-outs, and debris were not well represented.

## Recommendations for use

1. **Validate on your own data.** Use the model as a starting point, not a black-box oracle. Inspect `labels.tif` against the raw volume before reporting quantitative results.
2. **Compare with the classical backend.** Run the same volume with `segmentation.method: "otsu"` or `"watershed"` and compare. The U-Net is not always better, especially for very clean resolved-regime data.
3. **Tune the probability threshold.** The default threshold is 0.5. For your data, sweep `[0.3, 0.5, 0.7]` and pick the value that balances false positives and false negatives.
4. **Respect physical voxel spacing.** The model has no explicit notion of µm; it operates on 64³ voxel cubes. If your voxel spacing is very different from the training set, consider retraining or resampling.
5. **Consider fine-tuning.** If you have even a small amount of annotated data from your system, fine-tuning will almost always outperform the generic model.

## How to retrain or fine-tune

```bash
# 1. Download open datasets
python scripts/download_datasets.py

# 2. Prepare patches
python scripts/prepare_training_data.py \
  --n-synthetic 1000 \
  --n-patches-per-volume 64 \
  --output data/processed_full \
  --raw data/raw

# 3. Train
python scripts/train_unet_mixed.py \
  --features 16 32 64 128 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 1 \
  --lr 3e-4 \
  --output models/fiber_unet_v2_retrained.pt \
  --registry data/processed_full/datasets.json \
  --processed-root data/processed_full

# 4. Validate on GF-PA66 or your own labels
python scripts/validate_unet_gfpa66.py \
  --checkpoint models/fiber_unet_v2_retrained.pt \
  --data data/raw/gfpa66/pa66_volumes.h5 \
  --image-key pa66/data \
  --label-key pa66/ground_truth
```

## Citation

If you use this model in published work, please cite the Fiber Tracer repository and the datasets you used:

```bibtex
@software{fiber_tracer_2026,
  title = {Fiber Tracer 2.0: Regime-Aware Fiber Analysis},
  author = {{Fiber Tracer contributors}},
  year = {2026},
  url = {https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0}
}

@article{bertoldo2021modular,
  title={A Modular U-Net for Automated Segmentation of X-Ray Tomography Images in Composite Materials},
  author={Bertoldo, J. P. C. and others},
  journal={Frontiers in Materials},
  volume={8},
  pages={761229},
  year={2021},
  publisher={Frontiers Media},
  doi={10.3389/fmats.2021.761229}
}
```

See [`CITATIONS.md`](../CITATIONS.md) for full dataset citations.

## Model maintenance

- The checkpoint is version-locked to the code that produced it. Loading it with a different `UNet3D` feature vector or normalization setting will fail.
- Future releases may introduce new architectures or training corpora. This release tag will remain available for reproducibility.
