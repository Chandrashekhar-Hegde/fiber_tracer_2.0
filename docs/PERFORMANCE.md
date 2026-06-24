# Performance Guide

This page explains how to get the best runtime and memory performance from Fiber Tracer.

## General principles

- **Use a reasonable output directory on a fast filesystem.** Writing many TIFF slices to a slow network drive can dominate runtime.
- **Normalize and denoise only when needed.** The classical pipeline is fast; U-Net inference is the slowest stage.
- **Disable analysis features you do not need.** Morphometry, orientation tensors, and TDA descriptors all add compute time.

## Classical pipeline (`otsu` / `watershed`)

For a 1,024³ voxel volume on a modern laptop:

- **Otsu segmentation:** seconds to a few minutes.
- **Watershed separation:** adds minutes depending on fiber density.
- **Skeletonization + per-fiber analysis:** scales with the number of fibers.

Progress bars are shown for fiber property computation and orientation-window loops.

## U-Net inference

U-Net inference is the most expensive step. Performance depends on:

| Parameter | Effect |
|-----------|--------|
| `patch_size` | Larger patches mean fewer windows but more memory. Default is the checkpoint's training patch size (64³ for `fiber_unet_v2_full.pt`). |
| `overlap` | Larger overlap improves boundary blending but increases patch count. Default is 16 voxels. |
| `batch_size` | Higher values use more memory but process patches in parallel. Start with 1 and increase until GPU/CPU memory is saturated. |

### Recommended tuning

```bash
# Baseline (slowest, lowest memory)
fiber-tracer --data stack.tif --output out/ --segmentation-method unet --batch-size 1

# Faster if you have 16 GB+ RAM / unified memory
fiber-tracer --data stack.tif --output out/ --segmentation-method unet --batch-size 4

# Maximum throughput on Apple Silicon with ample memory
fiber-tracer --data stack.tif --output out/ --segmentation-method unet --batch-size 8
```

On Apple MPS, memory is shared with the system. If you see slowdowns or memory pressure, reduce `batch_size`.

### Memory estimate

For patch size `(64, 64, 64)` and batch size `B`:

- Input patch memory: ~1 MB per patch.
- Activations during U-Net forward pass: ~50–100 MB per patch, depending on feature depth.
- Safe starting point: `B = 1` for 8 GB systems, `B = 4` for 16 GB systems, `B = 8` for 32 GB+ systems.

## Large volumes and out-of-core processing

If a volume does not fit in RAM:

1. Convert the TIFF stack to Zarr:

   ```bash
   python -c "from fiber_tracer.chunked import tiff_to_zarr; tiff_to_zarr('stack.tif', 'stack.zarr', chunks=(64, 64, 64))"
   ```

2. Process sub-volumes and assemble results, or use the `parallel` extra with Dask.

Out-of-core integration with the main CLI is planned but not yet implemented.

## Profiling a run

Set `--log-level DEBUG` to see per-stage timings:

```bash
fiber-tracer --data stack.tif --output out/ --log-level DEBUG
```

The final `summary.json` also contains `elapsed_seconds` for the full pipeline.

## Known bottlenecks

- **MPS fallback:** On Apple Silicon, `max_pool3d_with_indices` falls back to CPU. This is a PyTorch/MPS limitation, not a Fiber Tracer bug.
- **Watershed on dense volumes:** Can be slow for high fiber volume fractions.
- **TDA descriptors:** Computing persistence diagrams is expensive; enable only when needed.
