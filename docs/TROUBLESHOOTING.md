# Troubleshooting

This guide helps you diagnose and fix common problems when running `fiber-tracer`. For background on the RAFA pipeline, see [`docs/methodology.md`](methodology.md); for configuration details, see [`docs/parameter_guide.md`](parameter_guide.md).

---

## 1. `FileNotFoundError` / `data_path does not exist: ...`

### Symptom
```text
FileNotFoundError: [Errno 2] No such file or directory: '...'
```
or
```text
ValueError: data_path does not exist: ...
```

### Likely cause
The `data_path` supplied via `--data`, a config file, or a batch entry does not point to a file or directory that the current process can see. Common reasons:
- Typo in the path.
- Running from a different working directory than you expect.
- Using a relative path that is not valid from the location where `fiber-tracer` is executed.
- For TIFF slice directories, the directory exists but contains no files, or the expected slice naming pattern is different from what the loader uses.

### Fix
1. Check that the file or directory exists:
   ```bash
   ls -la /path/to/data
   ```
2. Use an absolute path when in doubt, or change directory to the project root before running.
3. For directories of TIFF slices, ensure the files are named in a consistent alphanumeric order (for example `slice_000.tif`, `slice_001.tif`, ...). Rename or symlink them if necessary.
4. Verify permissions; the process must be able to read the path.

### Example
```bash
# Use absolute path
fiber-tracer --data /home/user/data/sample_a.tif --output results/sample_a/ \
  --voxel-spacing 1.0 1.0 1.0 --fiber-diameter 6.0
```

**Related docs:** [`docs/parameter_guide.md`](parameter_guide.md), `fiber_tracer.io` module in [`docs/architecture.md`](architecture.md).

---

## 2. `TypeError: expected dict or VoxelSpacing, got list`

### Symptom
```text
TypeError: expected dict or VoxelSpacing, got list
```

### Likely cause
Older scripts or configs mixed two different ways of specifying `voxel_spacing_um`. `fiber_tracer.config.Config.from_dict()` accepts a list/tuple `[z, y, x]` and converts it to a `VoxelSpacing` object automatically, but the internal dataclass validation expects either that list/tuple or a dict like `{"z": 1.0, "y": 1.0, "x": 1.0}`. Passing a partially-converted or nested structure can trigger this error.

### Fix
Use one of the two supported forms consistently in YAML/JSON configs, and avoid mixing them:

```yaml
# Form A: list in (z, y, x) order
voxel_spacing_um: [1.0, 1.0, 1.0]

# Form B: explicit dict
voxel_spacing_um:
  z: 1.0
  y: 1.0
  x: 1.0
```

On the command line, pass three separate floats:

```bash
fiber-tracer --data input.tif --output results/ \
  --voxel-spacing 1.0 1.0 1.0 --fiber-diameter 6.0
```

**Related docs:** [`docs/parameter_guide.md`](parameter_guide.md), [`src/fiber_tracer/config.py`](../src/fiber_tracer/config.py).

---

## 3. Wrong regime selected / poor segmentation

### Symptom
The summary reports a regime you did not expect (for example `subvoxel` for a clearly resolved scan), or the segmentation/morphometry looks qualitatively wrong.

### Likely cause
Regime selection is driven by the ratio

```text
r = min(voxel_spacing_z, voxel_spacing_y, voxel_spacing_x) / fiber_diameter_um
```

If `fiber_diameter_um` or `voxel_spacing_um` is off by an order of magnitude, the wrong regime is chosen. Anisotropic spacing can also push the conservative minimum into a different regime.

### Fix
1. Compute `r` by hand using your actual values.
2. Compare with the thresholds:
   - `resolved`: `r <= 0.3`
   - `marginal`: `0.3 < r <= 3.0`
   - `subvoxel`: `r > 3.0`
3. If the ratio is borderline, force the regime that matches your data physically:

   ```bash
   # Fibers are clearly larger than several voxels
   fiber-tracer ... --regime resolved

   # Fiber diameter is comparable to the voxel size
   fiber-tracer ... --regime marginal

   # Many fibers fit inside one voxel
   fiber-tracer ... --regime subvoxel
   ```
4. Double-check units: `voxel_spacing_um` and `fiber_diameter_um` must both be in micrometres.

### Example
```bash
# spacing 2.0 x 1.0 x 1.0 µm, fiber diameter 6.0 µm
# min(spacing) = 1.0 -> r = 1.0 / 6.0 = 0.167 -> resolved
fiber-tracer --data sample.tif --output results/ \
  --voxel-spacing 2.0 1.0 1.0 --fiber-diameter 6.0 --regime resolved
```

**Related docs:** [`docs/methodology.md`](methodology.md), `regime` section in [`docs/parameter_guide.md`](parameter_guide.md).

---

## 4. Touching or over-segmented fibers in the resolved regime

### Symptom
Multiple fibers are merged into one label, or one fiber is split into many small labels.

### Likely cause
In the resolved regime the default segmentation method is `otsu`, which labels connected components. Touching fibers are therefore merged. Conversely, if the foreground mask is noisy or fragmented, a single fiber can be broken into pieces.

### Fix
- For touching fibers, switch to marker-controlled watershed:
  ```yaml
  segmentation:
    method: watershed
  ```
- If the mask is noisy, increase Gaussian denoising before thresholding:
  ```yaml
  processing:
    denoise_sigma: 1.0
  ```
- If watershed over-segments elongated fibers, reduce `denoise_sigma` or try `otsu` again after improving contrast.
- For very densely packed or irregular fibers, no single classical method may be perfect; consider preprocessing or an advanced backend (see [`docs/methodology.md`](methodology.md) limitations).

### Example
Create `watershed.yaml`:

```yaml
segmentation:
  method: watershed
```

Run:

```bash
fiber-tracer --config watershed.yaml --data sample.tif --output results/ \
  --voxel-spacing 1.0 1.0 1.0 --fiber-diameter 6.0 --regime resolved
```

**Related docs:** Segmentation parameters in [`docs/parameter_guide.md`](parameter_guide.md), resolved-regime pipeline in [`docs/methodology.md`](methodology.md).

---

## 5. U-Net backend issues

### Symptom: `FileNotFoundError` when using `--segmentation-method unet`

```text
FileNotFoundError: [Errno 2] No such file or directory: 'models/fiber_unet_v2_full.pt'
```

### Likely cause

The U-Net backend requires a PyTorch checkpoint. The production model is not committed to git because of its size; it must be downloaded separately.

### Fix

Download the checkpoint from the [v3.2.0-unet-v2 release](https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/releases/tag/v3.2.0-unet-v2) and place it at the path specified by `--model-path` (default: `models/fiber_unet_v2_full.pt`).

```bash
# Example
mkdir -p models
curl -L -o models/fiber_unet_v2_full.pt \
  https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/releases/download/v3.2.0-unet-v2/fiber_unet_v2_full.pt
```

---

### Symptom: U-Net output is mostly background or mostly foreground

### Likely cause

The default probability threshold (0.5) may not be appropriate for your data, or your data distribution differs from the training set.

### Fix

1. Inspect the probability-like output or the binary `labels.tif` in napari.
2. Adjust the decision threshold if your application exposes one, or post-process the mask with morphological operations.
3. Validate the model on a small annotated subset of your data before running the full pipeline.

**Related docs:** Model card and failure modes in [`docs/MODEL_CARD.md`](MODEL_CARD.md), U-Net workflow in [`docs/USER_GUIDE.md`](USER_GUIDE.md).

---

## 6. `BackendNotAvailableError` for optional extras

### Symptom
```text
fiber_tracer.exceptions.BackendNotAvailableError: Install viz extra: pip install fiber-tracer[viz]
```
Similar messages reference `skeleton`, `structure`, `tda`, `ml`, or `parallel` extras.

### Likely cause
A feature that depends on an optional package was requested, but that package is not installed. Optional extras are not installed by the core package.

### Fix
Install the relevant extra in editable mode from the repository root:

```bash
# Visualization (napari + plotly)
pip install -e ".[viz]"

# Skeleton graph adapter (skan)
pip install -e ".[skeleton]"

# Optional structure-tensor package backend
pip install -e ".[structure]"

# Topological descriptors via gudhi
pip install -e ".[tda]"

# PyTorch/scikit-learn ML backend
pip install -e ".[ml]"

# Chunked/distributed helpers (zarr + dask)
pip install -e ".[parallel]"

# All optional backends at once
pip install -e ".[all]"
```

Then re-run the command.

**Related docs:** Installation section in [`README.md`](../README.md), optional backends in [`docs/architecture.md`](architecture.md).

---

## 7. napari / GUI viewer issues on headless servers

### Symptom
`fiber-tracer view ...` fails with display errors such as:
```text
Could not initialize Qt
```
or
```text
$DISPLAY not set
```
or the command hangs because no X server is available.

### Likely cause
The `view` subcommand launches a local napari GUI, which requires a display. It will not work on remote/headless servers, CI runners, or through plain SSH without X forwarding.

### Fix
1. On a headless machine, skip the GUI viewer and use the HTML report instead:
   ```bash
   # Generate the pipeline summary first
   fiber-tracer --data sample.tif --output results/ \
     --voxel-spacing 1.0 1.0 1.0 --fiber-diameter 6.0

   # Then create a self-contained interactive report
   fiber-tracer report-viz --summary results/summary.json --output report.html
   ```
2. If you must use napari remotely, enable X11 forwarding (`ssh -X ...`) or run a VNC/NoMachine desktop.
3. Verify the `viz` extra is installed; otherwise you will hit a `BackendNotAvailableError` before reaching the display issue.

**Related docs:** Visualization section in [`README.md`](../README.md), [`docs/parameter_guide.md`](parameter_guide.md).

---

## 8. Plotly HTML report fails to load

### Symptom
`report.html` opens as a blank page, or the browser shows a loading spinner indefinitely.

### Likely cause
- The file was not written completely (process interrupted).
- The browser blocks local JavaScript when opening the file with `file://` restrictions, or an outdated browser cannot load the self-contained Plotly bundle.
- The report was moved without its embedded resources (if it were not self-contained, but `fiber-tracer` writes self-contained files).

### Fix
1. Regenerate the report:
   ```bash
   fiber-tracer report-viz --summary results/summary.json --output report.html
   ```
2. Use a modern browser (recent Firefox, Chrome, Edge, or Safari).
3. Try opening the file directly by double-clicking, or serve it temporarily:
   ```bash
   python -m http.server 8000
   # then open http://localhost:8000/report.html
   ```
4. Check that `summary.json` is valid JSON and contains the expected regime and metrics.

**Related docs:** Visualization section in [`README.md`](../README.md), report output reference in [`README.md`](../README.md).

---

## 9. Out-of-memory on large volumes

### Symptom
The process is killed by the OS, raises `MemoryError`, or begins heavy swapping and becomes extremely slow.

### Likely cause
The volume is larger than available RAM. The default CLI pipeline keeps the full array and intermediate results in memory.

### Fix
1. Preprocess the volume in chunks using the `fiber_tracer.chunked` helpers:
   ```python
   from fiber_tracer.chunked import tiff_to_zarr, normalize_intensity_chunked
   import zarr

   input_zarr = tiff_to_zarr("large_stack.tif", "input.zarr", chunks=(64, 64, 64))
   output_zarr = zarr.open_array(
       "normalized.zarr", mode="w", shape=input_zarr.shape,
       chunks=(64, 64, 64), dtype="float32"
   )
   normalize_intensity_chunked(input_zarr, output_zarr, chunk_shape=(64, 64, 64))
   ```
2. Install the `parallel` extra for `dask` support if you need distributed workflows:
   ```bash
   pip install -e ".[parallel]"
   ```
3. Use smaller chunks and write intermediate arrays to disk instead of keeping everything in RAM.
4. Downsample or crop a representative sub-volume first to estimate parameters before running the full dataset.

**Related docs:** [`docs/parameter_guide.md`](parameter_guide.md) (Chunked / out-of-core processing), [`docs/architecture.md`](architecture.md), `src/fiber_tracer/chunked.py`.

---

## 10. Batch config parse errors

### Symptom
```text
yaml.scanner.ScannerError: ...
```
or
```text
TypeError: expected dict, got str
```
or
```text
ValueError: data_path does not exist: ...
```
when running `fiber-tracer batch --config batch.yaml`.

### Likely cause
- YAML syntax error (missing colons, inconsistent indentation, tabs instead of spaces).
- A `volumes` entry is not indented under the list correctly.
- Required keys (`data_path`, `output_dir`) are missing.
- A per-volume override uses an invalid key or misspelled section name.

### Fix
Use a minimal valid structure and validate it before scaling up:

```yaml
common:
  voxel_spacing_um: [1.0, 1.0, 1.0]
  fiber_diameter_um: 6.0
  regime: auto
  processing:
    denoise_sigma: 0.8
    normalize: true
  segmentation:
    method: otsu
  analysis:
    compute_morphometry: true
    compute_orientation_tensor: true

volumes:
  - data_path: sample_a.tif
    output_dir: results/sample_a
  - data_path: sample_b.tif
    output_dir: results/sample_b
    fiber_diameter_um: 4.0
    regime: resolved
```

Tips:
- Indent lists with two spaces per level.
- Do not use tabs in YAML.
- Make sure `data_path` and `output_dir` exist or can be created.
- Per-volume keys override only the matching keys in `common`; unknown keys are not silently accepted by the dataclass loader.

Run with:
```bash
fiber-tracer batch --config batch.yaml --aggregate-csv batch_summary.csv
```

**Related docs:** Batch processing in [`README.md`](../README.md), [`docs/parameter_guide.md`](parameter_guide.md), [`src/fiber_tracer/config.py`](../src/fiber_tracer/config.py).

---

## 11. Validation or benchmark failures

### Symptom
`scripts/benchmark_phantoms.py` exits non-zero, reports a Dice score below 0.85, or a mean angular error above 5°. External GF-PA66 validation produces lower scores than expected.

### Likely cause
- The deterministic phantom benchmark has strict thresholds designed for ideal synthetic fibers. Small environment differences, dependency versions, or parameter drift can push results just outside the passing window.
- Real XCT data contains noise, partial-volume effects, and touching fibers that the phantom benchmark does not model. Applying the same thresholds to real data is usually unrealistic.

### Fix
1. Check that you are using the same dependency versions as the test suite:
   ```bash
   pip install -e ".[dev]"
   pytest tests/test_benchmark.py -v
   ```
2. Inspect the generated outputs (`labels.tif`, `skeleton.tif`, `report.csv`) to decide whether the results are qualitatively acceptable even if a single numeric threshold is missed.
3. For real data, adjust expectations. Use the phantom benchmark to verify that the installation is healthy, then tune `denoise_sigma`, `segmentation.method`, and `fiber_diameter_um` on a representative subset.
4. When comparing against the GF-PA66 dataset, follow the protocol in `scripts/validate_gfpa66.py` and document any preprocessing choices.

### Example
```bash
# Verify the core benchmark path
python scripts/benchmark_phantoms.py

# Run the regression test that wraps the benchmark script
pytest tests/test_benchmark_regression.py -v
```

**Related docs:** [`docs/validation_protocol.md`](validation_protocol.md), Validation and benchmarking in [`README.md`](../README.md), [`docs/methodology.md`](methodology.md) limitations.

---

## Still stuck?

If an issue is not covered here:
1. Re-run with `--log-level DEBUG` to capture the full traceback.
2. Confirm the installation is complete: `pip install -e ".[dev]"` and `pytest tests/ -v`.
3. Open an issue at the repository with the error message, command line or config, and a minimal input file if possible.
