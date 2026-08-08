# Capabilities status & roadmap

This document is the single source of truth for what Fiber Tracer can do today,
how each capability is switched on, and what is planned next. It covers six
focus areas: **segmentation, thresholding, fibre tracking, Digital Volume
Correlation (DVC), Digital Image Correlation (DIC), and the digital twin**.

Status legend: ✅ implemented · ⚠️ partial / not wired up · ❌ absent.

## Status matrix

| Area | Status | Where it lives / how it switches |
|---|---|---|
| Segmentation — Otsu | ✅ | `segmentation/classical.py::segment_otsu_3d`; `segmentation.method=otsu` (default) |
| Segmentation — watershed | ✅ | `segmentation/classical.py::segment_watershed_3d`; `segmentation.method=watershed` |
| Segmentation — connected components | ✅ | `segmentation/classical.py::segment_connected_components_3d` (labeling step) |
| Segmentation — 3D U-Net | ✅ | `backends/unet3d.py`, `backends/ml_segmentation.py`; `segmentation.method=unet` + `--model-path` (needs `ml` extra) |
| Segmentation — nnU-Net | ⚠️ | dependency declared in `pyproject.toml` (`unet` extra) but no pipeline path uses it |
| Thresholding — global Otsu | ✅ | `segment_otsu_3d`; default of `binarize_volume` |
| Thresholding — manual / adaptive / multi-Otsu | ✅ | `segmentation/classical.py::binarize_volume`; `segmentation.threshold_method` + `--threshold-method` / `--threshold-value` |
| Pre-processing — normalize / Gaussian denoise | ✅ | `preprocess.py`; `processing.normalize`, `processing.denoise_sigma` |
| Fibre tracking — skeletonization | ✅ | `centerline/skeleton.py::skeletonize_label_volume` |
| Fibre tracking — per-fibre PCA orientation | ✅ | `orientation/pca.py::pca_orientation`; `analysis.compute_orientation_tensor` |
| Fibre tracking — equivalent diameter | ✅ | `analysis/morphometry.py::equivalent_diameter_from_volume`; `analysis.compute_morphometry` |
| Fibre tracking — centerline paths / length / tortuosity | ✅ | `centerline/paths.py::extract_fiber_paths` feeds `ordered_path_length`/`tortuosity` in the resolved per-fiber loop; `analysis.compute_tracking` |
| Fibre tracking — skeleton graph (skan) | ⚠️ | `centerline/graph.py::skeleton_to_skan` wrapper exists (needs `skeleton` extra) but is not integrated |
| Orientation field (structure tensor) | ✅ | `orientation/structure_tensor.py`, `orientation/tensor.py` (marginal/subvoxel regimes) |
| DVC (3D displacement/strain) | ✅ | `correlation/dvc.py::run_local_dvc`; `dvc.enabled` — see [`DVC_BENCHMARK.md`](DVC_BENCHMARK.md) for accuracy/convergence figures |
| DIC (2D displacement/strain) | ✅ | `correlation/dic.py::run_local_dic`; `dic.enabled` — shares the engine in `correlation/core.py` with DVC |
| Digital twin | ✅ | `twin/fitting.py::fit_twin_parameters`; `twin.enabled` — resolved regime only |

## How switches flow (frontend → backend)

A single configuration object travels through four layers; any new switch must be
threaded through all of them:

1. **TUI** — `tui/src/types.ts` (`AnalysisConfig`) collects choices in the
   wizard, then `tui/src/bridge.ts` (`buildJson`) serializes them to a JSON
   config and spawns `fiber-tracer run --config <file>`.
2. **CLI** — `cli.py` parses subcommands and flags, loads the config file, and
   applies flag overrides (`cli.py` `_run_pipeline`, the override block around
   `cli.py:369-383`).
3. **Config** — `config.py` defines the typed schema and validates choices
   (`Config`, `ProcessingConfig`, `SegmentationConfig`, `OrientationConfig`,
   `AnalysisConfig`).
4. **Pipeline** — `pipeline.py` detects the regime (`regime.py::detect_regime`)
   and dispatches algorithms / backends (`pipeline.py:181-203`).

Key switches today:

| Switch | Config key | CLI flag | TUI field |
|---|---|---|---|
| Regime | `regime` | `--regime` | `regime` |
| Segmentation method | `segmentation.method` | `--segmentation-method` | `method` |
| Threshold method | `segmentation.threshold_method` | `--threshold-method` | `thresholdMethod` |
| Manual threshold value | `segmentation.threshold_value` | `--threshold-value` | `thresholdValue` |
| Centerline tracking | `analysis.compute_tracking` | (config file) | `computeTracking` |
| Model | `segmentation.model_path` | `--model-path` | `model` |
| Batch size | `segmentation.batch_size` | `--batch-size` | `batchSize` |
| Voxel spacing | `voxel_spacing_um` | `--voxel-spacing` | `voxelSpacing` |
| Fibre diameter | `fiber_diameter_um` | `--fiber-diameter` | `fiberDiameter` |
| Morphometry / orientation / TDA toggles | `analysis.*` | (config file) | `computeMorphometry` / `computeOrientationTensor` / `computeTda` |
| DVC enable + reference/deformed volumes | `dvc.enabled`, `dvc.reference_path`, `dvc.deformed_path` | (config file) | `computeDvc`, `dvcReferencePath`, `dvcDeformedPath` |
| DVC grid/window/convergence tuning | `dvc.node_spacing_voxels`, `dvc.half_window_size_voxels`, `dvc.min_convergence_rate` | (config file) | (config file) |
| DIC enable + reference/deformed images | `dic.enabled`, `dic.reference_path`, `dic.deformed_path` | (config file) | `computeDic`, `dicReferencePath`, `dicDeformedPath` |
| DIC grid/window/convergence tuning | `dic.node_spacing_pixels`, `dic.half_window_size_pixels`, `dic.min_convergence_rate` | (config file) | (config file) |
| Digital twin enable | `twin.enabled` | (config file) | `computeTwin` |
| Digital twin constituent properties | `twin.fiber_modulus_gpa`, `twin.matrix_modulus_gpa`, `twin.aspect_ratio` | (config file) | (config file) |

## Roadmap

Work is tracked as GitHub epics. The eight existing issues (#3–#10) all concern
the ML / U-Net data and validation pipeline and roll up under *Segmentation
quality*.

### Milestone: thresholding + fibre tracking (next)

- **Thresholding options** — add a `threshold_method` switch (otsu / manual /
  adaptive / multi-Otsu) reusing scikit-image, threaded through config, CLI, and
  TUI. Centralize binarization behind a single helper in
  `segmentation/classical.py`.
- **Complete fibre tracking** — extract ordered per-fibre centerlines and report
  `length_um` and `tortuosity` (the maths already lives in
  `analysis/morphometry.py`); optionally surface skan graph metrics when the
  `skeleton` extra is installed.

### Milestone: segmentation quality

Umbrella for the open ML issues: patch foreground sampling (#10), advanced
phantom generation (#4), domain-randomization augmentation (#5), domain
adaptation / fine-tuning CLI (#6), uncertainty / ensemble inference (#7), dataset
downloader expansion (#8, #3), public benchmark leaderboard (#9), plus
integrating the declared nnU-Net backend.

### Shipped: DVC & DIC (epics #19, #20)

Local grid-based correlation is implemented for both 3D volumes (DVC) and 2D
images (DIC), sharing one engine (`correlation/core.py`, `spam` backend) —
the DIC spike (`docs/superpowers/specs/2026-08-04-dic-spike-design.md`)
confirmed the same `spam.DIC.register` call needs no modality-specific
algorithm code, just a `(1, H, W)`-shaped image instead of a full volume.
`correlation/dvc.py` and `correlation/dic.py` are thin re-exports. See the
status matrix above and [`DVC_BENCHMARK.md`](DVC_BENCHMARK.md) for the
accuracy/convergence figures and cross-tool comparison (which applies to both,
being the same engine). DVC was preceded by a feasibility spike
(`docs/superpowers/specs/2026-08-02-dvc-spike-design.md`), DIC by
`docs/superpowers/specs/2026-08-04-dic-spike-design.md`.

### Shipped: Digital twin (epic #21)

Fits phantom-generator parameters (fiber count, cross-sectional diameter,
orientation mode) from a run's own resolved-regime output, regenerates a
statistically-matched synthetic volume, and estimates an effective modulus
via Halpin-Tsai (`RESEARCH_FOUNDATION.md` ref 67). Scope is deliberately
narrow — see `docs/superpowers/specs/2026-08-07-digital-twin-spike-design.md`
for what's explicitly excluded (CAD/nominal-design reconciliation, DVC/DIC
deformation coupling, FE mesh export, continuous orientation-tensor fitting).
Preceded by a scope-definition spike per the epic's own acceptance criteria
(sign-off before build, not a working-prototype bar like DVC/DIC's spikes).

Known limitation: [issue #28](https://github.com/Chandrashekhar-Hegde/fiber_tracer_2.0/issues/28)
— `extract_fiber_paths` fails to produce a centerline for most fibers in a
realistic densely-packed unidirectional bundle, so the twin's diameter fit
falls back to a less-accurate volume-extent estimate for those fibers.
