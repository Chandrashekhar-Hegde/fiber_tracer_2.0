# Fiber Tracer Roadmap

This document describes where the project is headed and how you can help. For details on the current release, see [`CHANGELOG.md`](CHANGELOG.md). For a per-capability status matrix and how features are switched on, see [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md).

## Current status — v3.2.0

- Regime-aware classical pipeline (`resolved`, `marginal`, `subvoxel`) is stable.
- Optional 3D U-Net backend ships with `fiber_unet_v2_full.pt`.
- Model trained on 2,152 mixed synthetic + real XCT patches.
- Held-out validation on GF-PA66: Dice ≈ 0.90, IoU ≈ 0.81.
- CI/CD, documentation, and honest model card are in place.

## Current status — v3.3.0

- Regime-aware classical pipeline (`resolved`, `marginal`, `subvoxel`) is stable.
- Optional 3D U-Net backend ships with `fiber_unet_v2_full.pt`.
- Model registry, experiment tracking, and local training CLI are implemented.
- **FiberTracer-X foundation model skeleton** is on `main`: synthetic corpus generator,
  shared encoder + task adapters, multi-task trainer, and benchmark harness.

## Short term — v3.4.0 (target: 4–6 weeks)

### FiberTracer-X

- [x] Synthetic phantom architectures (UD, short-fiber, woven bundles, recycled) and XCT domain randomization.
- [x] Shared 3D encoder with segmentation and orientation-regression adapters.
- [x] Multi-task synthetic pre-training trainer (`FiberTracerXTrainer`).
- [x] Volume-level train/val split option and staged pseudo-labeling.
- [x] Benchmark harness + auto-generated leaderboard.
- [ ] YOLO bundle-detection adapter for woven/twill composites.
- [ ] Void/crack segmentation adapter.
- [ ] Fine-tune segmentation adapter on GF-PA66 and at least one pultruded CFRP dataset.
- [ ] Release a pre-trained FiberTracer-X encoder checkpoint on GitHub.

### Engineering

- [ ] Add a GitHub Actions release workflow that builds wheels/sdists and attaches artifacts.
- [ ] Add a separate CI job for ML-backend tests that installs the `ml` extra.
- [ ] Add a Dockerfile for reproducible CPU/MPS inference.
- [x] Build an interactive Terminal UI (TUI) with Bun, Ink, and @inkjs/ui for guided analysis, dashboards, model registry, and experiments.
- [x] Add model registry, experiment tracking, and `fiber-tracer train` CLI for local U-Net training.

### Community

- [ ] Enable GitHub Discussions.
- [x] Add issue templates and PR template.
- [x] Add `CONTRIBUTING.md`, `CONTRIBUTORS.md`, and `CODE_OF_CONDUCT.md`.
- [ ] Open roadmap issues with labels and milestone.

## Medium term — v4.0.0 (target: 3 months)

- [ ] Breaking cleanup of legacy CLI flags and config keys where warranted.
- [ ] Evaluate integration of an nnU-Net backend for users who need state-of-the-art segmentation.
- [ ] Out-of-core / chunked processing exposed from the CLI for large volumes.
- [ ] Comprehensive benchmark suite with persistent JSON results and regression detection.
- [ ] Publish to PyPI with automated releases.

## Long term — research directions

- [ ] **Statistically equivalent synthetic microstructures**: integrate autoregressive fiber-placement models inspired by recent literature.
- [ ] **Domain adaptation / fine-tuning CLI**: let users fine-tune the production model on a small amount of their own labeled data.
- [ ] **Uncertainty quantification**: test-time augmentation or MC dropout confidence maps.
- [ ] **Porosity and damage segmentation**: extend the model to segment voids, cracks, and fracture surfaces alongside fibers.
- [ ] **Multi-modal data**: support for phase-contrast, dual-energy, or synchrotron XCT.

## How to propose roadmap items

1. Check existing [issues](https://github.com/llMr-Sweetll/fiber_tracer_2.0/issues) and [discussions](https://github.com/llMr-Sweetll/fiber_tracer_2.0/discussions).
2. Open a discussion for high-level ideas.
3. Open a focused issue for concrete, actionable work.
4. Reference the roadmap issue or milestone when submitting a pull request.

## Release cadence

- **Patch releases** (`v3.2.x`): bug fixes, documentation corrections, dependency updates.
- **Minor releases** (`v3.x.0`): new features, models, datasets, and non-breaking CLI additions.
- **Major releases** (`v4.0.0`): breaking changes that require user migration.
