# FiberTracer-X: Task-Aware Fiber Analysis Foundation Model

FiberTracer-X is the next-generation model family for `fiber-tracer`.  Instead of
a single generic U-Net trained on every dataset, it uses a **shared 3D encoder
pre-trained on a large, physically diverse synthetic corpus**, with lightweight
task- and material-specific adapters for segmentation, orientation regression,
bundle detection, and void/crack segmentation.

## Why a foundation model?

Recent literature (2022–2026) shows that no single architecture fits all XCT
fiber data:

- **3D U-Nets** work best for high-resolution individual-fiber segmentation.
- **2D/2.5D U-Nets** are often better for anisotropic scans.
- **YOLO-family detectors** excel at woven/twill bundle detection.
- **CNN regressors** are the right tool when fibers are below resolution.
- **Domain-randomized synthetic data** is the most reliable way to pre-train
  before fine-tuning on real scans.

FiberTracer-X unifies these insights: one encoder learns generic XCT fiber
features from synthetic data, and adapters specialize per task.

## Architecture

```text
Input XCT volume + metadata (regime, material, task)
           │
           ▼
   Shared 3D encoder (pre-trained on synthetic corpus)
           │
   ┌───────┼───────┬─────────────┐
   ▼       ▼       ▼             ▼
3D U-Net  YOLO  Orientation   Void/crack
segment   head  regressor     segmentation
adapter       adapter       adapter
```

Current adapters:

| Adapter | Task | Output |
|---|---|---|
| `Segmentation3DAdapter` | Semantic segmentation | Per-voxel class logits |
| `OrientationRegressorAdapter` | Orientation tensor | 6 unique A2 components |

Planned adapters: YOLO bundle-detection head, void/crack segmentation adapter.

## Synthetic corpus

The synthetic corpus generator (`scripts/generate_synthetic_corpus.py`) produces
mixed-architecture 64³ patches:

- UD continuous fibers (aligned, in-plane, angled)
- Short/discontinuous fibers with variable length and orientation concentration
- Woven/twill bundles
- Recycled/discontinuous fibers with variable diameter

Each patch is augmented with a random combination of XCT artifacts:

- Beam-hardening cupping
- Partial-volume blur
- Poisson/quantization noise
- Ring artifacts
- Contrast/gamma jitter

## Pre-training

Run multi-task pre-training on the synthetic corpus:

```bash
python scripts/generate_synthetic_corpus.py \
  --output data/synthetic_corpus \
  --n-samples 1000

python - <<'PY'
from fiber_tracer.training.fx_trainer import FiberTracerXTrainer

trainer = FiberTracerXTrainer(
    corpus_dir="data/synthetic_corpus",
    output_dir="models/fibertracer_x_pretrain",
    epochs=20,
    batch_size=2,
    device="auto",
    features=(16, 32, 64),
)
trainer.train("fx-pretrain-001")
PY
```

The trainer jointly optimizes segmentation (Focal + Dice) and orientation
regression (MSE + Frobenius) losses while sharing the encoder.

## Fine-tuning / adapters

After pre-training, the shared encoder is frozen (or kept at a very low learning
rate) and task adapters are fine-tuned on real labeled or pseudo-labeled data.

```python
from fiber_tracer.training.models import FiberTracerX

model = FiberTracerX(
    tasks={"segment": {"out_channels": 3}},
    features=(16, 32, 64),
)
model.load_state_dict(checkpoint["model_state_dict"])

# Freeze encoder, train adapter.
for param in model.encoder.parameters():
    param.requires_grad = False
```

## Benchmarking

Evaluate a checkpoint on the synthetic corpus validation set:

```bash
python scripts/run_benchmark.py \
  --checkpoint models/fibertracer_x_pretrain/checkpoint.pt \
  --corpus data/synthetic_corpus \
  --task segment \
  --output results/benchmark.json

python scripts/generate_leaderboard.py \
  --results results/benchmark.json \
  --output docs/BENCHMARK_LEADERBOARD.md
```

See [`BENCHMARK_LEADERBOARD.md`](BENCHMARK_LEADERBOARD.md) for current numbers.

## MPS / Apple Silicon notes

- Keep patch sizes and feature channels modest (e.g., 64³ patches,
  `features=(16, 32, 64)`).
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to route unsupported 3D ops to CPU.
- Validate every new architecture with a single forward/backward smoke test on
  MPS before committing.

## Roadmap

- [x] Synthetic phantom architectures + XCT domain randomization
- [x] Shared encoder + segmentation/orientation adapters
- [x] Multi-task synthetic pre-training trainer
- [x] Volume-level split option and staged pseudo-labeling
- [x] Benchmark harness + leaderboard generator
- [ ] YOLO bundle-detection adapter for woven/twill composites
- [ ] Void/crack segmentation adapter
- [ ] Fine-tuning scripts for real datasets (GF-PA66, DTU pultruded CFRP, IVW)
- [ ] Domain adaptation / DANN when synthetic-real gap is large
- [ ] Release v3.4.0-alpha with FiberTracer-X model zoo
