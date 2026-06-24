# Model Registry, Experiments, and Training

The model registry, experiment tracking, and training CLI are implemented and available on the `main` branch; they will be included in the upcoming v3.3.0 release. Everything is file-based and stored under the user configuration directory, which keeps project directories clean and makes models reusable across analyses.

- **Model registry** — a catalog of local `.pt` checkpoints with metadata such as architecture, version, and description.
- **Experiments** — lightweight records that link a training run to a model ID, hyper-parameters, and final metrics.
- **Training CLI** — a quick way to start a U-Net training run from a dataset directory with JSON progress output.

All registry and experiment data lives in:

```text
~/.config/fiber-tracer/
├── models.json       # registered models and default model pointer
└── experiments.jsonl # training experiment records (JSON Lines)
```

On macOS this resolves to `~/.config/fiber-tracer` by default. If the directory does not exist, it is created automatically when you first run `model add`, `train`, or the TUI.

---

## Model registry

Registered models can be selected by ID from the CLI; a registry ID can be supplied anywhere `--model-path` expects a checkpoint path, and the CLI resolves it to the registered file. Models are also surfaced in the TUI Model Registry screen.

### `fiber-tracer model list`

Show all registered models. The default model is highlighted.

```bash
fiber-tracer model list
```

JSON output for scripting:

```bash
fiber-tracer model list --json
```

### `fiber-tracer model add`

Add a checkpoint to the registry.

```bash
fiber-tracer model add \
  --model-id fiber_unet_v2_full \
  --name "Production 3D U-Net" \
  --path models/fiber_unet_v2_full.pt \
  --architecture UNet3D \
  --version 3.2.0 \
  --description "Trained on 2,152 mixed synthetic + XCT patches"
```

| Flag | Required | Description |
|------|----------|-------------|
| `--model-id` | Yes | Short unique identifier used when selecting the model. |
| `--name` | Yes | Human-readable name. |
| `--path` | Yes | Absolute or relative path to the `.pt` checkpoint. |
| `--architecture` | No | Model architecture, e.g. `UNet3D`. |
| `--version` | No | Version string. |
| `--description` | No | Free-text description. |

### `fiber-tracer model set-default`

Choose the model that the TUI and omitted `--model-path` runs should prefer.

```bash
fiber-tracer model set-default fiber_unet_v2_full
```

### `fiber-tracer model remove`

Remove a model from the registry. This only deletes the catalog entry; the checkpoint file on disk is left untouched.

```bash
fiber-tracer model remove fiber_unet_v2_full
```

---

## Experiments

Training runs are recorded as experiments. Each experiment stores a model ID, hyper-parameters, and metrics such as final loss and validation Dice, making it easy to compare iterations over time.

### `fiber-tracer experiment list`

```bash
fiber-tracer experiment list
fiber-tracer experiment list --json
```

### `fiber-tracer experiment show`

Inspect a single experiment.

```bash
fiber-tracer experiment show exp-20260624-abc123
```

### `fiber-tracer experiment compare`

Compare two or more experiments by a metric.

```bash
fiber-tracer experiment compare exp-20260624-abc123 exp-20260625-def456 --metric val_dice
```

If `--metric` is omitted, the CLI selects a sensible default such as `val_loss`.

---

## Training CLI

The `train` subcommand is a thin wrapper around the synthetic-to-real U-Net training pipeline. It expects a dataset directory produced by `scripts/prepare_training_data.py` (or any directory containing `images` and `masks` subfolders in TIFF or NumPy format).

### Quick-start example

```bash
# 1. Prepare data
python scripts/download_datasets.py
python scripts/prepare_training_data.py \
  --n-synthetic 1000 \
  --n-patches-per-volume 64 \
  --output data/processed/training/

# 2. Train and register the result as an experiment
fiber-tracer train \
  --dataset-dir data/processed/training/ \
  --output-dir models/experiments/exp-001/ \
  --model-id fiber_unet_v3 \
  --name "v3 mixed training" \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-4 \
  --device auto
```

**Note:** Training requires the `ml` extra: `pip install -e ".[ml]"`.

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dataset-dir` | Yes | — | Directory with `images/` and `masks/` training patches. |
| `--output-dir` | Yes | — | Directory for checkpoints, logs, and experiment artifacts. |
| `--model-id` | No | generated | ID used to register the resulting model in the registry. |
| `--name` | No | generated | Experiment name. |
| `--epochs` | No | `20` | Training epochs. |
| `--batch-size` | No | `4` | Batch size. |
| `--lr` | No | `1e-4` | Learning rate. |
| `--val-fraction` | No | `0.2` | Fraction of data held out for validation. |
| `--device` | No | `auto` | `cpu`, `cuda`, `mps`, or `auto`. |
| `--features` | No | — | Optional feature flags for the training run. |

### JSON progress

When `train` is run from a terminal or orchestrator, it writes compact JSON progress lines to stdout so that logs can be parsed programmatically:

```json
{"epoch": 1, "train_loss": 0.421, "val_loss": 0.398, "val_dice": 0.72}
```

This format is consumed by the TUI Training screen to show live loss and metric curves.

---

## Terminal UI screens

The TUI (`cd tui && bun run dev`) exposes the new functionality through dedicated screens:

- **Model Registry** — browse registered models, set the default, add new checkpoints, and remove obsolete entries.
- **Experiments** — list training experiments, inspect hyper-parameters and metrics, and compare runs.
- **Training** — launch a training run, watch live JSON progress, and see the resulting experiment recorded automatically.

These screens read from and write to the same `~/.config/fiber-tracer/` files as the CLI, so data is consistent whether you use the terminal UI or command-line tools.

---

## See also

- [`docs/CLI_REFERENCE.md`](CLI_REFERENCE.md) — full command reference including `model`, `experiment`, and `train`.
- [`docs/MODEL_CARD.md`](MODEL_CARD.md) — model card for the production U-Net.
- [`docs/parameter_guide.md`](parameter_guide.md) — configuration and parameter reference.
- [`docs/developer_guide.md`](developer_guide.md) — development setup and contribution workflow.
