# Design: Live Model Registry, Experiments, and Training

**Date:** 2026-06-25  
**Scope:** Turn the TUI placeholder screens (Model Registry, Experiments, Training) into real, local-first features backed by the existing Python engine.  
**Target release:** v3.3.0

## Goals

1. **Model Registry** — list, import, remove, and set default segmentation models.
2. **Experiments** — track and compare analysis and training runs.
3. **Training** — launch a training run from the CLI or TUI and stream live progress.
4. **Documentation** — keep README, CLI reference, roadmap, and changelog current.

## Non-goals

- Multi-user or remote model hosting.
- Distributed training.
- Integration with external experiment trackers (MLflow, W&B) in this phase.
- Replacing the existing `scripts/train_unet_*.py` examples immediately; they can remain as reference scripts.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           TUI (Ink)                             │
│  ModelRegistry  │  Experiments  │  Training  │  Run & Watch     │
└────────────┬─────────────────────┬─────────────┬──────────────────┘
             │                     │             │
             │ listModels()        │ listRuns()  │ startTraining()
             │                     │             │
┌────────────▼─────────────────────▼─────────────▼──────────────────┐
│                        Python backend                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   registry   │  │    store     │  │       trainer        │   │
│  │  (models.json)│  │(experiments.jsonl)│  │  (training/trainer.py)│   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                         CLI (typer/argparse)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| File | Responsibility |
|------|----------------|
| `src/fiber_tracer/models/registry.py` | CRUD + default management for the model manifest. |
| `src/fiber_tracer/experiments/store.py` | CRUD + compare for experiment records. |
| `src/fiber_tracer/training/trainer.py` | Reusable trainer that drives `UNet3D` and emits JSON progress. |
| `src/fiber_tracer/training/checkpoint.py` | Save/load checkpoints with metadata. |
| `src/fiber_tracer/cli.py` | New subcommands: `model`, `experiment`, `train`. |
| `tui/src/bridge.ts` | New bridge calls for model/experiment/training operations. |
| `tui/src/components/model-registry.tsx` | Live model list + import/default actions. |
| `tui/src/components/experiments.tsx` | Live experiment list + compare summary. |
| `tui/src/components/training.tsx` | Launcher + live progress for training. |

## Data models

### Model manifest (`~/.config/fiber-tracer/models.json`)

```json
{
  "version": 1,
  "default_model_id": "unet-v3.2",
  "models": [
    {
      "id": "unet-v3.2",
      "name": "Fiber U-Net v3.2",
      "architecture": "unet3d",
      "source": "bundled",
      "path": "models/fiber_unet_v2_full.pt",
      "version": "3.2.0",
      "created_at": "2026-06-25T12:00:00Z",
      "tags": ["production", "3d"],
      "description": "Default regime-aware 3D U-Net.",
      "status": "ready"
    }
  ]
}
```

Fields:
- `id` — stable identifier used by CLI/TUI.
- `name` — human-readable name.
- `architecture` — backend key (`unet3d`, `nnunet`, `classical`, etc.).
- `source` — `bundled`, `local`, or `remote`.
- `path` — absolute or repo-relative checkpoint path.
- `version` — model version string.
- `created_at` — ISO 8601 timestamp.
- `tags` — list of strings.
- `description` — short summary.
- `status` — `ready`, `missing`, `loading`, `error`.

### Experiment record (`~/.config/fiber-tracer/experiments.jsonl`)

Each line is a JSON object:

```json
{
  "id": "exp-20260625-001",
  "name": "U-Net fine-tune on HT3",
  "type": "train",
  "model_id": "unet-custom-001",
  "dataset": "/data/ht3_patches",
  "config_snapshot": {"epochs": 10, "batch_size": 4, "lr": 1e-4},
  "status": "completed",
  "metrics": {"train_loss": [0.5, 0.2, 0.1], "val_dice": [0.6, 0.8, 0.9]},
  "started_at": "2026-06-25T12:00:00Z",
  "finished_at": "2026-06-25T12:30:00Z",
  "artifact_dir": "./experiments/exp-20260625-001"
}
```

Fields:
- `id` — unique run ID (`exp-<YYYYMMDD>-<NNN>`).
- `name`, `type` (`train` or `analyze`).
- `model_id`, `dataset` — references.
- `config_snapshot` — full config at run start.
- `status` — `pending`, `running`, `completed`, `failed`, `cancelled`.
- `metrics` — arbitrary numeric dict/lists.
- `started_at`, `finished_at` — ISO 8601 timestamps.
- `artifact_dir` — directory containing checkpoints, logs, plots.

## CLI surface

### Model commands

```bash
fiber-tracer model list
fiber-tracer model add --id my-unet --name "My U-Net" --path ./my_unet.pt --architecture unet3d
fiber-tracer model remove my-unet
fiber-tracer model set-default my-unet
fiber-tracer model get-default
```

### Experiment commands

```bash
fiber-tracer experiment list
fiber-tracer experiment show exp-20260625-001
fiber-tracer experiment compare exp-20260625-001 exp-20260625-002
```

### Training command

```bash
fiber-tracer train \
  --dataset-dir ./data/patches \
  --model-id unet-v3.2 \
  --output-dir ./experiments/run-001 \
  --epochs 10 \
  --batch-size 4 \
  --lr 1e-4
```

## Training flow

1. CLI parses args and validates the dataset directory and model ID.
2. Create an experiment record with status `pending`.
3. Trainer loads the dataset, model, and optimizer.
4. Each epoch:
   - Train loop updates weights.
   - Validation loop computes metrics.
   - Append metrics to the experiment record.
   - Emit JSON progress line to stdout when `FIBER_TRACER_JSON_PROGRESS=1`.
5. On success: save checkpoint to `artifact_dir`, set status `completed`.
6. On failure: set status `failed`, record traceback, re-raise.

## Progress protocol

Extend the existing `FIBER_TRACER_JSON_PROGRESS` environment variable. Both pipeline and training emit newline-delimited JSON objects:

```json
{"stage":"train","epoch":2,"percent":20,"metrics":{"loss":0.12},"message":"epoch 2/10"}
{"stage":"complete","percent":100,"elapsedSeconds":120,"message":"Training complete"}
```

The TUI `bridge.ts` already parses JSON lines from stdout and can consume these objects directly.

## Error handling

- **Registry:** validate checkpoint exists and `torch.load` succeeds before adding. If validation fails, surface a clear error.
- **Store:** use atomic write (write to temp file, fsync, rename) to avoid corrupt JSONL files.
- **Training:** wrap the training loop in `try/except/finally`. On exception, update the experiment record with `status: failed` and `error_message`, then re-raise so the CLI exits non-zero and the TUI can show the error.
- **TUI bridge:** handle non-zero exit codes and parse stderr for human-readable messages.

## Testing strategy

- **Registry tests:** add/list/remove/default operations, malformed manifest fallback, missing checkpoint rejection.
- **Experiment store tests:** create/update/list/compare, corrupted line skipping, atomic write verification.
- **Trainer tests:** one-epoch smoke test on synthetic 32³ patches, checkpoint save/load, progress emission.
- **CLI tests:** invoke new subcommands and assert exit codes/outputs.
- **TUI tests:** mock bridge responses for `listModels`, `listExperiments`, and `startTraining`; assert component rendering.

## Documentation updates

- `docs/superpowers/specs/2026-06-25-model-registry-experiments-training-design.md` (this file)
- `docs/superpowers/plans/2026-06-25-model-registry-experiments-training-plan.md`
- `docs/MODEL_REGISTRY.md` — user guide for registry/experiments/training
- `README.md` — new sections for Model Registry & Experiments
- `docs/CLI_REFERENCE.md` — regenerate with new subcommands
- `CHANGELOG.md` — add under `[Unreleased]`
- `ROADMAP.md` — mark TUI placeholders and training CLI as in-progress/done

## Future-proofing

- `architecture` field leaves room for `nnunet`, `swin`, and classical backends.
- `source` field leaves room for remote model downloads.
- Experiment `type` field supports both `train` and `analyze` runs, so the analysis wizard can also record experiments.
- Abstract `ModelRegistry` and `ExperimentStore` interfaces allow swapping JSON files for SQLite or a remote API later.
