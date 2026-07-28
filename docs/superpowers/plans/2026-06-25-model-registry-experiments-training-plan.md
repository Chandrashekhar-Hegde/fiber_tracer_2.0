# Model Registry, Experiments, and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the TUI Model Registry, Experiments, and Training screens to life with a local-first Python backend, CLI commands, and cross-platform install/verification scripts.

**Architecture:** Add small, focused Python modules (`models/registry.py`, `experiments/store.py`, `training/trainer.py`) backed by JSON files in `~/.config/fiber-tracer`. Expose them through new `fiber-tracer model|experiment|train` CLI subcommands. Extend the TUI bridge and placeholder screens to consume these commands. Ship `scripts/install.sh`, `scripts/verify.sh`, and CI that proves a fresh machine can install and pass all tests.

**Tech Stack:** Python 3.10+, PyTorch, Bun/Ink, JSON/JSONL, GitHub Actions, shell/PowerShell.

---

## File structure

| File | Responsibility |
|------|----------------|
| `src/fiber_tracer/utils/paths.py` | Cross-platform config/data directory resolver. |
| `src/fiber_tracer/models/__init__.py` | Package marker. |
| `src/fiber_tracer/models/registry.py` | `ModelEntry`, `ModelRegistry`, CRUD, default model. |
| `src/fiber_tracer/experiments/__init__.py` | Package marker. |
| `src/fiber_tracer/experiments/store.py` | `Experiment`, JSONL store, compare helper. |
| `src/fiber_tracer/training/checkpoint.py` | Save/load checkpoints with metadata. |
| `src/fiber_tracer/training/trainer.py` | `UNetTrainer` with progress emission. |
| `src/fiber_tracer/cli.py` | New `model`, `experiment`, `train` subcommands. |
| `tests/test_model_registry.py` | Registry tests. |
| `tests/test_experiment_store.py` | Experiment store tests. |
| `tests/test_trainer.py` | Trainer smoke test. |
| `tests/test_cli_model_experiment_train.py` | CLI tests for new commands. |
| `tui/src/types.ts` | Add `Model`, `Experiment`, `TrainingJob` types. |
| `tui/src/bridge.ts` | Add `listModels`, `listExperiments`, `startTraining`. |
| `tui/src/components/model-registry.tsx` | Live model list. |
| `tui/src/components/experiments.tsx` | Live experiment list. |
| `tui/src/components/training.tsx` | Launcher + progress. |
| `tui/src/app.tsx` | Wire new bridge calls and screens. |
| `tui/src/bridge.test.ts` / `app.test.tsx` | Tests for new bridge behavior. |
| `scripts/install.sh` | macOS/Linux installer. |
| `scripts/install.ps1` | Windows installer. |
| `scripts/verify.sh` | macOS/Linux verification. |
| `scripts/verify.ps1` | Windows verification. |
| `.github/workflows/install.yml` | CI install/verify on Ubuntu/macOS/Windows. |
| `docs/INSTALL.md` | Update with one-line installer. |
| `README.md` | Quick-start and TUI screens. |
| `docs/CLI_REFERENCE.md` | Regenerate with new commands. |
| `docs/MODEL_REGISTRY.md` | New user guide. |
| `CHANGELOG.md` | Add `[Unreleased]` entries. |
| `ROADMAP.md` | Mark items in progress/done. |

---

## Task 1: Config directory utility

**Files:**
- Create: `src/fiber_tracer/utils/paths.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
from fiber_tracer.utils.paths import get_config_dir

def test_config_dir_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    assert get_config_dir() == str(tmp_path)

def test_config_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "fiber-tracer"
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(target))
    assert get_config_dir() == str(target)
    assert target.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_paths.py -v
```

Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
"""Cross-platform path helpers."""
from __future__ import annotations

import os
from pathlib import Path


def get_config_dir() -> str:
    """Return the fiber-tracer configuration directory.

    Uses ``FIBER_TRACER_CONFIG_DIR`` if set, otherwise ``~/.config/fiber-tracer``
    on POSIX or ``~/AppData/Roaming/fiber-tracer`` on Windows.
    """
    if env_dir := os.environ.get("FIBER_TRACER_CONFIG_DIR"):
        path = Path(env_dir)
    elif os.name == "nt":
        path = Path.home() / "AppData" / "Roaming" / "fiber-tracer"
    else:
        path = Path.home() / ".config" / "fiber-tracer"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_paths.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/utils/paths.py tests/test_paths.py
git commit -m "feat(paths): add cross-platform config directory helper"
```

---

## Task 2: Model registry module

**Files:**
- Create: `src/fiber_tracer/models/__init__.py`, `src/fiber_tracer/models/registry.py`
- Modify: `src/fiber_tracer/utils/paths.py` (already done)
- Test: `tests/test_model_registry.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from fiber_tracer.models.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    return ModelRegistry()


def test_list_includes_default_model(registry):
    models = registry.list_models()
    assert any(m.id == "unet-v3.2" for m in models)


def test_add_and_remove_model(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        id="local-1",
        name="Local Model",
        path=str(path),
        architecture="unet3d",
    )
    assert registry.get_model("local-1").name == "Local Model"
    registry.remove_model("local-1")
    assert registry.get_model("local-1") is None


def test_set_default(registry):
    registry.set_default("unet-v3.2")
    assert registry.get_default().id == "unet-v3.2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_model_registry.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
"""Local model registry backed by a JSON manifest."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fiber_tracer.utils.paths import get_config_dir


@dataclass
class ModelEntry:
    id: str
    name: str
    architecture: str
    source: str
    path: str
    version: str = "unknown"
    created_at: str = ""
    tags: list[str] = field(default_factory=list)
    description: str = ""
    status: str = "ready"
    is_default: bool = False


class ModelRegistry:
    """Read and write the model manifest at ``~/.config/fiber-tracer/models.json``."""

    def __init__(self, config_dir: str | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path(get_config_dir())
        self.manifest_path = self.config_dir / "models.json"
        self._manifest = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return self._default_manifest()
        try:
            data = json.loads(self.manifest_path.read_text())
            if "models" not in data:
                return self._default_manifest()
            return data
        except (json.JSONDecodeError, OSError):
            return self._default_manifest()

    def _save(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest, indent=2))
        tmp.replace(self.manifest_path)

    def _default_manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "default_model_id": "unet-v3.2",
            "models": [
                asdict(
                    ModelEntry(
                        id="unet-v3.2",
                        name="Fiber U-Net v3.2",
                        architecture="unet3d",
                        source="bundled",
                        path="models/fiber_unet_v2_full.pt",
                        version="3.2.0",
                        description="Default regime-aware 3D U-Net.",
                        is_default=True,
                    )
                )
            ],
        }

    def list_models(self) -> list[ModelEntry]:
        return [ModelEntry(**m) for m in self._manifest["models"]]

    def get_model(self, model_id: str) -> ModelEntry | None:
        for m in self.list_models():
            if m.id == model_id:
                return m
        return None

    def get_default(self) -> ModelEntry | None:
        default_id = self._manifest.get("default_model_id")
        return self.get_model(default_id) if default_id else None

    def add_model(
        self,
        id: str,
        name: str,
        path: str,
        architecture: str = "unet3d",
        version: str = "unknown",
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelEntry:
        if self.get_model(id) is not None:
            raise ValueError(f"model {id!r} already exists")
        entry = ModelEntry(
            id=id,
            name=name,
            architecture=architecture,
            source="local",
            path=path,
            version=version,
            description=description,
            tags=tags or [],
        )
        self._manifest["models"].append(asdict(entry))
        self._save()
        return entry

    def remove_model(self, model_id: str) -> None:
        if self._manifest.get("default_model_id") == model_id:
            raise ValueError("cannot remove the default model; change default first")
        before = len(self._manifest["models"])
        self._manifest["models"] = [m for m in self._manifest["models"] if m["id"] != model_id]
        if len(self._manifest["models"]) == before:
            raise KeyError(f"model {model_id!r} not found")
        self._save()

    def set_default(self, model_id: str) -> None:
        if self.get_model(model_id) is None:
            raise KeyError(f"model {model_id!r} not found")
        self._manifest["default_model_id"] = model_id
        for m in self._manifest["models"]:
            m["is_default"] = m["id"] == model_id
        self._save()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_model_registry.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/models tests/test_model_registry.py
git commit -m "feat(models): add local model registry"
```

---

## Task 3: Experiment store module

**Files:**
- Create: `src/fiber_tracer/experiments/__init__.py`, `src/fiber_tracer/experiments/store.py`
- Test: `tests/test_experiment_store.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import pytest

from fiber_tracer.experiments.store import ExperimentStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    return ExperimentStore()


def test_create_and_list(store):
    exp = store.create(name="test", type="train", model_id="unet-v3.2", dataset="/data")
    assert exp.status == "pending"
    listed = store.list_experiments()
    assert len(listed) == 1
    assert listed[0].id == exp.id


def test_update_and_compare(store):
    a = store.create(name="a", type="train", model_id="m", dataset="d")
    b = store.create(name="b", type="train", model_id="m", dataset="d")
    store.update(a.id, status="completed", metrics={"dice": 0.9})
    store.update(b.id, status="completed", metrics={"dice": 0.7})
    comparison = store.compare([a.id, b.id], metric="dice")
    assert comparison[a.id] == 0.9
    assert comparison[b.id] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_experiment_store.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
"""Experiment tracking backed by a JSONL file."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fiber_tracer.utils.paths import get_config_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"exp-{today}-{uuid.uuid4().hex[:6]}"


@dataclass
class Experiment:
    id: str
    name: str
    type: str
    model_id: str
    dataset: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    artifact_dir: str = ""
    error_message: str = ""


class ExperimentStore:
    """Read and write experiment records at ``~/.config/fiber-tracer/experiments.jsonl``."""

    def __init__(self, config_dir: str | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path(get_config_dir())
        self.store_path = self.config_dir / "experiments.jsonl"

    def _read_all(self) -> list[Experiment]:
        if not self.store_path.exists():
            return []
        experiments: list[Experiment] = []
        for line in self.store_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                experiments.append(Experiment(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
        return experiments

    def _write_all(self, experiments: list[Experiment]) -> None:
        tmp = self.store_path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(asdict(e)) + "\n" for e in experiments)
        )
        tmp.replace(self.store_path)

    def create(
        self,
        name: str,
        type: str,
        model_id: str,
        dataset: str,
        config_snapshot: dict[str, Any] | None = None,
        artifact_dir: str = "",
    ) -> Experiment:
        experiment = Experiment(
            id=_generate_id(),
            name=name,
            type=type,
            model_id=model_id,
            dataset=dataset,
            config_snapshot=config_snapshot or {},
            artifact_dir=artifact_dir,
        )
        experiments = self._read_all()
        experiments.append(experiment)
        self._write_all(experiments)
        return experiment

    def update(self, experiment_id: str, **kwargs: Any) -> Experiment | None:
        experiments = self._read_all()
        for i, exp in enumerate(experiments):
            if exp.id == experiment_id:
                for key, value in kwargs.items():
                    if hasattr(exp, key):
                        setattr(exp, key, value)
                experiments[i] = exp
                self._write_all(experiments)
                return exp
        return None

    def list_experiments(self) -> list[Experiment]:
        return list(reversed(self._read_all()))

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        for exp in self._read_all():
            if exp.id == experiment_id:
                return exp
        return None

    def compare(self, experiment_ids: list[str], metric: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp is None:
                continue
            value = exp.metrics.get(metric)
            if isinstance(value, list) and value:
                value = value[-1]
            result[exp_id] = value
        return result
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_experiment_store.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/experiments tests/test_experiment_store.py
git commit -m "feat(experiments): add JSONL experiment store"
```

---

## Task 4: Training checkpoint helper

**Files:**
- Create: `src/fiber_tracer/training/checkpoint.py`
- Test: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
import tempfile
from pathlib import Path

import torch

from fiber_tracer.training.checkpoint import save_checkpoint, load_checkpoint
from fiber_tracer.backends.unet3d import UNet3D


def test_save_and_load_checkpoint():
    model = UNet3D(features=(8, 16))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, model, metadata={"epoch": 2})
        loaded = load_checkpoint(path)
        assert loaded["metadata"]["epoch"] == 2
        assert "model_state_dict" in loaded
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_checkpoint.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
"""Checkpoint helpers with metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state dict and metadata to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint produced by ``save_checkpoint``."""
    return torch.load(path, map_location="cpu", weights_only=False)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_checkpoint.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(training): add checkpoint save/load helper"
```

---

## Task 5: Reusable training trainer

**Files:**
- Create: `src/fiber_tracer/training/trainer.py`
- Test: `tests/test_trainer.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from fiber_tracer.training.trainer import UNetTrainer
from fiber_tracer.experiments.store import ExperimentStore


def test_trainer_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path / "config"))
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    registry = {"sources": [{"patch_dir": "patches"}]}
    (dataset_dir / "datasets.json").write_text(json.dumps(registry))
    patch_dir = dataset_dir / "patches"
    patch_dir.mkdir()
    for i in range(4):
        np.savez(
            patch_dir / f"patch_{i}.npz",
            volume=np.random.rand(32, 32, 32).astype(np.float32),
            mask=(np.random.rand(32, 32, 32) > 0.5).astype(np.float32),
        )

    store = ExperimentStore()
    exp = store.create(name="smoke", type="train", model_id="unet-v3.2", dataset=str(dataset_dir))
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    trainer = UNetTrainer(
        dataset_dir=str(dataset_dir),
        output_dir=str(artifact_dir),
        epochs=1,
        batch_size=2,
        device="cpu",
        features=(8, 16),
    )
    metrics = trainer.train(experiment_id=exp.id)
    assert "train_loss" in metrics
    assert (artifact_dir / "checkpoint.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_trainer.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write implementation**

```python
"""Reusable 3D U-Net trainer with JSON progress emission."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.experiments.store import ExperimentStore
from fiber_tracer.training.checkpoint import save_checkpoint
from fiber_tracer.training.dataset import FiberVolumeDataset, numpy_collate


class BCEDiceLoss(nn.Module):
    """Combined binary cross-entropy and Dice loss."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


class UNetTrainer:
    """Train a 3D U-Net on a ``FiberVolumeDataset``."""

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-3,
        val_fraction: float = 0.1,
        device: str = "auto",
        features: tuple[int, ...] = (8, 16, 32),
        seed: int = 42,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_fraction = val_fraction
        self.seed = seed
        self.features = features
        self.progress_callback = progress_callback

        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _emit(self, stage: str, percent: float, message: str, metrics: dict[str, Any] | None = None) -> None:
        payload = {
            "stage": stage,
            "percent": round(percent, 2),
            "message": message,
            "metrics": metrics or {},
        }
        if os.environ.get("FIBER_TRACER_JSON_PROGRESS"):
            print(json.dumps(payload), flush=True)
        if self.progress_callback:
            self.progress_callback(payload)

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        registry = self.dataset_dir / "datasets.json"
        train_set = FiberVolumeDataset(
            registry_path=registry,
            processed_root=self.dataset_dir,
            split="train",
            val_fraction=self.val_fraction,
            augment=True,
            seed=self.seed,
        )
        val_set = FiberVolumeDataset(
            registry_path=registry,
            processed_root=self.dataset_dir,
            split="val",
            val_fraction=self.val_fraction,
            augment=False,
            seed=self.seed,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
            num_workers=0,
        )
        return train_loader, val_loader

    def _numpy_to_tensor(self, batch: tuple[np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        volumes, masks = batch
        return (
            torch.from_numpy(volumes).float().to(self.device),
            torch.from_numpy(masks).float().to(self.device),
        )

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        model.eval()
        criterion = BCEDiceLoss().to(self.device)
        total_loss = 0.0
        total_dice = 0.0
        n = 0
        for batch in loader:
            inputs, targets = self._numpy_to_tensor(batch)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            pred_flat = outputs.view(outputs.size(0), -1)
            target_flat = targets.view(targets.size(0), -1)
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            total_dice += dice.mean().item()
            n += 1
        return {
            "loss": total_loss / max(n, 1),
            "dice": total_dice / max(n, 1),
        }

    def train(self, experiment_id: str) -> dict[str, Any]:
        """Run training and return final metrics."""
        store = ExperimentStore()
        store.update(experiment_id, status="running")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = self._build_loaders()

        model = UNet3D(in_channels=1, out_channels=1, features=self.features).to(self.device)
        criterion = BCEDiceLoss().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_dice = -1.0
        history: dict[str, Any] = {"train_loss": [], "val_loss": [], "val_dice": []}

        try:
            for epoch in range(1, self.epochs + 1):
                model.train()
                epoch_loss = 0.0
                for batch in train_loader:
                    inputs, targets = self._numpy_to_tensor(batch)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_loss = epoch_loss / max(len(train_loader), 1)
                val_metrics = self._evaluate(model, val_loader)

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_dice"].append(val_metrics["dice"])

                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    save_checkpoint(
                        self.output_dir / "checkpoint.pt",
                        model,
                        metadata={
                            "epoch": epoch,
                            "val_dice": best_dice,
                            "features": self.features,
                        },
                    )

                percent = 100 * epoch / self.epochs
                self._emit(
                    "train",
                    percent,
                    f"epoch {epoch}/{self.epochs}",
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_dice": val_metrics["dice"],
                    },
                )

                store.update(
                    experiment_id,
                    metrics={k: v for k, v in history.items()},
                )

            final_metrics = {
                "train_loss": history["train_loss"][-1],
                "val_loss": history["val_loss"][-1],
                "val_dice": history["val_dice"][-1],
                "best_val_dice": best_dice,
            }
            store.update(
                experiment_id,
                status="completed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                metrics=final_metrics,
                artifact_dir=str(self.output_dir),
            )
            self._emit("complete", 100, "Training complete", final_metrics)
            return final_metrics

        except Exception as exc:
            store.update(
                experiment_id,
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error_message=str(exc),
            )
            self._emit("error", 0, f"Training failed: {exc}", {})
            raise
```

(Note: add `from datetime import datetime, timezone` at the top.)

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_trainer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/training/trainer.py tests/test_trainer.py
git commit -m "feat(training): add reusable UNetTrainer with JSON progress"
```

---

## Task 6: CLI subcommands

**Files:**
- Modify: `src/fiber_tracer/cli.py`
- Test: `tests/test_cli_model_experiment_train.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "fiber_tracer.cli", *args],
        capture_output=True,
        text=True,
    )


def test_model_list():
    result = run_cli("model", "list")
    assert result.returncode == 0
    assert "unet-v3.2" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_cli_model_experiment_train.py::test_model_list -v
```

Expected: FAIL (`unrecognized arguments: model`).

- [ ] **Step 3: Modify CLI**

Add to `src/fiber_tracer/cli.py`:

```python
def _build_model_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("model", help="Manage segmentation models")
    model_sub = parser.add_subparsers(dest="model_command")

    list_p = model_sub.add_parser("list", help="List registered models")
    list_p.set_defaults(func=_run_model_list)

    add_p = model_sub.add_parser("add", help="Add a local model")
    add_p.add_argument("--id", required=True)
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--path", required=True)
    add_p.add_argument("--architecture", default="unet3d")
    add_p.add_argument("--version", default="unknown")
    add_p.set_defaults(func=_run_model_add)

    remove_p = model_sub.add_parser("remove", help="Remove a model")
    remove_p.add_argument("id")
    remove_p.set_defaults(func=_run_model_remove)

    default_p = model_sub.add_parser("set-default", help="Set the default model")
    default_p.add_argument("id")
    default_p.set_defaults(func=_run_model_set_default)


def _run_model_list(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    default = registry.get_default()
    for m in registry.list_models():
        marker = " (default)" if default and m.id == default.id else ""
        print(f"{m.id}: {m.name} [{m.source}]{marker}")
    return 0


def _run_model_add(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.add_model(
        id=args.id,
        name=args.name,
        path=args.path,
        architecture=args.architecture,
        version=args.version,
    )
    print(f"Added model {args.id}")
    return 0


def _run_model_remove(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.remove_model(args.id)
    print(f"Removed model {args.id}")
    return 0


def _run_model_set_default(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    registry.set_default(args.id)
    print(f"Default model set to {args.id}")
    return 0


def _build_experiment_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("experiment", help="Manage experiments")
    exp_sub = parser.add_subparsers(dest="experiment_command")

    list_p = exp_sub.add_parser("list", help="List experiments")
    list_p.set_defaults(func=_run_experiment_list)

    show_p = exp_sub.add_parser("show", help="Show experiment details")
    show_p.add_argument("id")
    show_p.set_defaults(func=_run_experiment_show)

    compare_p = exp_sub.add_parser("compare", help="Compare experiments by metric")
    compare_p.add_argument("ids", nargs="+")
    compare_p.add_argument("--metric", default="val_dice")
    compare_p.set_defaults(func=_run_experiment_compare)


def _run_experiment_list(args: argparse.Namespace) -> int:
    import json

    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    for exp in store.list_experiments():
        print(f"{exp.id} {exp.name} {exp.status}")
    return 0


def _run_experiment_show(args: argparse.Namespace) -> int:
    import json

    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    exp = store.get_experiment(args.id)
    if exp is None:
        print(f"Experiment {args.id} not found", file=sys.stderr)
        return 1
    print(json.dumps(exp.__dict__, indent=2))
    return 0


def _run_experiment_compare(args: argparse.Namespace) -> int:
    import json

    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    result = store.compare(args.ids, metric=args.metric)
    print(json.dumps(result, indent=2))
    return 0


def _build_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Train a segmentation model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-id", default="unet-v3.2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.set_defaults(func=_run_train)


def _run_train(args: argparse.Namespace) -> int:
    from fiber_tracer.experiments.store import ExperimentStore
    from fiber_tracer.models.registry import ModelRegistry
    from fiber_tracer.training.trainer import UNetTrainer

    registry = ModelRegistry()
    model = registry.get_model(args.model_id)
    if model is None:
        print(f"Model {args.model_id} not found", file=sys.stderr)
        return 1

    store = ExperimentStore()
    exp = store.create(
        name=args.name or f"train-{args.model_id}",
        type="train",
        model_id=args.model_id,
        dataset=args.dataset_dir,
        config_snapshot={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
        },
        artifact_dir=args.output_dir,
    )

    trainer = UNetTrainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    trainer.train(experiment_id=exp.id)
    return 0
```

Then wire the parsers in `_build_parser`:

```python
    _build_model_parser(subparsers)
    _build_experiment_parser(subparsers)
    _build_train_parser(subparsers)
```

And dispatch in `main`:

```python
    if args.command == "model":
        return args.func(args)
    if args.command == "experiment":
        return args.func(args)
    if args.command == "train":
        return args.func(args)
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_cli_model_experiment_train.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fiber_tracer/cli.py tests/test_cli_model_experiment_train.py
git commit -m "feat(cli): add model, experiment, and train subcommands"
```

---

## Task 7: TUI types

**Files:**
- Modify: `tui/src/types.ts`

- [ ] **Step 1: Add types**

Append to `tui/src/types.ts`:

```typescript
export interface Model {
  id: string;
  name: string;
  architecture: string;
  source: string;
  path: string;
  version: string;
  createdAt: string;
  tags: string[];
  description: string;
  status: string;
  isDefault: boolean;
}

export interface Experiment {
  id: string;
  name: string;
  type: string;
  modelId: string;
  dataset: string;
  configSnapshot: Record<string, unknown>;
  status: string;
  metrics: Record<string, unknown>;
  startedAt: string;
  finishedAt?: string;
  artifactDir: string;
  errorMessage?: string;
}

export interface TrainingOptions {
  datasetDir: string;
  modelId: string;
  outputDir: string;
  name: string;
  epochs: number;
  batchSize: number;
  lr: number;
  device: string;
}
```

- [ ] **Step 2: Commit**

```bash
git add tui/src/types.ts
git commit -m "feat(tui): add Model, Experiment, TrainingOptions types"
```

---

## Task 8: TUI bridge extensions

**Files:**
- Modify: `tui/src/bridge.ts`
- Test: `tui/src/bridge.test.ts`

- [ ] **Step 1: Add bridge functions**

Replace the contents of `tui/src/bridge.ts` with an extended version that keeps `runAnalysis` and adds:

```typescript
export async function listModels(): Promise<Model[]> {
  return new Promise((resolve, reject) => {
    const proc = spawn("fiber-tracer", ["model", "list", "--json"], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    let stdout = "";
    let stderr = "";
    proc.stdout!.on("data", (d) => (stdout += d.toString()));
    proc.stderr!.on("data", (d) => (stderr += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0) return reject(new Error(stderr || stdout));
      try {
        resolve(JSON.parse(stdout) as Model[]);
      } catch {
        reject(new Error("Invalid JSON from model list"));
      }
    });
  });
}

export async function listExperiments(): Promise<Experiment[]> {
  return new Promise((resolve, reject) => {
    const proc = spawn("fiber-tracer", ["experiment", "list", "--json"], {
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    let stdout = "";
    let stderr = "";
    proc.stdout!.on("data", (d) => (stdout += d.toString()));
    proc.stderr!.on("data", (d) => (stderr += d.toString()));
    proc.on("close", (code) => {
      if (code !== 0) return reject(new Error(stderr || stdout));
      try {
        resolve(JSON.parse(stdout) as Experiment[]);
      } catch {
        reject(new Error("Invalid JSON from experiment list"));
      }
    });
  });
}

export async function startTraining(
  options: TrainingOptions,
  callbacks: BridgeOptions = {}
): Promise<BridgeResult> {
  return new Promise((resolve) => {
    const proc = spawn("fiber-tracer", [
      "train",
      "--dataset-dir", options.datasetDir,
      "--model-id", options.modelId,
      "--output-dir", options.outputDir,
      "--epochs", String(options.epochs),
      "--batch-size", String(options.batchSize),
      "--lr", String(options.lr),
      "--device", options.device,
      "--name", options.name,
    ], {
      env: { ...process.env, PYTHONUNBUFFERED: "1", FIBER_TRACER_JSON_PROGRESS: "1" },
    });

    let stdout = "";
    let stderr = "";
    proc.stdout!.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      if (callbacks.onLog) callbacks.onLog(text.trim());
      const event = parseProgress(text);
      if (event && callbacks.onProgress) callbacks.onProgress(event);
    });
    proc.stderr!.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      if (callbacks.onLog) callbacks.onLog(text.trim());
    });
    proc.on("error", (err) => resolve({ success: false, outputDir: options.outputDir, error: String(err) }));
    proc.on("close", (code) => {
      if (code === 0) {
        resolve({ success: true, outputDir: options.outputDir });
      } else {
        resolve({ success: false, outputDir: options.outputDir, error: stderr || stdout });
      }
    });
  });
}
```

Also update `runAnalysis` to return `summary` from the output directory if available, or keep as-is.

To support `--json` in the Python CLI, add `--json` flags to `model list` and `experiment list` that output JSON instead of text. Do this in `cli.py`.

- [ ] **Step 2: Update tests**

Add tests in `tui/src/bridge.test.ts` for `listModels` error path (missing CLI) similar to existing `runAnalysis` test.

- [ ] **Step 3: Run tests**

```bash
cd tui && bun run typecheck && bun test
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tui/src/bridge.ts tui/src/bridge.test.ts
git commit -m "feat(tui): extend bridge for models, experiments, and training"
```

---

## Task 9: Live TUI screens

**Files:**
- Modify: `tui/src/components/model-registry.tsx`, `tui/src/components/experiments.tsx`, `tui/src/components/training.tsx`, `tui/src/app.tsx`

- [ ] **Step 1: Model Registry screen**

```tsx
import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { Model } from "../types";
import { listModels } from "../bridge";

interface ModelRegistryProps {
  theme: Theme;
}

export function ModelRegistry({ theme }: ModelRegistryProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    listModels().then(setModels).catch((e) => setError(String(e)));
  }, []);

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Model Registry</Text>
      {error && <Text color={theme.danger}>{error}</Text>}
      {models.map((m) => (
        <Text key={m.id} color={theme.foreground}>
          {m.isDefault ? "★" : " "} {m.name} ({m.id}) — {m.status}
        </Text>
      ))}
      <Text color={theme.muted}>Press i to import a model (future).</Text>
    </Box>
  );
}
```

- [ ] **Step 2: Experiments screen**

```tsx
import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { Experiment } from "../types";
import { listExperiments } from "../bridge";

interface ExperimentsProps {
  theme: Theme;
}

export function Experiments({ theme }: ExperimentsProps) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    listExperiments().then(setExperiments).catch((e) => setError(String(e)));
  }, []);

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Experiments</Text>
      {error && <Text color={theme.danger}>{error}</Text>}
      {experiments.length === 0 && <Text color={theme.muted}>No experiments yet.</Text>}
      {experiments.map((e) => (
        <Text key={e.id} color={theme.foreground}>
          {e.status === "completed" ? "✓" : "•"} {e.name} — {e.status} ({e.modelId})
        </Text>
      ))}
    </Box>
  );
}
```

- [ ] **Step 3: Training screen**

```tsx
import React, { useState } from "react";
import { Box, Text } from "ink";
import type { Theme } from "../theme";
import type { ProgressEvent, TrainingOptions } from "../types";
import { startTraining } from "../bridge";

interface TrainingProps {
  theme: Theme;
}

export function Training({ theme }: TrainingProps) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string>("");

  const handleStart = () => {
    setRunning(true);
    setError("");
    const options: TrainingOptions = {
      datasetDir: "./data/patches",
      modelId: "unet-v3.2",
      outputDir: "./experiments/quick-test",
      name: "Quick test",
      epochs: 2,
      batchSize: 2,
      lr: 1e-3,
      device: "auto",
    };
    startTraining(options, {
      onProgress: setProgress,
      onLog: (line) => console.log(line),
    })
      .then((res) => {
        if (!res.success) setError(res.error || "Training failed");
      })
      .catch((e) => setError(String(e)))
      .finally(() => setRunning(false));
  };

  return (
    <Box flexDirection="column">
      <Text bold color={theme.accent}>Training</Text>
      {running && progress && (
        <Text color={theme.foreground}>{progress.stage}: {progress.percent}% — {progress.message}</Text>
      )}
      {error && <Text color={theme.danger}>{error}</Text>}
      <Text color={theme.muted}>Press s to start a quick training run.</Text>
    </Box>
  );
}
```

- [ ] **Step 4: Wire screens in `tui/src/app.tsx`**

Pass `theme` to the screens (already done). Add keyboard handlers for `s` when on the Training screen to call `handleStart`. Keep the existing section routing.

- [ ] **Step 5: Run tests**

```bash
cd tui && bun run typecheck && bun test
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tui/src/components tui/src/app.tsx
git commit -m "feat(tui): wire live data into Model Registry, Experiments, and Training"
```

---

## Task 10: Install scripts

**Files:**
- Create: `scripts/install.sh`, `scripts/install.ps1`, `scripts/verify.sh`, `scripts/verify.ps1`

- [ ] **Step 1: Write `scripts/install.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"

echo "==> Fiber Tracer installer"

# Check Python version.
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version=$("$cmd" --version 2>&1 | awk '{print $2}')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python >=3.10 is required. Please install it and re-run."
    exit 1
fi

echo "==> Using Python $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# Install Bun if missing.
if ! command -v bun >/dev/null 2>&1; then
    echo "==> Installing Bun..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
fi

echo "==> Creating virtual environment..."
"$PYTHON_CMD" -m venv "$REPO_ROOT/.venv"

# shellcheck source=/dev/null
source "$REPO_ROOT/.venv/bin/activate"

echo "==> Installing Python package..."
pip install -U pip
pip install -e "$REPO_ROOT[dev]"

echo "==> Installing TUI dependencies..."
cd "$REPO_ROOT/tui"
bun install

echo "==> Installation complete."
echo "Run: source .venv/bin/activate && bun run dev"
```

- [ ] **Step 2: Write `scripts/verify.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Verifying Python package"
fiber-tracer --version

echo "==> Running Python tests"
pytest

echo "==> Running TUI checks"
cd tui
bun run typecheck
bun test

echo "==> Verification complete."
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/install.sh scripts/verify.sh
```

- [ ] **Step 4: Write PowerShell variants**

Create `scripts/install.ps1` and `scripts/verify.ps1` with equivalent logic for Windows (check Python, install Bun, create venv, pip install, bun install, run tests).

- [ ] **Step 5: Test locally**

```bash
./scripts/install.sh
./scripts/verify.sh
```

Expected: install succeeds, verify passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/install.sh scripts/verify.sh scripts/install.ps1 scripts/verify.ps1
git commit -m "chore(scripts): add cross-platform install and verify scripts"
```

---

## Task 11: CI install workflow

**Files:**
- Create: `.github/workflows/install.yml`

- [ ] **Step 1: Write workflow**

```yaml
name: Install & Verify

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  install-verify:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install (Unix)
        if: runner.os != 'Windows'
        run: ./scripts/install.sh
        shell: bash

      - name: Verify (Unix)
        if: runner.os != 'Windows'
        run: ./scripts/verify.sh
        shell: bash

      - name: Install (Windows)
        if: runner.os == 'Windows'
        run: .\scripts\install.ps1
        shell: pwsh

      - name: Verify (Windows)
        if: runner.os == 'Windows'
        run: .\scripts\verify.ps1
        shell: pwsh
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/install.yml
git commit -m "ci: add install-and-verify workflow for Ubuntu, macOS, and Windows"
```

---

## Task 12: Documentation updates

**Files:**
- Create: `docs/MODEL_REGISTRY.md`
- Modify: `README.md`, `docs/INSTALL.md`, `docs/CLI_REFERENCE.md`, `CHANGELOG.md`, `ROADMAP.md`

- [ ] **Step 1: Write `docs/MODEL_REGISTRY.md`**

Include:
- What the registry is and where it lives.
- `fiber-tracer model` commands with examples.
- `fiber-tracer experiment` commands with examples.
- `fiber-tracer train` quick-start.
- TUI screens overview.

- [ ] **Step 2: Update `README.md`**

Add a "Quick start" section:

```bash
curl -fsSL https://raw.githubusercontent.com/Chandrashekhar-Hegde/fiber_tracer_2.0/main/scripts/install.sh | bash
source .venv/bin/activate
bun run dev
```

Also add short sections for Model Registry, Experiments, and Training.

- [ ] **Step 3: Update `docs/INSTALL.md`**

Replace or augment with the one-line installer instructions and manual fallback steps.

- [ ] **Step 4: Regenerate `docs/CLI_REFERENCE.md`**

Run the CLI help generator or manually add new commands. Keep consistent with existing doc style.

- [ ] **Step 5: Update `CHANGELOG.md`**

Under `[Unreleased]` add:

```markdown
- Added: Local model registry (`fiber-tracer model`).
- Added: Experiment tracking store (`fiber-tracer experiment`).
- Added: `fiber-tracer train` command with live JSON progress.
- Added: Cross-platform `scripts/install.sh` and `scripts/verify.sh`.
- Added: GitHub Actions install-and-verify workflow for Ubuntu, macOS, and Windows.
- Updated: TUI Model Registry, Experiments, and Training screens now show live data.
```

- [ ] **Step 6: Update `ROADMAP.md`**

Mark TUI placeholder implementation and training CLI as done/in-progress for v3.3.0.

- [ ] **Step 7: Commit**

```bash
git add docs/MODEL_REGISTRY.md README.md docs/INSTALL.md docs/CLI_REFERENCE.md CHANGELOG.md ROADMAP.md
git commit -m "docs: add model registry, experiments, training, and install guide"
```

---

## Task 13: Final integration and push

- [ ] **Step 1: Run full local verification**

```bash
./scripts/verify.sh
```

Expected: Python tests pass, TUI typecheck and tests pass.

- [ ] **Step 2: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 3: Monitor CI**

Open `.github/workflows/install.yml` runs and ensure Ubuntu/macOS/Windows all pass.

---

## Self-review checklist

- [ ] Spec coverage: every section of the design doc maps to at least one task.
- [ ] No placeholders: every task has concrete code/commands.
- [ ] Type consistency: `ModelEntry`, `Experiment`, `UNetTrainer` signatures match CLI and TUI usage.
- [ ] Tests: each new module has a failing test written before implementation.
- [ ] Cross-platform: install/verify scripts and CI cover Unix and Windows.
- [ ] Docs: README, INSTALL, CLI_REFERENCE, CHANGELOG, ROADMAP updated.


## Implementation notes

This section records intentional deviations between the original plan text and the merged implementation.

- **Model identifier field:** The plan uses `id` for `ModelEntry`. The implementation uses `model_id` to avoid shadowing the built-in `id` function. The CLI JSON output uses `model_id`; the TUI bridge maps it to `Model.id`.
- **Experiment record:** The implementation adds a `history` field to `Experiment` for per-epoch metric lists, separate from the scalar `metrics` field.
- **Training dataset format:** The plan shows a simplified `datasets.json` with a single `patch_dir`. The implementation reuses the existing `FiberVolumeDataset`, which expects a `datasets.json` registry with a `sources` list of objects containing `patch_dir`.
- **TUI bridge config serialization:** The plan shows YAML config serialization for `runAnalysis`. The implementation writes JSON (`config.json`) because the Python `Config` parser accepts JSON and it avoids YAML quoting edge cases in the bridge.
- **Training checkpoint registration:** The plan did not explicitly require auto-registering the trained checkpoint. The implementation upserts the registry entry for `--model-id` after training and resolves registry IDs passed to `--model-path` in `fiber-tracer run`.
- **Install script:** The implementation additionally installs the `dev` extra (`pip install -e ".[dev]"`) so that verification can run the full test suite.
