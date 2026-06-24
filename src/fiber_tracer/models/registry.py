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
        if not Path(path).exists():
            raise FileNotFoundError(f"model path does not exist: {path}")
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
