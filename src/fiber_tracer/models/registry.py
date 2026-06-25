"""Local model registry backed by a JSON manifest."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from fiber_tracer.utils.paths import get_config_dir

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    model_id: str
    name: str
    architecture: str
    source: str
    path: str
    version: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
            data = cast(dict[str, Any], json.loads(self.manifest_path.read_text()))
            if "models" not in data:
                return self._default_manifest()
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load model manifest from %s (%s). Falling back to default manifest.",
                self.manifest_path,
                exc.__class__.__name__,
            )
            return self._default_manifest()

    def _save(self, manifest: dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(self.manifest_path)

    def _default_manifest(self) -> dict[str, Any]:
        return {
            "version": 1,
            "default_model_id": "unet-v3.2",
            "models": [
                asdict(
                    ModelEntry(
                        model_id="unet-v3.2",
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
        entries: list[ModelEntry] = []
        for raw in self._manifest["models"]:
            try:
                entries.append(ModelEntry(**raw))
            except TypeError as exc:
                if isinstance(raw, dict):
                    model_id = raw.get("model_id", "<unknown>")
                else:
                    model_id = "<unknown>"
                raise ValueError(f"malformed model entry for {model_id!r}") from exc
        return entries

    def get_model(self, model_id: str) -> ModelEntry | None:
        for m in self.list_models():
            if m.model_id == model_id:
                return m
        return None

    def get_default(self) -> ModelEntry | None:
        default_id = self._manifest.get("default_model_id")
        return self.get_model(default_id) if default_id else None

    def add_model(
        self,
        model_id: str,
        name: str,
        path: str,
        architecture: str = "unet3d",
        version: str = "unknown",
        description: str = "",
        tags: list[str] | None = None,
    ) -> ModelEntry:
        if self.get_model(model_id) is not None:
            raise ValueError(f"model {model_id!r} already exists")
        if not Path(path).exists():
            raise FileNotFoundError(f"model path does not exist: {path}")
        entry = ModelEntry(
            model_id=model_id,
            name=name,
            architecture=architecture,
            source="local",
            path=path,
            version=version,
            description=description,
            tags=tags or [],
        )
        updated = deepcopy(self._manifest)
        updated["models"].append(asdict(entry))
        self._save(updated)
        self._manifest = updated
        return entry

    def remove_model(self, model_id: str) -> None:
        if self._manifest.get("default_model_id") == model_id:
            raise ValueError("cannot remove the default model; change default first")
        updated = deepcopy(self._manifest)
        before = len(updated["models"])
        updated["models"] = [m for m in updated["models"] if m.get("model_id") != model_id]
        if len(updated["models"]) == before:
            raise KeyError(f"model {model_id!r} not found")
        self._save(updated)
        self._manifest = updated

    def set_default(self, model_id: str) -> None:
        if self.get_model(model_id) is None:
            raise KeyError(f"model {model_id!r} not found")
        updated = deepcopy(self._manifest)
        updated["default_model_id"] = model_id
        for m in updated["models"]:
            m["is_default"] = m.get("model_id") == model_id
        self._save(updated)
        self._manifest = updated
