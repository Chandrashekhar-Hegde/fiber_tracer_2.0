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


def test_add_model_requires_existing_path(registry, tmp_path):
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        registry.add_model(
            id="missing",
            name="Missing Model",
            path=str(missing),
            architecture="unet3d",
        )


def test_atomic_write(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        id="atomic-1",
        name="Atomic Model",
        path=str(path),
        architecture="unet3d",
    )
    manifest_path = Path(registry.config_dir) / "models.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert "models" in data
    assert any(m["id"] == "atomic-1" for m in data["models"])
