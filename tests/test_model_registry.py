import json
from pathlib import Path

import pytest

from fiber_tracer.models.registry import ModelEntry, ModelRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    return ModelRegistry()


def test_list_includes_default_model(registry):
    models = registry.list_models()
    assert any(m.model_id == "unet-v3.2" for m in models)


def test_add_and_remove_model(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        model_id="local-1",
        name="Local Model",
        path=str(path),
        architecture="unet3d",
    )
    assert registry.get_model("local-1").name == "Local Model"
    registry.remove_model("local-1")
    assert registry.get_model("local-1") is None


def test_set_default(registry):
    registry.set_default("unet-v3.2")
    assert registry.get_default().model_id == "unet-v3.2"


def test_add_model_requires_existing_path(registry, tmp_path):
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        registry.add_model(
            model_id="missing",
            name="Missing Model",
            path=str(missing),
            architecture="unet3d",
        )


def test_atomic_write(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        model_id="atomic-1",
        name="Atomic Model",
        path=str(path),
        architecture="unet3d",
    )
    manifest_path = Path(registry.config_dir) / "models.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert "models" in data
    assert any(m["model_id"] == "atomic-1" for m in data["models"])


def test_add_duplicate_raises(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        model_id="dup-1",
        name="Duplicate Model",
        path=str(path),
    )
    with pytest.raises(ValueError, match="already exists"):
        registry.add_model(
            model_id="dup-1",
            name="Duplicate Model",
            path=str(path),
        )


def test_remove_default_raises(registry):
    with pytest.raises(ValueError, match="cannot remove the default model"):
        registry.remove_model("unet-v3.2")


def test_remove_missing_raises(registry):
    with pytest.raises(KeyError, match="not found"):
        registry.remove_model("does-not-exist")


def test_set_default_missing_raises(registry):
    with pytest.raises(KeyError, match="not found"):
        registry.set_default("does-not-exist")


def test_corrupt_manifest_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(tmp_path))
    manifest_path = tmp_path / "models.json"
    manifest_path.write_text("not valid json{")
    registry = ModelRegistry()
    assert registry.get_model("unet-v3.2") is not None


def test_set_default_updates_flags(registry, tmp_path):
    path = tmp_path / "dummy.pt"
    path.write_text("not a real checkpoint")
    registry.add_model(
        model_id="second",
        name="Second Model",
        path=str(path),
    )
    assert registry.get_model("unet-v3.2").is_default is True
    assert registry.get_model("second").is_default is False
    registry.set_default("second")
    assert registry.get_model("unet-v3.2").is_default is False
    assert registry.get_model("second").is_default is True
    manifest_path = Path(registry.config_dir) / "models.json"
    data = json.loads(manifest_path.read_text())
    assert data["default_model_id"] == "second"


def test_malformed_entry_raises_with_model_id(registry):
    registry._manifest["models"].append({"model_id": "bad", "unexpected": "value"})
    with pytest.raises(ValueError, match="malformed model entry for 'bad'"):
        registry.list_models()


def test_model_entry_created_at_is_iso_utc():
    before = ModelEntry(
        model_id="x",
        name="X",
        architecture="unet3d",
        source="local",
        path="models/x.pt",
    )
    assert before.created_at.endswith("+00:00")
