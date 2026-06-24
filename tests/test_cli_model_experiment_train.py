import json
import os
import subprocess
import sys

from fiber_tracer.experiments.store import ExperimentStore


def run_cli(tmp_path, *args):
    """Run the CLI in an isolated config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FIBER_TRACER_CONFIG_DIR"] = str(config_dir)
    return subprocess.run(
        [sys.executable, "-m", "fiber_tracer.cli", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_model_list(tmp_path):
    result = run_cli(tmp_path, "model", "list")
    assert result.returncode == 0, result.stderr
    assert "unet-v3.2" in result.stdout


def test_model_list_json(tmp_path):
    result = run_cli(tmp_path, "model", "list", "--json")
    assert result.returncode == 0, result.stderr
    models = json.loads(result.stdout)
    assert isinstance(models, list)
    assert any(m["model_id"] == "unet-v3.2" for m in models)


def test_model_add_remove_and_set_default(tmp_path):
    model_path = tmp_path / "dummy.pt"
    model_path.write_text("not a real checkpoint")

    add_result = run_cli(
        tmp_path,
        "model",
        "add",
        "--model-id",
        "local-1",
        "--name",
        "Local Model",
        "--path",
        str(model_path),
        "--architecture",
        "unet3d",
    )
    assert add_result.returncode == 0, add_result.stderr
    assert "Added model local-1" in add_result.stdout

    list_result = run_cli(tmp_path, "model", "list")
    assert list_result.returncode == 0, list_result.stderr
    assert "local-1" in list_result.stdout

    set_default_result = run_cli(tmp_path, "model", "set-default", "local-1")
    assert set_default_result.returncode == 0, set_default_result.stderr
    assert "Default model set to local-1" in set_default_result.stdout

    list_default_result = run_cli(tmp_path, "model", "list")
    assert "local-1" in list_default_result.stdout
    assert "(default)" in list_default_result.stdout

    # Switch default back to bundled model so local-1 can be removed.
    reset_default_result = run_cli(tmp_path, "model", "set-default", "unet-v3.2")
    assert reset_default_result.returncode == 0, reset_default_result.stderr

    remove_result = run_cli(tmp_path, "model", "remove", "local-1")
    assert remove_result.returncode == 0, remove_result.stderr
    assert "Removed model local-1" in remove_result.stdout

    list_after_remove = run_cli(tmp_path, "model", "list")
    assert "local-1" not in list_after_remove.stdout


def test_experiment_list_json(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIBER_TRACER_CONFIG_DIR", str(config_dir))

    store = ExperimentStore()
    exp = store.create(name="smoke", type="train", model_id="unet-v3.2", dataset="/data")

    result = run_cli(tmp_path, "experiment", "list", "--json")
    assert result.returncode == 0, result.stderr
    experiments = json.loads(result.stdout)
    assert isinstance(experiments, list)
    assert any(e["id"] == exp.id for e in experiments)


def test_train_help_returns_zero(tmp_path):
    result = run_cli(tmp_path, "train", "--help")
    assert result.returncode == 0, result.stderr
    assert "--dataset-dir" in result.stdout
    assert "--model-id" in result.stdout
