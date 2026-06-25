import logging

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


def test_get_experiment_missing(store):
    assert store.get_experiment("does-not-exist") is None


def test_update_missing_id_returns_none(store):
    assert store.update("does-not-exist", status="completed") is None


def test_update_rejects_unknown_kwarg(store):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    with pytest.raises(ValueError, match="unknown experiment field: bad_field"):
        store.update(exp.id, bad_field="value")


def test_update_rejects_id_change(store):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    with pytest.raises(ValueError, match="cannot change experiment id"):
        store.update(exp.id, id="new-id")


def test_update_rejects_id_change_even_without_existing(store):
    with pytest.raises(ValueError, match="cannot change experiment id"):
        store.update("any-id", id="new-id")


@pytest.mark.parametrize(
    "status",
    ["pending", "running", "completed", "failed", "cancelled"],
)
def test_valid_status_values(store, status):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    updated = store.update(exp.id, status=status)
    assert updated is not None
    assert updated.status == status


def test_invalid_status(store):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    with pytest.raises(ValueError, match="invalid status 'bad_status'"):
        store.update(exp.id, status="bad_status")


def test_corrupt_line_skipped_with_log(store, caplog):
    store.store_path.write_text(
        '{"id": "exp-1", "name": "good", "type": "train", "model_id": "m", "dataset": "d"}\n'
        "this is not json\n"
    )
    with caplog.at_level(logging.WARNING, logger="fiber_tracer.experiments.store"):
        experiments = store.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].id == "exp-1"
    assert any("skipping corrupt JSONL line" in r.message for r in caplog.records)
    assert any("this is not json" in r.message for r in caplog.records)


def test_empty_store(store):
    assert store.list_experiments() == []
    assert store.get_experiment("missing") is None
    assert store.compare(["missing"], metric="dice") == {}


def test_compare_with_list_metric(store):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    store.update(exp.id, metrics={"loss": [0.9, 0.5, 0.1]})
    comparison = store.compare([exp.id], metric="loss")
    assert comparison[exp.id] == 0.1


def test_compare_with_empty_list_metric(store):
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    store.update(exp.id, metrics={"loss": []})
    comparison = store.compare([exp.id], metric="loss")
    assert comparison[exp.id] == []


def test_custom_config_dir_auto_created(tmp_path):
    config_dir = tmp_path / "sub" / "config"
    assert not config_dir.exists()
    store = ExperimentStore(config_dir=str(config_dir))
    assert config_dir.exists()
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    assert exp.id
    assert store.list_experiments()[0].id == exp.id


def test_leftover_tmp_file_cleaned(store):
    tmp = store.store_path.with_suffix(".jsonl.tmp")
    tmp.write_text("leftover garbage\n")
    exp = store.create(name="test", type="train", model_id="m", dataset="d")
    assert exp.id
    assert not tmp.exists()
