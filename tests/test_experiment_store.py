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
