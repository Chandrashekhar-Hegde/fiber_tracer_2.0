"""Experiment tracking backed by a JSONL file."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fiber_tracer.utils.locking import file_lock
from fiber_tracer.utils.paths import get_config_dir

logger = logging.getLogger(__name__)

_ALLOWED_STATUS = {"pending", "running", "completed", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"exp-{today}-{uuid.uuid4().hex[:6]}"


def _validate_status(status: str) -> None:
    if status not in _ALLOWED_STATUS:
        allowed = ", ".join(sorted(_ALLOWED_STATUS))
        raise ValueError(f"invalid status {status!r}; must be one of: {allowed}")


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
    history: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    artifact_dir: str = ""
    error_message: str = ""


class ExperimentStore:
    """Read and write experiment records at ``~/.config/fiber-tracer/experiments.jsonl``."""

    def __init__(self, config_dir: str | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir else Path(get_config_dir())
        self.config_dir.mkdir(parents=True, exist_ok=True)
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
                logger.warning(
                    "skipping corrupt JSONL line in %s: %s",
                    self.store_path,
                    line[:200],
                )
                continue
        return experiments

    def _write_all(self, experiments: list[Experiment]) -> None:
        tmp = self.store_path.with_suffix(".jsonl.tmp")
        if tmp.exists():
            tmp.unlink()
        tmp.write_text("".join(json.dumps(asdict(e)) + "\n" for e in experiments))
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
        with file_lock(self.store_path):
            experiments = self._read_all()
            experiments.append(experiment)
            self._write_all(experiments)
        return experiment

    def update(self, experiment_id: str, **kwargs: Any) -> Experiment | None:
        if "id" in kwargs:
            raise ValueError("cannot change experiment id")

        status = kwargs.get("status")
        if status is not None:
            _validate_status(status)

        with file_lock(self.store_path):
            experiments = self._read_all()
            for i, exp in enumerate(experiments):
                if exp.id == experiment_id:
                    for key, value in kwargs.items():
                        if not hasattr(exp, key):
                            raise ValueError(f"unknown experiment field: {key}")
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
        """Compare ``metric`` across ``experiment_ids``.

        If the metric value is a non-empty list, the last element of that list
        is used as the comparison value.
        """
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
