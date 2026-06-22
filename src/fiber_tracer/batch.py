"""Batch processing for multiple fiber volumes."""

import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml  # type: ignore[import-untyped]

from fiber_tracer.config import (
    AnalysisConfig,
    Config,
    OrientationConfig,
    ProcessingConfig,
    SegmentationConfig,
    VoxelSpacing,
)
from fiber_tracer.pipeline import FiberAnalysisPipeline


def load_batch_config(path: str) -> dict[str, Any]:
    """Load a batch configuration from YAML or JSON."""
    p = Path(path)
    data: dict[str, Any]
    if p.suffix in {".yaml", ".yml"}:
        with open(p) as f:
            data = yaml.safe_load(f)
    elif p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported batch config format: {p.suffix}")
    return data


def _merge_common(common: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    merged = dict(common)
    merged.update(entry)
    return merged


def _to_voxel_spacing(value: Any) -> Optional[VoxelSpacing]:
    if value is None or isinstance(value, VoxelSpacing):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return VoxelSpacing(*value)
    raise TypeError(f"voxel_spacing_um must be a VoxelSpacing or 3-element sequence, got {value!r}")


def _to_dataclass(value: Any, cls: type) -> Any:
    if value is None or isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"expected dict or {cls.__name__}, got {type(value).__name__}")


def build_config(entry: dict[str, Any]) -> Config:
    """Build a validated Config from a batch entry dict."""
    kwargs: dict[str, Any] = {
        "data_path": entry["data_path"],
        "output_dir": entry["output_dir"],
        "regime": entry.get("regime", "auto"),
    }
    if entry.get("voxel_spacing_um") is not None:
        kwargs["voxel_spacing_um"] = _to_voxel_spacing(entry["voxel_spacing_um"])
    if entry.get("fiber_diameter_um") is not None:
        kwargs["fiber_diameter_um"] = entry["fiber_diameter_um"]
    if entry.get("processing") is not None:
        kwargs["processing"] = _to_dataclass(entry["processing"], ProcessingConfig)
    if entry.get("segmentation") is not None:
        kwargs["segmentation"] = _to_dataclass(entry["segmentation"], SegmentationConfig)
    if entry.get("orientation") is not None:
        kwargs["orientation"] = _to_dataclass(entry["orientation"], OrientationConfig)
    if entry.get("analysis") is not None:
        kwargs["analysis"] = _to_dataclass(entry["analysis"], AnalysisConfig)
    return Config(**kwargs)


def process_batch(
    config_path: str,
    aggregate_csv: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Process all volumes in a batch config and return per-volume summaries."""
    batch = load_batch_config(config_path)
    common = batch.get("common", {})
    entries = batch.get("volumes", [])
    if not entries:
        raise ValueError("Batch config must contain a 'volumes' list")

    results: list[dict[str, Any]] = []
    for entry in entries:
        merged = _merge_common(common, entry)
        config = build_config(merged)
        config.validate()

        start = time.time()
        pipeline = FiberAnalysisPipeline(config)
        summary = pipeline.run()
        elapsed = time.time() - start

        results.append(
            {
                "data_path": config.data_path,
                "output_dir": config.output_dir,
                "regime": summary.get("regime", config.regime),
                "n_labels": summary.get("n_labels", 0),
                "elapsed_s": elapsed,
            }
        )

    if aggregate_csv:
        df = pd.DataFrame(results)
        df.to_csv(aggregate_csv, index=False)

    return results
