"""CSV report exporter for per-fiber / per-window results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _records_from_summary(summary: dict) -> list[dict[str, Any]]:
    """Extract tabular records from a pipeline summary."""
    records = []
    regime = summary.get("regime", "unknown")
    for fiber in summary.get("fibers", []):
        record = {"regime": regime}
        record.update(fiber)
        records.append(record)
    for i, entry in enumerate(summary.get("a2_windows", [])):
        record = {"regime": regime, "window_id": i}
        record.update(entry)
        records.append(record)
    for i, window in enumerate(summary.get("dvc_windows", [])):
        node_z, node_y, node_x = window["node_position"]
        disp_z, disp_y, disp_x = window["displacement_voxels"]
        strain_z, strain_y, strain_x = window["strain"]
        records.append(
            {
                "regime": "dvc",
                "window_id": i,
                "node_z": node_z,
                "node_y": node_y,
                "node_x": node_x,
                "displacement_z_voxels": disp_z,
                "displacement_y_voxels": disp_y,
                "displacement_x_voxels": disp_x,
                "strain_z": strain_z,
                "strain_y": strain_y,
                "strain_x": strain_x,
                "return_status": window["return_status"],
                "converged": window["converged"],
            }
        )
    return records


def write_csv_report(path: str | Path, summary: dict) -> None:
    """Write per-fiber / per-window records to CSV."""
    path = Path(path)
    records = _records_from_summary(summary)
    if not records:
        # Write empty DataFrame with expected columns so the file still exists.
        pd.DataFrame(columns=["regime"]).to_csv(path, index=False)
        return
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
