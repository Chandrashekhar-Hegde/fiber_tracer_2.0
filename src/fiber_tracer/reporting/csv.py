"""CSV report exporter for per-fiber / per-window results."""

from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


def _records_from_summary(summary: dict) -> List[Dict[str, Any]]:
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
    return records


def write_csv_report(path: Union[str, Path], summary: dict) -> None:
    """Write per-fiber / per-window records to CSV."""
    path = Path(path)
    records = _records_from_summary(summary)
    if not records:
        # Write empty DataFrame with expected columns so the file still exists.
        pd.DataFrame(columns=["regime"]).to_csv(path, index=False)
        return
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
