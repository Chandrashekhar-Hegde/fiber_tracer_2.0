"""JSON report exporter."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_report(path: str | Path, summary: dict) -> None:
    """Write the analysis summary to a JSON file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
