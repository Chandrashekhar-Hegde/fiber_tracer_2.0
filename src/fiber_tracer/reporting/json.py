"""JSON report exporter."""

import json
from pathlib import Path
from typing import Any, Union


def write_json_report(path: Union[str, Path], summary: dict) -> None:
    """Write the analysis summary to a JSON file."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
