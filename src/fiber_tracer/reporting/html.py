"""HTML report exporter."""

from pathlib import Path
from typing import Union

from fiber_tracer.reporting import CITATIONS


def write_html_report(path: Union[str, Path], summary: dict) -> None:
    """Write a human-readable HTML report with caveats and citations."""
    path = Path(path)
    regime = summary.get("regime", "unknown")
    n_labels = summary.get("n_labels", summary.get("n_voxels", 0))
    caveats = {
        "resolved": (
            "Resolved-regime results depend on successful segmentation and skeletonization. "
            "Overlapping or sub-voxel fibers may be misclassified."
        ),
        "marginal": (
            "Marginal-regime results are computed from a local structure-tensor field. "
            "Accuracy degrades when the fiber diameter is close to the voxel size."
        ),
        "subvoxel": (
            "Subvoxel-regime results aggregate orientations over large windows because individual fibers are not resolved. "
            "Only population-level orientation statistics are reliable."
        ),
    }
    citations = summary.get("citations", CITATIONS)
    citation_items = "".join(f"    <li>{citation}</li>\n" for citation in citations)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAFA Fiber Analysis Report</title>
</head>
<body>
  <h1>RAFA Fiber Analysis Report</h1>
  <p><strong>Regime:</strong> {regime}</p>
  <p><strong>Fibers / windows:</strong> {n_labels}</p>
  <h2>Caveats</h2>
  <p>{caveats.get(regime, "No specific caveats.")}</p>
  <h2>Summary JSON</h2>
  <pre>{summary}</pre>
  <h2>Citations</h2>
  <ul>
{citation_items}  </ul>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
