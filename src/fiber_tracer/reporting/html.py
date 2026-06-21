"""HTML report exporter."""

from pathlib import Path
from typing import Union


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
    <li>Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. <em>Journal of Rheology</em>, 31(8), 751–784.</li>
    <li>Jeppesen, N., et al. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. <em>Composites Part A</em>, 149, 106541.</li>
    <li>van der Walt et al. (2014). scikit-image: Image processing in Python. <em>PeerJ</em>, 2, e453.</li>
  </ul>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
