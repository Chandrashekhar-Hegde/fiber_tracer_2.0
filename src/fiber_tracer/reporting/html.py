"""HTML report exporter."""

from __future__ import annotations

from pathlib import Path

from fiber_tracer.reporting.citations import CITATIONS, REGIME_CAVEATS

_CSS = """
:root{--ink:#181a1f;--muted:#5c6470;--line:#e4e3dc;--accent:#0e7c86;--paper:#fbfbf8}
@media (prefers-color-scheme:dark){:root{--ink:#e8eaed;--muted:#9aa4b2;
 --line:#252b34;--accent:#35b7c4;--paper:#0e1116}}
body{margin:0;background:var(--paper);color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:48px 24px}
h1{font-size:1.7rem;margin:0 0 .2em}h2{font-size:1.05rem;margin:2em 0 .6em;color:var(--accent)}
.meta{color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:.85rem}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.92rem}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
th{color:var(--muted);text-transform:uppercase;font-size:.72rem;letter-spacing:.06em}
ul{padding-left:1.1em;color:var(--muted)}li{margin:.3em 0}
.caveat{border-left:3px solid var(--accent);padding:.4em 0 .4em 1em;color:var(--muted)}
"""


def _fmt_vec(vec: object) -> str:
    if isinstance(vec, (list, tuple)) and len(vec) == 3:
        return "(" + ", ".join(f"{float(c):+.3f}" for c in vec) + ")"
    return str(vec)


def _fibers_table(fibers: list[dict]) -> str:
    head = (
        "<tr><th>Label</th><th>Voxels</th><th>Equivalent diameter (µm)</th>"
        "<th>Orientation (z, y, x)</th></tr>"
    )
    rows = []
    for f in fibers:
        diam = f.get("equivalent_diameter_um")
        diam_str = f"{float(diam):.2f}" if isinstance(diam, (int, float)) else "–"
        rows.append(
            f"<tr><td>{f.get('label', '–')}</td><td>{f.get('n_voxels', '–')}</td>"
            f"<td>{diam_str}</td><td>{_fmt_vec(f.get('orientation'))}</td></tr>"
        )
    return f"<table>{head}{''.join(rows)}</table>"


def _population_table(summary: dict) -> str:
    """Scalar population-level stats (marginal / subvoxel regimes)."""
    keys = ("fa", "global_fa", "n_windows", "principal_axis")
    rows = []
    for key in keys:
        if key in summary:
            value = summary[key]
            shown = _fmt_vec(value) if key == "principal_axis" else value
            rows.append(f"<tr><td>{key}</td><td>{shown}</td></tr>")
    if not rows:
        return ""
    return "<table><tr><th>Statistic</th><th>Value</th></tr>" + "".join(rows) + "</table>"


def write_html_report(path: str | Path, summary: dict) -> None:
    """Write a human-readable HTML report with a results table, caveats, and citations."""
    path = Path(path)
    regime = summary.get("regime", "unknown")
    n_labels = summary.get("n_labels", summary.get("n_voxels", 0))
    caveats = summary.get("caveats", REGIME_CAVEATS.get(regime, ""))
    citations = summary.get("citations", CITATIONS)

    fibers = summary.get("fibers")
    if fibers:
        results = f"<h2>Per-fibre results ({len(fibers)})</h2>{_fibers_table(fibers)}"
    else:
        pop = _population_table(summary)
        results = f"<h2>Population statistics</h2>{pop}" if pop else ""

    citation_items = "".join(f"    <li>{citation}</li>\n" for citation in citations)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAFA Fiber Analysis Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="wrap">
    <h1>RAFA Fiber Analysis Report</h1>
    <p class="meta">Regime: {regime} &nbsp;·&nbsp; Fibers / windows: {n_labels}</p>
    {results}
    <h2>Caveats</h2>
    <p class="caveat">{caveats}</p>
    <h2>Citations</h2>
    <ul>
{citation_items}    </ul>
  </div>
</body>
</html>"""
    with open(path, "w") as f:
        f.write(html)
