"""Plotly-based interactive visualizations for RAFA summary reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _import_plotly() -> Any:
    try:
        import plotly.graph_objects as go

        return go
    except ImportError as exc:
        from fiber_tracer.exceptions import BackendNotAvailableError

        raise BackendNotAvailableError("Install viz extra: pip install fiber-tracer[viz]") from exc


def _safe_fibers(summary: dict[str, Any]) -> list:
    return summary.get("fibers", []) or []


def _principal_axis(fibers: list) -> np.ndarray:
    """Estimate the principal population axis from per-fiber orientations."""
    if not fibers:
        return np.array([0.0, 0.0, 1.0])
    orientations = np.array([f["orientation"] for f in fibers])
    cov = np.cov(orientations, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, np.argmax(evals)]


def plot_orientation_distribution(summary: dict[str, Any], output_html: str) -> None:
    """Plot a histogram of angles between fiber orientations and the principal axis."""
    go = _import_plotly()
    fibers = _safe_fibers(summary)
    if not fibers:
        fig = go.Figure()
        fig.update_layout(title="No fiber orientations available")
        fig.write_html(output_html)
        return
    axis = _principal_axis(fibers)
    angles = []
    for f in fibers:
        v = np.array(f["orientation"])
        dot = np.clip(np.abs(np.dot(v, axis)), 0, 1)
        angles.append(float(np.degrees(np.arccos(dot))))
    fig = go.Figure(data=[go.Histogram(x=angles, nbinsx=20)])
    fig.update_layout(
        title="Distribution of fiber orientations relative to principal axis",
        xaxis_title="Angle (degrees)",
        yaxis_title="Count",
    )
    fig.write_html(output_html)


def plot_fiber_property_histogram(
    summary: dict[str, Any],
    property_name: str,
    output_html: str,
) -> None:
    """Plot a histogram of a per-fiber numeric property (e.g. equivalent_diameter_um)."""
    go = _import_plotly()
    fibers = _safe_fibers(summary)
    values = [
        f.get(property_name)
        for f in fibers
        if property_name in f and f.get(property_name) is not None
    ]
    if not values:
        fig = go.Figure()
        fig.update_layout(title=f"No values for property '{property_name}'")
        fig.write_html(output_html)
        return
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=20)])
    fig.update_layout(
        title=f"Distribution of {property_name}",
        xaxis_title=property_name,
        yaxis_title="Count",
    )
    fig.write_html(output_html)


def plot_a2_ellipsoid(a2: np.ndarray, output_html: str, n_points: int = 50) -> None:
    """Plot the orientation tensor A2 as a 3D ellipsoid."""
    go = _import_plotly()
    evals, evecs = np.linalg.eigh(a2)
    # Avoid division by zero for isotropic / zero tensor
    radii = np.sqrt(np.clip(evals, 1e-6, None))
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # Rotate to eigenvector frame
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
    rotated = np.dot(evecs, points)
    xr = rotated[0, :].reshape(x.shape)
    yr = rotated[1, :].reshape(y.shape)
    zr = rotated[2, :].reshape(z.shape)
    fig = go.Figure(data=[go.Surface(x=xr, y=yr, z=zr)])
    fig.update_layout(
        title="A₂ orientation tensor ellipsoid",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    )
    fig.write_html(output_html)


def _figure_to_html_div(fig: Any, div_id: str) -> str:
    """Serialize a Plotly figure as a non-full HTML div for embedding."""
    return str(
        fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            div_id=div_id,
            default_width="100%",
            default_height="500px",
        )
    )


def _combined_report_html(title: str, divs: list[str]) -> str:
    """Build a standalone HTML page embedding multiple Plotly divs."""
    div_blocks = "\n".join(f'<section class="plot-section">{div}</section>' for div in divs)
    style = (
        "body{font-family:Arial,Helvetica,sans-serif;margin:2em;background:#f9f9f9;}"
        "h1{text-align:center;color:#333;}"
        ".plot-section{margin:2em 0;background:#fff;padding:1em;"
        "border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}"
    )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n'
        f"<style>{style}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title}</h1>\n"
        f"{div_blocks}\n"
        "</body>\n"
        "</html>"
    )


def _a2_from_summary(summary: dict[str, Any]) -> np.ndarray | None:
    """Resolve a usable 3x3 A2 tensor from a summary, or None."""
    a2_raw = summary.get("a2")
    if a2_raw is None:
        return None
    a2 = np.asarray(a2_raw, dtype=float)
    if a2.shape != (3, 3):
        return None
    return a2


def generate_interactive_report(summary: dict[str, Any], output_html: str) -> None:
    """Generate a single HTML report with all available plots."""
    go = _import_plotly()
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fibers = _safe_fibers(summary)
    a2 = _a2_from_summary(summary)

    divs: list[str] = []
    title = "RAFA Interactive Report"

    if fibers:
        # Orientation distribution histogram.
        axis = _principal_axis(fibers)
        angles = []
        for f in fibers:
            v = np.array(f["orientation"])
            dot = np.clip(np.abs(np.dot(v, axis)), 0, 1)
            angles.append(float(np.degrees(np.arccos(dot))))
        fig_orientation = go.Figure(data=[go.Histogram(x=angles, nbinsx=20)])
        fig_orientation.update_layout(
            title="Distribution of fiber orientations relative to principal axis",
            xaxis_title="Angle (degrees)",
            yaxis_title="Count",
        )
        divs.append(_figure_to_html_div(fig_orientation, "orientation-distribution"))

        # Equivalent-diameter histogram.
        diameter_values = [
            f.get("equivalent_diameter_um")
            for f in fibers
            if f.get("equivalent_diameter_um") is not None
        ]
        if diameter_values:
            fig_diameter = go.Figure(data=[go.Histogram(x=diameter_values, nbinsx=20)])
            fig_diameter.update_layout(
                title="Distribution of equivalent_diameter_um",
                xaxis_title="equivalent_diameter_um",
                yaxis_title="Count",
            )
            divs.append(_figure_to_html_div(fig_diameter, "diameter-distribution"))

    if a2 is not None:
        evals, evecs = np.linalg.eigh(a2)
        radii = np.sqrt(np.clip(evals, 1e-6, None))
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
        rotated = np.dot(evecs, points)
        xr = rotated[0, :].reshape(x.shape)
        yr = rotated[1, :].reshape(y.shape)
        zr = rotated[2, :].reshape(z.shape)
        fig_a2 = go.Figure(data=[go.Surface(x=xr, y=yr, z=zr)])
        fig_a2.update_layout(
            title="A₂ orientation tensor ellipsoid",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        )
        divs.append(_figure_to_html_div(fig_a2, "a2-ellipsoid"))

    if not divs:
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            '<head><meta charset="utf-8"><title>No visualizations available</title></head>\n'
            "<body><h1>No visualizations available</h1>"
            "<p>The provided summary does not contain fiber data or an A₂ tensor.</p></body>\n"
            "</html>"
        )
        output_path.write_text(html, encoding="utf-8")
        return

    html = _combined_report_html(title, divs)
    output_path.write_text(html, encoding="utf-8")
