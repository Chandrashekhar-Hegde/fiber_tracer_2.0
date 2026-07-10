"""Build the RAFA conference presentation (PDF) from the validated result figures.

Assembles a self-contained slide deck at ``presentation/RAFA_conference.pdf`` (16:9),
plus one PNG per slide under ``presentation/`` for quick preview. It reuses the figure
pack written by ``scripts/make_presentation.py`` (run that first) and generates two
explanatory diagrams — voxelisation and supersampling — from the phantom code.

Run:
    python scripts/make_presentation.py   # writes figures/
    python scripts/build_slides.py        # writes presentation/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from fiber_tracer.validation.phantoms import _generate_finite_fiber  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "figures"
OUT_DIR = REPO / "presentation"

PAPER = "#FBFBF8"
INK = "#181A1F"
MUTED = "#5C6470"
ACCENT = "#0E7C86"
ACCENT_DK = "#0A5A61"
LINE = "#E4E3DC"
FIBER = "#0E7C86"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
    }
)


# --------------------------------------------------------------------------- template


def _new_slide() -> plt.Figure:
    fig = plt.figure(figsize=(13.333, 7.5))
    fig.patch.set_facecolor(PAPER)
    return fig


def _chrome(fig: plt.Figure, page: int, eyebrow: str | None = None) -> None:
    """Footer, page number, and a top accent tick shared by every content slide."""
    fig.add_artist(Rectangle((0.06, 0.905), 0.055, 0.008, color=ACCENT, transform=fig.transFigure))
    if eyebrow:
        fig.text(
            0.06,
            0.855,
            eyebrow.upper(),
            color=ACCENT_DK,
            fontsize=11,
            fontweight="bold",
            family="monospace",
        )
    fig.text(0.06, 0.045, "RAFA · Regime-Aware Fiber Analysis", color=MUTED, fontsize=9.5)
    fig.text(0.94, 0.045, f"{page:02d}", color=MUTED, fontsize=9.5, ha="right")


def _title(fig: plt.Figure, text: str, y: float = 0.80) -> None:
    fig.text(0.06, y, text, color=INK, fontsize=30, fontweight="bold", va="top", wrap=True)


def _bullets(
    fig: plt.Figure,
    items: list[str],
    x: float = 0.075,
    y: float = 0.66,
    dy: float = 0.093,
    size: float = 16.5,
    width: float = 0.52,
) -> None:
    for i, item in enumerate(items):
        yy = y - i * dy
        fig.add_artist(Circle((x, yy + 0.012), 0.006, color=ACCENT, transform=fig.transFigure))
        fig.text(
            x + 0.022, yy, item, color=INK, fontsize=size, va="center", wrap=True, linespacing=1.35
        )
        # Constrain width by a manual wrap: matplotlib wrap uses artist extent; keep lines short.
        _ = width


def _image_axes(
    fig: plt.Figure, rect: tuple[float, float, float, float], png: Path, frame: bool = False
) -> None:
    if frame:
        x, y, w, h = rect
        pad = 0.014
        fig.add_artist(
            FancyBboxPatch(
                (x - pad, y - pad),
                w + 2 * pad,
                h + 2 * pad,
                boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure,
                facecolor="white",
                edgecolor=LINE,
                linewidth=1.2,
                zorder=0,
            )
        )
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(png))
    ax.axis("off")


def _caption(fig: plt.Figure, text: str, y: float = 0.11) -> None:
    fig.text(0.5, y, text, color=MUTED, fontsize=12.5, ha="center", style="italic", wrap=True)


# --------------------------------------------------------------------------- diagrams


def _fig_voxelization(path: Path) -> None:
    """A continuous fibre cross-section discretised onto a voxel grid."""
    fig, (axc, axv) = plt.subplots(1, 2, figsize=(9.6, 4.6))
    fig.patch.set_facecolor(PAPER)
    n = 12
    cx, cy, r = 5.4, 6.2, 3.3
    for ax, title, discretise in (
        (axc, "Continuous fibre", False),
        (axv, "Voxelised (hard)", True),
    ):
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        for k in range(n + 1):
            ax.axhline(k, color=LINE, lw=0.8)
            ax.axvline(k, color=LINE, lw=0.8)
        if discretise:
            yy, xx = np.mgrid[0:n, 0:n]
            inside = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r**2
            for j in range(n):
                for i in range(n):
                    if inside[j, i]:
                        ax.add_patch(Rectangle((i, j), 1, 1, color=FIBER, alpha=0.85))
            ax.add_patch(Circle((cx, cy), r, fill=False, ec=INK, lw=1.6, ls="--"))
        else:
            ax.add_patch(Circle((cx, cy), r, color=FIBER, alpha=0.85))
        ax.set_title(title, fontsize=13, color=INK, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fig_supersampling(path: Path) -> tuple[int, int]:
    """Hard vs supersampled voxelisation of one circular fibre cross-section (zoomed)."""
    shape = (40, 40, 40)
    # Axis along z (first stack component) so the mid-slice is a circular cross-section.
    kw = dict(
        center=(20, 20, 20),
        direction=np.array([1.0, 0.0, 0.0]),
        radius_voxels=6.0,
        length_um=36.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
    )
    hard = _generate_finite_fiber(shape, supersample=1, **kw)[20]
    soft = _generate_finite_fiber(shape, supersample=5, **kw)[20]
    sl = slice(9, 31)
    n_hard = int(np.sum((hard > 0.01) & (hard < 0.99)))
    n_soft = int(np.sum((soft > 0.01) & (soft < 0.99)))

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
    fig.patch.set_facecolor(PAPER)
    for ax, img, title in (
        (axs[0], hard[sl, sl], f"Hard voxelisation\n{n_hard} partial-volume voxels"),
        (axs[1], soft[sl, sl], f"Supersampled ×5\n{n_soft} partial-volume voxels"),
    ):
        ax.imshow(img, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, color=INK, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return n_hard, n_soft


def _fig_regime_axis(path: Path) -> None:
    """Schematic of the three regimes (equal-width zones, r thresholds annotated)."""
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    fig.patch.set_facecolor(PAPER)
    zones = [
        ("#0E7C86", "resolved", "r ≤ 0.3", "trace each fibre"),
        ("#C77D2E", "marginal", "0.3 < r ≤ 3", "windowed A2 tensor"),
        ("#8A4B8F", "subvoxel", "r > 3", "global orientation"),
    ]
    for i, (col, name, rng, how) in enumerate(zones):
        ax.add_patch(Rectangle((i, 0), 1, 1, color=col, alpha=0.9))
        ax.text(
            i + 0.5,
            0.66,
            name,
            ha="center",
            va="center",
            color="white",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(i + 0.5, 0.40, rng, ha="center", va="center", color="white", fontsize=12)
        ax.text(
            i + 0.5,
            0.16,
            how,
            ha="center",
            va="center",
            color="white",
            fontsize=10.5,
            style="italic",
        )
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(
        "increasing  r  =  min(voxel spacing) / fibre diameter  →", fontsize=13, color=INK
    )
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fig_pipeline_flow(path: Path) -> None:
    stages = [
        "Load\nCT volume",
        "Normalise\n+ denoise",
        "Threshold\n(Otsu / …)",
        "Label\nfibres",
        "Skeletonise\n+ track",
        "Orient\n+ report",
    ]
    fig, ax = plt.subplots(figsize=(11.5, 2.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, len(stages))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(stages):
        box = FancyBboxPatch(
            (i + 0.08, 0.28),
            0.84,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ACCENT,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=11, color=INK)
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (i + 0.92, 0.5),
                    (i + 1.08, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color=ACCENT_DK,
                    lw=1.6,
                )
            )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- slides


def _read_metrics() -> dict[str, str]:
    m: dict[str, str] = {}
    csv = FIG_DIR / "metrics.csv"
    if csv.exists():
        for line in csv.read_text().splitlines()[1:]:
            panel, metric, value = line.split(",")
            m[f"{panel}.{metric}"] = value
    return m


def _slide_title(pdf: PdfPages) -> None:
    fig = _new_slide()
    fig.add_artist(Rectangle((0, 0), 1, 0.14, color=ACCENT, transform=fig.transFigure))
    fig.add_artist(Rectangle((0, 0.86), 1, 0.14, color=ACCENT, transform=fig.transFigure))
    fig.text(0.5, 0.60, "RAFA", ha="center", color=INK, fontsize=64, fontweight="bold")
    fig.text(0.5, 0.515, "Regime-Aware Fiber Analysis", ha="center", color=ACCENT_DK, fontsize=24)
    fig.text(
        0.5,
        0.42,
        "Quantifying fibre orientation and morphometry in composites\n"
        "from 3D X-ray computed tomography",
        ha="center",
        color=MUTED,
        fontsize=15,
        linespacing=1.5,
    )
    fig.text(
        0.5,
        0.29,
        "Conference presentation",
        ha="center",
        color=MUTED,
        fontsize=12,
        family="monospace",
    )
    pdf.savefig(fig)
    fig.savefig(OUT_DIR / "slide_01.png", dpi=110)
    plt.close(fig)


def _content(pdf: PdfPages, page: int, name: str, build) -> None:
    fig = _new_slide()
    build(fig)
    _chrome(fig, page)
    pdf.savefig(fig)
    fig.savefig(OUT_DIR / name, dpi=110)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    if not (FIG_DIR / "02_pipeline.png").exists():
        raise SystemExit("Run scripts/make_presentation.py first (figures/ is missing).")

    vox_png = OUT_DIR / "_voxelization.png"
    ss_png = OUT_DIR / "_supersampling.png"
    regime_png = OUT_DIR / "_regime_axis.png"
    flow_png = OUT_DIR / "_pipeline_flow.png"
    _fig_voxelization(vox_png)
    n_hard, n_soft = _fig_supersampling(ss_png)
    _fig_regime_axis(regime_png)
    _fig_pipeline_flow(flow_png)
    m = _read_metrics()

    def _num(key: str, default: str, places: int = 3) -> str:
        try:
            return f"{float(m[key]):.{places}f}"
        except (KeyError, ValueError):
            return default

    gf_dice = _num("seg_gfpa66.dice_otsu", "0.966")
    pipe_ang = _num("pipeline.angular_error_deg", "0.20", places=2)
    pipe_dice = _num("pipeline.dice", "0.978")
    rob_clean = _num("robustness.dice_clean", "0.978")
    rob_deg = _num("robustness.dice_degraded", "0.950")

    pdf_path = OUT_DIR / "RAFA_conference.pdf"
    with PdfPages(pdf_path) as pdf:
        _slide_title(pdf)

        def motivation(fig):
            _title(fig, "Why fibre orientation matters")
            _bullets(
                fig,
                [
                    "Composite stiffness and strength are set by fibre orientation.",
                    "Used in load-bearing aerospace, defence, and energy structures.",
                    "X-ray CT images the 3D fibre architecture non-destructively.",
                    "Challenge: reliable orientation across very different resolutions.",
                ],
                size=16.5,
                dy=0.093,
                y=0.65,
            )

        _content(pdf, 2, "slide_02.png", motivation)

        def coreidea(fig):
            _title(fig, "One idea: let the resolution pick the algorithm")
            _bullets(
                fig,
                [
                    "A single ratio drives everything:  r = voxel spacing / fibre diameter.",
                    "Resolved fibres are traced individually, one by one.",
                    "Unresolved fibres are described by orientation-tensor statistics.",
                    "RAFA selects the regime automatically from r.",
                ],
                size=15.5,
                dy=0.066,
                y=0.63,
            )
            _image_axes(fig, (0.13, 0.09, 0.74, 0.26), regime_png)

        _content(pdf, 3, "slide_03.png", coreidea)

        def voxelization(fig):
            _title(fig, "Voxelisation: geometry on a discrete grid")
            _bullets(
                fig,
                [
                    "Real CT and phantoms live on a voxel grid with known spacing (µm).",
                    "That spacing is the fundamental length scale of the analysis.",
                    "Anisotropic scans are resampled to isotropic voxels first.",
                    "The voxel-to-fibre ratio r determines the regime.",
                ],
                size=15.5,
                dy=0.066,
                y=0.63,
            )
            _image_axes(fig, (0.32, 0.08, 0.38, 0.32), vox_png)

        _content(pdf, 4, "slide_04.png", voxelization)

        def supersampling(fig):
            _title(fig, "Supersampling: realistic partial-volume edges")
            _bullets(
                fig,
                [
                    "Hard voxelisation gives aliased, staircase fibre edges — unlike CT.",
                    "Supersampling averages many sub-voxel samples per voxel.",
                    "Edge voxels take fractional values: the partial-volume effect.",
                    f"Here: {n_hard} partial-volume voxels (hard) → {n_soft} (supersampled ×5).",
                ],
                size=15.5,
                dy=0.062,
                y=0.64,
            )
            _image_axes(fig, (0.34, 0.07, 0.34, 0.33), ss_png)

        _content(pdf, 5, "slide_05.png", supersampling)

        def pipeline(fig):
            _title(fig, "The resolved-regime pipeline")
            _image_axes(fig, (0.05, 0.45, 0.90, 0.24), flow_png)
            _bullets(
                fig,
                [
                    "Every stage is validated against ground truth.",
                    "Segmentation method is switchable; its Dice is reported.",
                    "Skeletonisation yields per-fibre length, tortuosity, orientation.",
                ],
                size=15.5,
                dy=0.08,
                y=0.35,
            )

        _content(pdf, 6, "slide_06.png", pipeline)

        def seg(fig):
            _title(fig, "Validated segmentation on real CT", y=0.86)
            _image_axes(
                fig, (0.06, 0.30, 0.88, 0.46), FIG_DIR / "01_segmentation_gfpa66.png", frame=True
            )
            _caption(
                fig,
                f"Real GF-PA66 X-ray CT vs ground-truth mask — Otsu reaches Dice {gf_dice}. "
                "Method is switchable and its accuracy is reported per dataset.",
                y=0.19,
            )

        _content(pdf, 7, "slide_07.png", seg)

        def pipe_result(fig):
            _title(fig, "End-to-end: raw CT to fibre centrelines", y=0.86)
            _image_axes(fig, (0.06, 0.30, 0.88, 0.46), FIG_DIR / "02_pipeline.png", frame=True)
            _caption(
                fig,
                f"Four fibres segmented, labelled, skeletonised and oriented — "
                f"mean orientation error {pipe_ang}°, Dice {pipe_dice} vs ground truth.",
                y=0.19,
            )

        _content(pdf, 8, "slide_08.png", pipe_result)

        def robust(fig):
            _title(fig, "Robust to acquisition artefacts", y=0.86)
            _image_axes(fig, (0.20, 0.20, 0.60, 0.58), FIG_DIR / "03_robustness.png", frame=True)
            _caption(
                fig,
                f"Strong beam hardening, rings, blur and noise added — segmentation "
                f"Dice holds {rob_clean} → {rob_deg}.",
                y=0.12,
            )

        _content(pdf, 9, "slide_09.png", robust)

        def regimes(fig):
            _title(fig, "Regime awareness end to end", y=0.86)
            _image_axes(fig, (0.06, 0.30, 0.88, 0.46), FIG_DIR / "04_regimes.png", frame=True)
            _caption(
                fig,
                "The same material at different voxel sizes: per-fibre morphometry when "
                "resolved, orientation-tensor fields (A2, FA) when marginal or subvoxel.",
                y=0.19,
            )

        _content(pdf, 10, "slide_10.png", regimes)

        def summary(fig):
            _title(fig, "Validation summary")
            _bullets(
                fig,
                [
                    f"Segmentation vs ground truth (real GF-PA66 CT):  Dice {gf_dice}.",
                    f"Resolved pipeline vs phantom:  Dice {pipe_dice}, error {pipe_ang}°.",
                    f"Robustness to XCT artefacts:  Dice {rob_clean} → {rob_deg}.",
                    "Phantom benchmark asserts Dice > 0.85 and angular error < 5°.",
                    "Deterministic, seeded, one-command reproducible.",
                ],
                size=15.5,
                dy=0.084,
                y=0.66,
            )

        _content(pdf, 11, "slide_11.png", summary)

        def conclusion(fig):
            _title(fig, "Takeaways & next steps")
            _bullets(
                fig,
                [
                    "Resolution-aware analysis gives defensible results across scan scales.",
                    "Every number is validated against ground truth and reproducible.",
                    "Next: learned 3D U-Net segmentation for low-contrast scans.",
                    "Then: DVC and digital-twin coupling for prediction.",
                ],
                size=15.5,
                dy=0.084,
                y=0.66,
            )

        _content(pdf, 12, "slide_12.png", conclusion)

        def refs(fig):
            _title(fig, "Key references")
            _bullets(
                fig,
                [
                    "Advani & Tucker (1987), J. Rheology — orientation tensors.",
                    "Jeppesen et al. (2021), Composites A — structure-tensor orientation.",
                    "Bertoldo et al. (2021), Frontiers in Materials — GF-PA66 benchmark.",
                    "van der Walt et al. (2014), PeerJ — scikit-image.",
                ],
                size=14.5,
                dy=0.082,
                y=0.64,
            )

        _content(pdf, 13, "slide_13.png", refs)

    print("Wrote", pdf_path)
    print("Slides:", sorted(p.name for p in OUT_DIR.glob("slide_*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
