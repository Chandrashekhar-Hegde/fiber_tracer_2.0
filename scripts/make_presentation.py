"""Generate conference before/after figures, metrics, and an HTML deck for RAFA.

Renders four before/after panels as PNGs, a metrics table (CSV + Markdown), and a
self-contained HTML results deck into ``figures/``. Everything is built from a single
seeded synthetic phantom plus one real GF-PA66 CT patch, reusing the existing analysis
code (no new dependencies).

Run:
    python scripts/make_presentation.py
"""

from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage.color import label2rgb  # noqa: E402

from fiber_tracer.config import Config, VoxelSpacing  # noqa: E402
from fiber_tracer.io import load_tiff_stack, save_tiff_stack  # noqa: E402
from fiber_tracer.pipeline import FiberAnalysisPipeline  # noqa: E402
from fiber_tracer.segmentation.classical import binarize_volume  # noqa: E402
from fiber_tracer.validation.benchmark import _align_labels, mean_dice_per_label  # noqa: E402
from fiber_tracer.validation.metrics import dice_score, mean_angular_error  # noqa: E402
from fiber_tracer.validation.phantoms import (  # noqa: E402
    add_beam_hardening,
    add_partial_volume_blur,
    add_poisson_noise,
    add_ring_artifacts,
    generate_fiber_phantom,
)

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "figures"
GFPA66_PATCHES = REPO / "data/processed_test/real/gfpa66_pa66_volumes_pa66"
BENCHMARK_JSON = REPO / "benchmark_results/benchmark_results.json"

# Instrument-teal accent grounded in the XCT / false-colour world; skeleton in a warm
# signal orange so it separates from the teal fibres.
SKELETON_RGB = (1.0, 0.42, 0.02)
plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight", "font.size": 10})


def _best_z(mask: np.ndarray) -> int:
    """Z index of the slice carrying the most foreground (so figures show fibres)."""
    fg = np.asarray(mask).astype(bool).reshape(mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(fg))


def _show(ax: plt.Axes, img: np.ndarray, title: str, cmap: str | None = "gray") -> None:
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _pick_axis(mask: np.ndarray) -> int:
    """Projection axis whose max-projection shows the most fibre foreground."""
    return int(np.argmax([(mask.max(axis=a) > 0).sum() for a in range(3)]))


def _label_mip(norm: np.ndarray, labels: np.ndarray, axis: int) -> np.ndarray:
    """Colour each fibre in a max-intensity projection (label of the brightest voxel)."""
    idx = np.expand_dims(norm.argmax(axis=axis), axis=axis)
    flat = np.take_along_axis(labels, idx, axis=axis).squeeze(axis=axis)
    rgb = label2rgb(flat, image=norm.max(axis=axis), bg_label=0, alpha=0.6, image_alpha=1.0)
    return np.clip(rgb, 0.0, 1.0)


def _skeleton_mip(norm: np.ndarray, skel: np.ndarray, axis: int) -> np.ndarray:
    gray = norm.max(axis=axis)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(float)
    rgb[skel.max(axis=axis) > 0] = SKELETON_RGB
    return np.clip(rgb, 0.0, 1.0)


def _run_resolved(volume: np.ndarray, phantom, tmp: Path, name: str) -> dict:
    """Run the resolved pipeline on *volume*, returning outputs + accuracy vs phantom GT."""
    out = tmp / name
    out.mkdir(parents=True, exist_ok=True)
    save_tiff_stack(out / "input.tif", volume)
    config = Config(
        data_path=str(out / "input.tif"),
        output_dir=str(out),
        voxel_spacing_um=VoxelSpacing(*phantom.voxel_spacing_um),
        fiber_diameter_um=phantom.fiber_diameter_um,
        regime="resolved",
    )
    summary = FiberAnalysisPipeline(config).run()

    normalized = load_tiff_stack(out / "normalized_input.tif")
    labels = load_tiff_stack(out / "labels.tif").astype(np.int32)
    skeleton = load_tiff_stack(out / "skeleton.tif")

    aligned, mapping = _align_labels(labels, phantom.labels)
    dice = mean_dice_per_label(aligned, phantom.labels)

    orient_by_pred = {f["label"]: np.asarray(f["orientation"]) for f in summary["fibers"]}
    preds, trues = [], []
    for pred_id, true_id in mapping.items():
        if pred_id in orient_by_pred:
            preds.append(orient_by_pred[pred_id])
            trues.append(phantom.orientations[true_id - 1])
    ang = mean_angular_error(np.array(preds), np.array(trues)) if preds else float("nan")

    return {
        "normalized": normalized,
        "labels": labels,
        "skeleton": skeleton,
        "n_fibers": int(summary["n_labels"]),
        "dice": float(dice),
        "angular_error_deg": float(ang),
    }


# --------------------------------------------------------------------------- panels


def panel_segmentation(volume: np.ndarray, gt_mask: np.ndarray, path: Path, title: str) -> dict:
    """Otsu (baseline) vs multi-Otsu vs adaptive vs ground truth, with Dice per method."""
    masks = {m: binarize_volume(volume, method=m) for m in ("otsu", "multiotsu", "adaptive")}
    dice = {m: dice_score(masks[m], gt_mask) for m in masks}
    z = _best_z(gt_mask if gt_mask.any() else masks["otsu"])
    best = max(dice, key=dice.get)

    fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.6))
    _show(axs[0], volume[z], "Raw CT slice")
    _show(axs[1], masks["otsu"][z], f"Otsu\nDice {dice['otsu']:.3f}")
    _show(axs[2], masks["multiotsu"][z], f"Multi-Otsu\nDice {dice['multiotsu']:.3f}")
    _show(axs[3], gt_mask[z], "Ground truth")
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    return {"z": z, "best_method": best, **{f"dice_{m}": dice[m] for m in dice}}


def panel_pipeline(res: dict, axis: int, path: Path) -> dict:
    """Input -> segmented fibres -> centrelines, shown as max-intensity projections."""
    norm = res["normalized"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9))
    _show(axs[0], norm.max(axis=axis), "1 · Input (max projection)")
    _show(
        axs[1],
        _label_mip(norm, res["labels"], axis),
        f"2 · Segmented fibres ({res['n_fibers']})",
        cmap=None,
    )
    _show(axs[2], _skeleton_mip(norm, res["skeleton"], axis), "3 · Fibre centrelines", cmap=None)
    fig.suptitle(
        f"Resolved pipeline · {res['n_fibers']} fibres · "
        f"Dice {res['dice']:.3f} · orientation error {res['angular_error_deg']:.2f}°",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return {
        "n_fibers": res["n_fibers"],
        "dice": res["dice"],
        "angular_error_deg": res["angular_error_deg"],
    }


def panel_robustness(
    clean: dict, degraded: dict, clean_vol, degraded_vol, axis: int, path: Path
) -> dict:
    """Clean input vs strongly XCT-artefact-degraded input; the analysis still holds."""
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 8.2))
    _show(axs[0, 0], clean_vol.max(axis=axis), "Clean input")
    _show(
        axs[0, 1],
        _label_mip(clean["normalized"], clean["labels"], axis),
        f"Clean → Dice {clean['dice']:.3f}",
        cmap=None,
    )
    _show(
        axs[1, 0],
        degraded_vol.max(axis=axis),
        "Degraded input\n(beam hardening · rings · blur · noise)",
    )
    _show(
        axs[1, 1],
        _label_mip(degraded["normalized"], degraded["labels"], axis),
        f"Degraded → Dice {degraded['dice']:.3f}",
        cmap=None,
    )
    fig.suptitle("Robustness to XCT acquisition artefacts", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return {
        "dice_clean": clean["dice"],
        "dice_degraded": degraded["dice"],
        "angular_error_clean_deg": clean["angular_error_deg"],
        "angular_error_degraded_deg": degraded["angular_error_deg"],
    }


def panel_regimes(results: list[dict], path: Path) -> dict:
    """Regime-awareness: metrics table + subvoxel orientation distribution."""
    fig, (ax_t, ax_h) = plt.subplots(
        1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    ax_t.axis("off")
    rows = [["Regime", "Dice", "Ang. err (°)", "FA", "Windows"]]
    by = {r["regime"]: r for r in results}
    for reg in ("resolved", "marginal", "subvoxel"):
        r = by.get(reg, {})
        rows.append(
            [
                reg,
                f"{r['mean_dice']:.3f}" if "mean_dice" in r else "–",
                f"{r['mean_angular_error_deg']:.2f}" if "mean_angular_error_deg" in r else "–",
                (
                    f"{r['global_fa']:.3f}"
                    if "global_fa" in r
                    else (f"{r['fa']:.3f}" if "fa" in r else "–")
                ),
                f"{r['n_windows']:,}" if "n_windows" in r else "–",
            ]
        )
    tbl = ax_t.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for c in range(len(rows[0])):
        tbl[0, c].set_facecolor("#0E7C86")
        tbl[0, c].set_text_props(color="white", weight="bold")
    ax_t.set_title(
        "Auto-selected algorithm per resolution regime", fontsize=12, weight="bold", pad=14
    )

    sub = by.get("subvoxel", {}).get("orientation_distribution")
    if sub:
        edges = np.asarray(sub["bin_edges"], dtype=float)
        counts = np.asarray(sub["counts"], dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_h.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color="#0E7C86")
        ax_h.set_xlabel("Orientation angle (°)")
        ax_h.set_ylabel("Voxel count")
        ax_h.set_title("Subvoxel orientation distribution", fontsize=12, weight="bold")
        ax_h.spines[["top", "right"]].set_visible(False)
    else:
        ax_h.axis("off")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        r["regime"]: {k: v for k, v in r.items() if k != "orientation_distribution"}
        for r in results
    }


# --------------------------------------------------------------------------- outputs


def _load_gfpa66_patch() -> tuple[np.ndarray, np.ndarray] | None:
    patches = sorted(glob.glob(str(GFPA66_PATCHES / "patch_*.npz")))
    if not patches:
        return None
    # Pick the patch with the most ground-truth fibre voxels so the slice is informative.
    best = max(patches, key=lambda p: float(np.load(p)["mask"].sum()))
    d = np.load(best)
    return d["volume"].astype(np.float64), (d["mask"] > 0).astype(np.int32)


def _is_scalar_panel(vals: dict) -> bool:
    return all(isinstance(x, (int, float, str)) for x in vals.values())


def _write_metrics(metrics: dict) -> None:
    flat: list[tuple[str, str, str]] = []
    for panel, vals in metrics.items():
        if not _is_scalar_panel(vals):
            continue
        for k, v in vals.items():
            flat.append((panel, k, f"{v:.4f}" if isinstance(v, float) else str(v)))
    with open(FIG_DIR / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "metric", "value"])
        w.writerows(flat)
    lines = ["# RAFA before/after metrics", "", "| Panel | Metric | Value |", "| --- | --- | --- |"]
    lines += [f"| {p} | {k} | {v} |" for p, k, v in flat]
    (FIG_DIR / "metrics.md").write_text("\n".join(lines) + "\n")


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


DECK_CSS = """
:root{
  --paper:#FBFBF8; --panel:#ffffff; --ink:#181A1F; --muted:#5C6470;
  --line:#E4E3DC; --accent:#0E7C86; --accent-ink:#0A5A61;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#0E1116; --panel:#161A21; --ink:#E8EAED; --muted:#9AA4B2;
         --line:#252B34; --accent:#35B7C4; --accent-ink:#8FE3EC; }
}
:root[data-theme="light"]{ --paper:#FBFBF8; --panel:#ffffff; --ink:#181A1F; --muted:#5C6470;
  --line:#E4E3DC; --accent:#0E7C86; --accent-ink:#0A5A61; }
:root[data-theme="dark"]{ --paper:#0E1116; --panel:#161A21; --ink:#E8EAED; --muted:#9AA4B2;
  --line:#252B34; --accent:#35B7C4; --accent-ink:#8FE3EC; }
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding:56px 24px 96px}
.eyebrow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.72rem;
  letter-spacing:.22em;text-transform:uppercase;color:var(--accent-ink)}
h1{font-size:clamp(1.9rem,4vw,2.9rem);margin:.28em 0 .1em;text-wrap:balance;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:60ch;font-size:1.05rem}
figure{margin:40px 0 0;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;overflow:hidden}
figure img{display:block;width:100%;height:auto}
figcaption{padding:16px 20px;border-top:1px solid var(--line)}
figcaption b{color:var(--accent-ink)}
.tag{display:inline-block;font-family:ui-monospace,monospace;font-size:.72rem;
  letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:3px 10px;margin-bottom:10px}
.metrics{width:100%;border-collapse:collapse;margin-top:8px;
  font-variant-numeric:tabular-nums;font-size:.92rem}
.metrics th,.metrics td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--line)}
.metrics th{color:var(--muted);font-weight:600;font-size:.76rem;
  letter-spacing:.06em;text-transform:uppercase}
.metrics tbody tr:last-child td{border-bottom:none}
footer{margin-top:56px;color:var(--muted);font-size:.85rem;
  border-top:1px solid var(--line);padding-top:20px}
"""


def _figure_block(img: Path, tag: str, caption_html: str) -> str:
    return (
        f'<figure><img alt="{tag}" src="data:image/png;base64,{_b64(img)}">'
        f'<figcaption><span class="tag">{tag}</span><div>{caption_html}</div></figcaption></figure>'
    )


def _deck_body(meta: dict) -> str:
    syn = meta["seg_synthetic"]
    gf = meta.get("seg_gfpa66")
    pipe = meta["pipeline"]
    rob = meta["robustness"]
    blocks = []
    if gf:
        blocks.append(
            _figure_block(
                FIG_DIR / "01_segmentation_gfpa66.png",
                "Validated segmentation · real GF-PA66 CT",
                f"Raw X-ray CT of a glass-fibre / PA66 composite segmented against its "
                f"ground-truth mask. Otsu reaches <b>Dice {gf['dice_otsu']:.3f}</b>; the "
                f"pipeline also exposes multi-Otsu (<b>{gf['dice_multiotsu']:.3f}</b>) and "
                f"adaptive (<b>{gf['dice_adaptive']:.3f}</b>) switches and reports Dice for "
                f"each, so the operator can pick and defend a method per dataset.",
            )
        )
    blocks.append(
        _figure_block(
            FIG_DIR / "01_segmentation_synthetic.png",
            "Segmentation vs ground truth · synthetic phantom",
            f"On a clean, high-contrast synthetic phantom the same thresholding is "
            f"near-exact — Otsu <b>Dice {syn['dice_otsu']:.3f}</b>, multi-Otsu "
            f"<b>{syn['dice_multiotsu']:.3f}</b> — confirming the accuracy ceiling "
            f"before real-data degradation is introduced.",
        )
    )
    blocks.append(
        _figure_block(
            FIG_DIR / "02_pipeline.png",
            "End-to-end pipeline",
            f"Raw CT to per-fibre centrelines in the resolved regime: "
            f"<b>{pipe['n_fibers']} fibres</b> segmented, labelled, skeletonised and "
            f"oriented, with a mean orientation error of <b>{pipe['angular_error_deg']:.2f}°</b> "
            f"(Dice <b>{pipe['dice']:.3f}</b>) against ground truth.",
        )
    )
    blocks.append(
        _figure_block(
            FIG_DIR / "03_robustness.png",
            "Robustness to acquisition artefacts",
            f"The same specimen with strong simulated XCT artefacts — partial-volume "
            f"blur, beam-hardening cupping, ring artefacts and Poisson noise. "
            f"Segmentation Dice on the degraded volume is "
            f"<b>{rob['dice_degraded']:.3f}</b> (clean <b>{rob['dice_clean']:.3f}</b>).",
        )
    )
    blocks.append(
        _figure_block(
            FIG_DIR / "04_regimes.png",
            "Regime awareness",
            "RAFA picks its algorithm from the voxel-size / fibre-diameter ratio: "
            "per-fibre morphometry when resolved, orientation-tensor fields when "
            "marginal or subvoxel.",
        )
    )

    def _fmt(v: object) -> str:
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    metric_rows = "".join(
        f"<tr><td>{p}</td><td>{k}</td><td>{_fmt(v)}</td></tr>"
        for p, vals in meta["metrics_flat"]
        for k, v in vals
    )
    return f"""<div class="wrap">
  <div class="eyebrow">RAFA · Regime-Aware Fiber Analysis</div>
  <h1>Fibre analysis of composites from X-ray CT: before &amp; after</h1>
  <p class="lede">Quantitative, ground-truth-validated before/after results for the
  RAFA pipeline — segmentation quality, end-to-end tracing, artefact robustness, and
  resolution-regime awareness.</p>
  {''.join(blocks)}
  <figure><figcaption><span class="tag">Metrics</span>
    <table class="metrics"><thead><tr><th>Panel</th><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{metric_rows}</tbody></table></figcaption></figure>
  <footer>Generated by <code>scripts/make_presentation.py</code> · synthetic phantom (seed 42) +
  real GF-PA66 CT · reproducible end-to-end.</footer>
</div>"""


def _write_deck(meta: dict) -> None:
    body = _deck_body(meta)
    # Body + inline <style>, no document skeleton: ready to publish as a hosted Artifact.
    (FIG_DIR / "deck_body.html").write_text(
        f"<title>RAFA — Before &amp; After</title>\n<style>{DECK_CSS}</style>\n{body}"
    )
    standalone = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>RAFA — Before &amp; After</title>"
        f"<style>{DECK_CSS}</style></head><body>{body}</body></html>"
    )
    (FIG_DIR / "index.html").write_text(standalone)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    FIG_DIR.mkdir(exist_ok=True)

    phantom = generate_fiber_phantom(
        shape=(96, 96, 96),
        n_fibers=5,
        fiber_diameter_um=6.0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    # Deterministic, visibly strong (but recoverable) XCT artefact chain for the
    # robustness story: partial-volume blur, beam-hardening cupping, rings, Poisson grain.
    rng = np.random.default_rng(7)
    degraded_vol = add_partial_volume_blur(phantom.volume, sigma_voxels=1.0)
    degraded_vol = add_beam_hardening(degraded_vol, strength=0.35, rng=rng)
    degraded_vol = add_ring_artifacts(degraded_vol, n_rings=8, strength=0.16, rng=rng)
    degraded_vol = add_poisson_noise(degraded_vol, scale=80.0, rng=rng)

    metrics: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clean = _run_resolved(phantom.volume, phantom, tmp, "clean")
        degraded = _run_resolved(degraded_vol, phantom, tmp, "degraded")
    axis = _pick_axis(clean["labels"] > 0)

    # Panel 1a: segmentation on synthetic phantom (GT = fibre labels > 0).
    syn_gt = (phantom.labels > 0).astype(np.int32)
    metrics["seg_synthetic"] = panel_segmentation(
        phantom.volume,
        syn_gt,
        FIG_DIR / "01_segmentation_synthetic.png",
        "Segmentation quality vs ground truth (synthetic phantom)",
    )
    # Panel 1b: segmentation on real GF-PA66 CT (GT = provided mask).
    gf = _load_gfpa66_patch()
    if gf is not None:
        vol, mask = gf
        metrics["seg_gfpa66"] = panel_segmentation(
            vol,
            mask,
            FIG_DIR / "01_segmentation_gfpa66.png",
            "Segmentation quality vs ground truth (real GF-PA66 CT)",
        )
    else:
        print("WARN: GF-PA66 patches not found; skipping real-data segmentation panel.")

    metrics["pipeline"] = panel_pipeline(clean, axis, FIG_DIR / "02_pipeline.png")
    metrics["robustness"] = panel_robustness(
        clean, degraded, phantom.volume, degraded_vol, axis, FIG_DIR / "03_robustness.png"
    )

    if BENCHMARK_JSON.exists():
        regime_results = json.loads(BENCHMARK_JSON.read_text())
        metrics["regimes"] = panel_regimes(regime_results, FIG_DIR / "04_regimes.png")
    else:
        print(f"WARN: {BENCHMARK_JSON} missing; run scripts/benchmark_phantoms.py first.")

    _write_metrics(metrics)
    meta = dict(metrics)
    meta["metrics_flat"] = [
        (p, list(v.items()))
        for p, v in metrics.items()
        if isinstance(v, dict) and _is_scalar_panel(v)
    ]
    _write_deck(meta)

    print("Wrote figures to", FIG_DIR)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Self-check (ponytail): the demo must actually recover fibres, or fail loudly.
    assert metrics["seg_synthetic"]["dice_otsu"] > 0.85, "synthetic Otsu Dice regressed"
    assert metrics["pipeline"]["dice"] > 0.85, "resolved pipeline Dice regressed"
    if "seg_gfpa66" in metrics:
        assert metrics["seg_gfpa66"]["dice_otsu"] > 0.0, "GF-PA66 Otsu produced empty mask"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
