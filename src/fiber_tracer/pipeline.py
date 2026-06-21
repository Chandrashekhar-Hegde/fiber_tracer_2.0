"""Pipeline orchestrator for RAFA."""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import ndimage

from fiber_tracer.analysis.morphometry import equivalent_diameter_from_volume, per_fiber_volumes
from fiber_tracer.centerline.skeleton import skeletonize_label_volume
from fiber_tracer.config import Config
from fiber_tracer.io import get_shape_info, load_tiff_stack, save_tiff_stack
from fiber_tracer.orientation.pca import pca_orientation
from fiber_tracer.orientation.structure_tensor import (
    compute_local_orientation_field,
    orientation_from_smallest_eigenvector,
)
from fiber_tracer.orientation.tensor import (
    aggregate_direction_tensor,
    fractional_anisotropy,
    windowed_orientation_tensor_field,
)
from fiber_tracer.preprocess import gaussian_denoise, normalize_intensity
from fiber_tracer.regime import detect_regime
from fiber_tracer.reporting import (
    CITATIONS,
    REGIME_CAVEATS,
    write_csv_report,
    write_html_report,
    write_json_report,
)
from fiber_tracer.segmentation.classical import (
    segment_connected_components_3d,
    segment_otsu_3d,
    segment_watershed_3d,
)

logger = logging.getLogger(__name__)


class FiberAnalysisPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.volume: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def run(self) -> dict:
        self.config.validate()
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        raw = load_tiff_stack(self.config.data_path)
        spacing = (
            self.config.voxel_spacing_um.z,
            self.config.voxel_spacing_um.y,
            self.config.voxel_spacing_um.x,
        )
        logger.info(get_shape_info(raw, spacing))

        if self.config.processing.normalize:
            volume = normalize_intensity(raw)
        else:
            raw_max = raw.max()
            if raw_max > 0:
                volume = raw.astype(np.float32, copy=False) / raw_max
            else:
                volume = raw.astype(np.float32, copy=False)
        if self.config.processing.denoise_sigma:
            volume = gaussian_denoise(
                volume, self.config.processing.denoise_sigma, self.config.voxel_spacing_um
            )

        regime = self.config.regime if self.config.regime != "auto" else detect_regime(self.config)
        logger.info(f"Selected regime: {regime}")

        if regime == "resolved":
            return self._run_resolved(volume, out)
        elif regime == "marginal":
            return self._run_marginal(volume, out)
        elif regime == "subvoxel":
            return self._run_subvoxel(volume, out)
        else:
            raise ValueError(f"unsupported regime: {regime}")

    def _compute_local_directions(
        self,
        volume: np.ndarray,
        rho_um: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the structure-tensor orientation field and apply an Otsu foreground mask.

        Parameters
        ----------
        rho_um :
            Optional integration scale in micrometres. When ``None``, falls back
            to ``self.config.orientation.rho_um`` and then to half the fiber
            diameter.

        Returns
        -------
        directions :
            Array of shape ``(N, 3)`` containing unit direction vectors for the
            foreground voxels.
        mask :
            Binary foreground mask from Otsu thresholding.
        """
        sigma_um = self.config.orientation.sigma_um
        if sigma_um is None:
            sigma_um = min(
                self.config.voxel_spacing_um.z,
                self.config.voxel_spacing_um.y,
                self.config.voxel_spacing_um.x,
            )

        if rho_um is None:
            rho_um = self.config.orientation.rho_um
        if rho_um is None:
            rho_um = self.config.fiber_diameter_um / 2.0

        eigenvalues, eigenvectors = compute_local_orientation_field(
            volume,
            sigma_um=sigma_um,
            rho_um=rho_um,
            voxel_spacing=self.config.voxel_spacing_um,
        )
        direction_field = orientation_from_smallest_eigenvector(eigenvectors)

        mask = segment_otsu_3d(volume)
        if not np.any(mask):
            return np.zeros((0, 3), dtype=np.float64), mask

        directions = direction_field[:, mask].T
        # Normalize to unit vectors (safeguard against numerical noise).
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        nonzero = norms.squeeze(axis=1) > 0
        directions[nonzero] = directions[nonzero] / norms[nonzero]
        return directions, mask

    def _resolve_window_size_voxels(self, window_size_um: float) -> int:
        """Convert a physical window size to an odd voxel size >= 3."""
        min_spacing = min(
            self.config.voxel_spacing_um.z,
            self.config.voxel_spacing_um.y,
            self.config.voxel_spacing_um.x,
        )
        window_size = int(round(window_size_um / min_spacing))
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            window_size = 3
        return window_size

    def _run_resolved(self, volume: np.ndarray, out: Path) -> dict:
        """Resolved-regime pipeline: segmentation, labeling, skeletonization."""
        mask = segment_otsu_3d(volume)
        # Remove small spurious foreground voxels and smooth boundaries while
        # keeping well-separated fibers distinct.
        mask = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1))
        if self.config.segmentation.method == "watershed":
            labels = segment_watershed_3d(mask)
        else:
            labels = segment_connected_components_3d(mask)
        skeleton = skeletonize_label_volume(labels)

        # Per-fiber properties
        spacing = (
            self.config.voxel_spacing_um.z,
            self.config.voxel_spacing_um.y,
            self.config.voxel_spacing_um.x,
        )
        volumes = per_fiber_volumes(labels)
        fibers: list[dict[str, Any]] = []
        for label_id, n_voxels in volumes.items():
            fiber: dict[str, Any] = {
                "label": int(label_id),
                "n_voxels": int(n_voxels),
            }
            if self.config.analysis.compute_orientation_tensor:
                coords = np.argwhere(labels == label_id).astype(np.float32)
                orientation = pca_orientation(coords)
                fiber["orientation"] = orientation.tolist()
            if self.config.analysis.compute_morphometry:
                diameter = equivalent_diameter_from_volume(n_voxels, spacing)
                fiber["equivalent_diameter_um"] = float(diameter)
            fibers.append(fiber)

        save_tiff_stack(out / "normalized_input.tif", volume)
        save_tiff_stack(out / "labels.tif", labels)
        save_tiff_stack(out / "skeleton.tif", skeleton.astype(np.uint8) * 255)
        summary: dict[str, Any] = {
            "regime": "resolved",
            "n_labels": len(fibers),
            "voxel_spacing_um": spacing,
            "fibers": fibers,
        }
        notes = []
        if not self.config.analysis.compute_morphometry:
            notes.append("Morphometry disabled; equivalent diameter not computed.")
        if not self.config.analysis.compute_orientation_tensor:
            notes.append(
                "Orientation tensor analysis disabled; per-fiber orientation not computed."
            )
        if notes:
            summary["notes"] = " ".join(notes)
        summary["config"] = self.config.to_dict()
        summary["citations"] = CITATIONS
        summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
        write_json_report(out / "summary.json", summary)
        write_csv_report(out / "report.csv", summary)
        write_html_report(out / "report.html", summary)
        self.volume = volume
        self.labels = labels
        return summary

    def _run_marginal(self, volume: np.ndarray, out: Path) -> dict:
        """Marginal-regime pipeline: windowed second-order orientation tensor field."""
        summary: dict[str, Any]
        if not self.config.analysis.compute_orientation_tensor:
            mask = segment_otsu_3d(volume)
            summary = {
                "regime": "marginal",
                "n_voxels": int(mask.sum()),
                "note": "Orientation tensor analysis disabled; A2/FA/distribution not computed.",
            }
            summary["config"] = self.config.to_dict()
            summary["citations"] = CITATIONS
            summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
            write_json_report(out / "summary.json", summary)
            write_csv_report(out / "report.csv", summary)
            write_html_report(out / "report.html", summary)
            return summary

        directions, mask = self._compute_local_directions(volume)

        if directions.shape[0] == 0:
            empty_map = np.zeros((0, 0, 0, 3, 3), dtype=np.float64)
            empty_centers = np.zeros((0, 0, 0, 3), dtype=np.int64)
            np.save(out / "a2_map.npy", empty_map)
            np.save(out / "a2_centers.npy", empty_centers)
            summary = {
                "regime": "marginal",
                "n_voxels": 0,
                "a2_map_shape": tuple(empty_map.shape),
                "a2_map_file": "a2_map.npy",
                "a2_centers_file": "a2_centers.npy",
                "a2": np.zeros((3, 3)).tolist(),
                "a2_windows": [],
            }
            summary["config"] = self.config.to_dict()
            summary["citations"] = CITATIONS
            summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
            write_json_report(out / "summary.json", summary)
            write_csv_report(out / "report.csv", summary)
            write_html_report(out / "report.html", summary)
            return summary

        window_size_um = self.config.orientation.window_size_um
        if window_size_um is None:
            window_size_um = self.config.fiber_diameter_um
        window_size = self._resolve_window_size_voxels(window_size_um)

        # Build a full spatial direction field for windowed aggregation.
        direction_field = np.zeros((3,) + volume.shape, dtype=np.float64)
        direction_field[:, mask] = directions.T

        a2_map, a2_centers = windowed_orientation_tensor_field(
            direction_field, window_size=window_size
        )
        global_a2 = aggregate_direction_tensor(directions)

        np.save(out / "a2_map.npy", a2_map)
        np.save(out / "a2_centers.npy", a2_centers)

        a2_windows = []
        for window_id, (i, j, k) in enumerate(np.ndindex(a2_map.shape[:3])):
            cz, cy, cx = a2_centers[i, j, k]
            tensor = a2_map[i, j, k]
            a2_windows.append(
                {
                    "window_id": window_id,
                    "center_z": int(cz),
                    "center_y": int(cy),
                    "center_x": int(cx),
                    "fa": float(fractional_anisotropy(tensor)),
                    "a2_00": float(tensor[0, 0]),
                    "a2_11": float(tensor[1, 1]),
                    "a2_22": float(tensor[2, 2]),
                }
            )

        summary = {
            "regime": "marginal",
            "n_voxels": int(directions.shape[0]),
            "a2_map": a2_map.tolist(),
            "a2_map_shape": tuple(a2_map.shape),
            "a2_map_file": "a2_map.npy",
            "a2_centers_file": "a2_centers.npy",
            "a2": global_a2.tolist(),
            "a2_windows": a2_windows,
        }
        summary["config"] = self.config.to_dict()
        summary["citations"] = CITATIONS
        summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
        write_json_report(out / "summary.json", summary)
        write_csv_report(out / "report.csv", summary)
        write_html_report(out / "report.html", summary)
        return summary

    def _run_subvoxel(self, volume: np.ndarray, out: Path) -> dict:
        """Subvoxel-regime pipeline: global orientation tensor and distribution."""
        summary: dict[str, Any]
        if not self.config.analysis.compute_orientation_tensor:
            mask = segment_otsu_3d(volume)
            summary = {
                "regime": "subvoxel",
                "n_voxels": int(mask.sum()),
                "note": "Orientation tensor analysis disabled; A2/FA/distribution not computed.",
            }
            summary["config"] = self.config.to_dict()
            summary["citations"] = CITATIONS
            summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
            write_json_report(out / "summary.json", summary)
            write_csv_report(out / "report.csv", summary)
            write_html_report(out / "report.html", summary)
            return summary

        # Use a larger integration scale for the subvoxel regime.
        original_rho_um = self.config.orientation.rho_um
        if original_rho_um is None:
            original_rho_um = self.config.fiber_diameter_um / 2.0
        min_spacing = min(
            self.config.voxel_spacing_um.z,
            self.config.voxel_spacing_um.y,
            self.config.voxel_spacing_um.x,
        )
        rho_um = max(original_rho_um, 3.0 * min_spacing)

        # Pass the larger integration scale directly instead of mutating config.
        directions, mask = self._compute_local_directions(volume, rho_um=rho_um)

        if directions.shape[0] == 0:
            summary = {
                "regime": "subvoxel",
                "n_voxels": 0,
                "a2": np.zeros((3, 3)).tolist(),
                "fa": 0.0,
                "orientation_distribution": {
                    "bin_edges": [0.0, 90.0],
                    "counts": [0],
                    "principal_axis": [0.0, 0.0, 1.0],
                },
            }
            summary["config"] = self.config.to_dict()
            summary["citations"] = CITATIONS
            summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
            write_json_report(out / "summary.json", summary)
            write_csv_report(out / "report.csv", summary)
            write_html_report(out / "report.html", summary)
            return summary

        a2 = aggregate_direction_tensor(directions)
        fa = fractional_anisotropy(a2)

        # Principal axis is the eigenvector for the largest eigenvalue.
        evals, evecs = np.linalg.eigh(a2)
        principal_axis = evecs[:, -1]

        # Orientation distribution: angles relative to the principal axis.
        # Use einsum instead of @ to avoid spurious BLAS warnings on macOS.
        dots = np.clip(np.abs(np.einsum("ij,j->i", directions, principal_axis)), 0.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))
        counts, bin_edges = np.histogram(angles_deg, bins=18, range=(0.0, 90.0))

        summary = {
            "regime": "subvoxel",
            "n_voxels": int(directions.shape[0]),
            "a2": a2.tolist(),
            "fa": float(fa),
            "orientation_distribution": {
                "bin_edges": bin_edges.tolist(),
                "counts": counts.tolist(),
                "principal_axis": principal_axis.tolist(),
            },
        }
        summary["config"] = self.config.to_dict()
        summary["citations"] = CITATIONS
        summary["caveats"] = REGIME_CAVEATS.get(summary["regime"], "No specific caveats.")
        write_json_report(out / "summary.json", summary)
        write_csv_report(out / "report.csv", summary)
        write_html_report(out / "report.html", summary)
        return summary
