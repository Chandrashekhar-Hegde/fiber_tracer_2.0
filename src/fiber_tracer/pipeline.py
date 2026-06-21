"""Pipeline orchestrator for RAFA."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from fiber_tracer.config import Config
from fiber_tracer.io import load_tiff_stack, get_shape_info, save_tiff_stack
from fiber_tracer.preprocess import normalize_intensity, gaussian_denoise
from fiber_tracer.regime import detect_regime
from fiber_tracer.segmentation.classical import segment_otsu_3d, segment_watershed_3d
from fiber_tracer.centerline.skeleton import skeletonize_label_volume
from fiber_tracer.analysis.morphometry import per_fiber_volumes, equivalent_diameter_from_volume
from fiber_tracer.orientation.pca import pca_orientation

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
        spacing = (self.config.voxel_spacing_um.z, self.config.voxel_spacing_um.y, self.config.voxel_spacing_um.x)
        logger.info(get_shape_info(raw, spacing))

        volume = normalize_intensity(raw)
        if self.config.processing.denoise_sigma:
            volume = gaussian_denoise(volume, self.config.processing.denoise_sigma, self.config.voxel_spacing_um)

        regime = self.config.regime if self.config.regime != "auto" else detect_regime(self.config)
        logger.info(f"Selected regime: {regime}")

        if regime == "resolved":
            mask = segment_otsu_3d(volume)
            labels = segment_watershed_3d(mask)
            skeleton = skeletonize_label_volume(labels)

            # Per-fiber properties
            volumes = per_fiber_volumes(labels)
            spacing = (self.config.voxel_spacing_um.z, self.config.voxel_spacing_um.y, self.config.voxel_spacing_um.x)
            fibers = []
            for label_id, n_voxels in volumes.items():
                coords = np.argwhere(labels == label_id).astype(np.float32)
                orientation = pca_orientation(coords)
                diameter = equivalent_diameter_from_volume(n_voxels, spacing)
                fibers.append({
                    "label": int(label_id),
                    "n_voxels": int(n_voxels),
                    "equivalent_diameter_um": float(diameter),
                    "orientation": orientation.tolist(),
                })

            save_tiff_stack(out / "normalized_input.tif", volume)
            save_tiff_stack(out / "labels.tif", labels)
            save_tiff_stack(out / "skeleton.tif", skeleton.astype(np.uint8) * 255)
            summary = {
                "regime": regime,
                "n_labels": len(fibers),
                "voxel_spacing_um": spacing,
                "fibers": fibers,
            }
            with open(out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            self.volume = volume
            self.labels = labels
            return summary
        elif regime == "marginal":
            raise NotImplementedError("Marginal regime is planned for Phase 2")
        else:
            raise NotImplementedError("Subvoxel regime is planned for Phase 2")
