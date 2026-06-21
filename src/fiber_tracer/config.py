"""Configuration management with validation and units."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import os
import json
import yaml


@dataclass
class VoxelSpacing:
    z: float
    y: float
    x: float

    def is_isotropic(self, tol: float = 1e-3) -> bool:
        return max(abs(self.z - self.y), abs(self.z - self.x), abs(self.y - self.x)) < tol


@dataclass
class ProcessingConfig:
    denoise_sigma: Optional[float] = None
    normalize: bool = True
    anisotropic_spacing: Optional[VoxelSpacing] = None


@dataclass
class SegmentationConfig:
    method: str = "otsu"  # otsu, watershed, adaptive, unet
    min_fiber_diameter_um: float = 10.0
    max_fiber_diameter_um: float = 50.0
    watershed_seed_sigma_um: Optional[float] = None


@dataclass
class OrientationConfig:
    method: str = "structure_tensor"  # structure_tensor, pca
    sigma_um: Optional[float] = None
    rho_um: Optional[float] = None
    window_size_um: Optional[float] = None


@dataclass
class AnalysisConfig:
    compute_morphometry: bool = True
    compute_orientation_tensor: bool = True
    compute_tda_descriptors: bool = False


@dataclass
class Config:
    data_path: str = ""
    output_dir: str = ""
    voxel_spacing_um: VoxelSpacing = field(default_factory=lambda: VoxelSpacing(1.0, 1.0, 1.0))
    fiber_diameter_um: float = 10.0
    regime: str = "auto"  # auto, resolved, marginal, subvoxel
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def validate(self) -> None:
        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError(f"data_path does not exist: {self.data_path}")
        if self.fiber_diameter_um <= 0:
            raise ValueError("fiber_diameter_um must be positive")
        for s in (self.voxel_spacing_um.z, self.voxel_spacing_um.y, self.voxel_spacing_um.x):
            if s <= 0:
                raise ValueError("voxel spacing must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            if path.endswith((".yaml", ".yml")):
                yaml.safe_dump(self.to_dict(), f)
            else:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)
