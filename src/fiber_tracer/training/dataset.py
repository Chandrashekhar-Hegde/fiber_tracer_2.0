"""PyTorch dataset for mixed synthetic/real fiber XCT patches."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from fiber_tracer.training.augment import augment_patch


class FiberVolumeDataset(Dataset):
    """Load 3D patch shards produced by ``prepare_training_data.py``.

    Parameters
    ----------
    registry_path :
        Path to ``datasets.json`` registry file.
    processed_root :
        Directory containing patch subdirectories.
    split :
        ``"train"`` or ``"val"``.
    val_fraction :
        Fraction of patches per source reserved for validation.
    augment :
        Whether to apply ``augment_patch`` to training samples.
    seed :
        Random seed for reproducible train/val splits.
    """

    def __init__(
        self,
        registry_path: str | Path,
        processed_root: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        with open(registry_path) as f:
            registry = json.load(f)
        self.processed_root = Path(processed_root)
        self.augment = augment and split == "train"
        self.patch_files: list[Path] = []
        rng = random.Random(seed)
        for entry in registry:
            patch_dir = self.processed_root / entry["patch_dir"]
            if not patch_dir.exists():
                continue
            files = sorted(patch_dir.glob("*.npz"))
            if not files:
                continue
            rng.shuffle(files)
            n_val = max(1, int(len(files) * val_fraction)) if len(files) > 1 else 0
            if split == "train":
                files = files[n_val:]
            else:
                files = files[:n_val]
            self.patch_files.extend(files)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.patch_files[idx])
        volume = data["volume"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        if self.augment:
            volume, mask = augment_patch(
                volume,
                mask,
                seed=self.rng.randint(0, 2**31 - 1),
            )
        return torch.from_numpy(volume).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
