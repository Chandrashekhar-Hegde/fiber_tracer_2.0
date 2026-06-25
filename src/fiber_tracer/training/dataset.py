"""Dataset for mixed synthetic/real fiber XCT patches.

The dataset returns NumPy arrays so that it can be tested and inspected without
installing PyTorch.  Training scripts are responsible for converting samples to
``torch.Tensor`` before feeding them to a model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from fiber_tracer.training.augment import augment_patch


class FiberVolumeDataset:
    """Load 3D patch shards produced by ``prepare_training_data.py``.

    This class intentionally does not inherit from ``torch.utils.data.Dataset``
    and does not import PyTorch, so it remains usable in environments where the
    ``ml`` extra is not installed.

    Parameters
    ----------
    registry_path :
        Path to ``datasets.json`` registry file.
    processed_root :
        Directory containing patch subdirectories.
    split :
        ``"train"`` or ``"val"``.
    val_fraction :
        Fraction of patches or sources reserved for validation.
    augment :
        Whether to apply ``augment_patch`` to training samples.
    seed :
        Random seed for reproducible train/val splits.
    split_mode :
        ``"patch"`` (default, legacy) or ``"volume"`` (recommended).  ``"volume"``
        reserves whole source directories for validation to avoid information
        leakage between overlapping patches.
    """

    def __init__(
        self,
        registry_path: str | Path,
        processed_root: str | Path,
        split: str = "train",
        val_fraction: float = 0.1,
        augment: bool = True,
        seed: int = 42,
        split_mode: str = "patch",
    ) -> None:
        with open(registry_path) as f:
            registry = json.load(f)
        self.processed_root = Path(processed_root)
        self.augment = augment and split == "train"
        self.patch_files: list[Path] = []
        rng = random.Random(seed)

        if split_mode == "volume":
            # Split at source level to avoid leakage.
            entries = list(registry)
            rng.shuffle(entries)
            n_val = max(1, int(len(entries) * val_fraction)) if len(entries) > 1 else 0
            if split == "train":
                selected = entries[n_val:]
            else:
                selected = entries[:n_val]
            for entry in selected:
                patch_dir = self.processed_root / entry["patch_dir"]
                if not patch_dir.exists():
                    continue
                self.patch_files.extend(sorted(patch_dir.glob("*.npz")))
        elif split_mode == "patch":
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
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(self.patch_files[idx])
        volume = data["volume"].astype(np.float32)
        mask = data["mask"].astype(np.float32)
        if self.augment:
            volume, mask = augment_patch(
                volume,
                mask,
                seed=self.rng.randint(0, 2**31 - 1),
            )
        return volume[np.newaxis, ...], mask[np.newaxis, ...]


def numpy_collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Collate a list of (volume, mask) NumPy samples into batched arrays."""
    volumes = np.stack([item[0] for item in batch], axis=0)
    masks = np.stack([item[1] for item in batch], axis=0)
    return volumes, masks
