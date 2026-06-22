"""Tests for the mixed fiber patch dataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fiber_tracer.training.dataset import FiberVolumeDataset


def test_dataset_loads_patch():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        patch_dir = root / "synthetic"
        patch_dir.mkdir()
        np.savez_compressed(
            patch_dir / "p_0000.npz",
            volume=np.random.rand(64, 64, 64).astype(np.float32),
            mask=np.ones((64, 64, 64), dtype=np.float32),
        )
        registry = [
            {
                "name": "synthetic",
                "type": "synthetic",
                "n_patches": 1,
                "patch_dir": "synthetic",
            }
        ]
        registry_path = root / "datasets.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        ds = FiberVolumeDataset(registry_path, root, split="train", augment=False)
        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (1, 64, 64, 64)
        assert y.shape == (1, 64, 64, 64)
