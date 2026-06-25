"""Dataset loader for the synthetic FiberTracer-X pre-training corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticCorpusDataset(Dataset):
    """Load synthetic XCT patches produced by ``scripts/generate_synthetic_corpus.py``.

    Each sample contains:
    - ``volume``: augmented input volume (D, H, W)
    - ``semantic``: semantic mask (0=matrix, 1=fiber, 2=void)
    - ``a2``: 3x3 orientation tensor
    - ``metadata``: sample architecture/material/diameter
    """

    def __init__(
        self,
        corpus_dir: str,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        registry_path = self.corpus_dir / "corpus.json"
        with open(registry_path) as f:
            registry = json.load(f)
        self.samples = registry["samples"]
        self.patch_size = tuple(registry["patch_size"])
        self.voxel_spacing_um = tuple(registry["voxel_spacing_um"])

        rng = np.random.default_rng(seed)
        n = len(self.samples)
        perm = rng.permutation(n)
        n_val = int(n * val_fraction)
        if split == "train":
            self.indices = perm[n_val:].tolist()
        elif split == "val":
            self.indices = perm[:n_val].tolist()
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        sample = self.samples[self.indices[idx]]
        data = np.load(self.corpus_dir / sample["file"])
        return {
            "volume": data["volume"].astype(np.float32),
            "semantic": data["semantic"].astype(np.int64),
            "a2": data["a2"].astype(np.float32),
            "metadata": sample,
        }


def synthetic_collate(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Collate a list of synthetic samples into tensors."""
    volumes = np.stack([b["volume"] for b in batch], axis=0)  # B, D, H, W
    semantic = np.stack([b["semantic"] for b in batch], axis=0)  # B, D, H, W
    a2 = np.stack([b["a2"] for b in batch], axis=0)  # B, 3, 3
    return {
        "volume": torch.from_numpy(volumes).unsqueeze(1).float(),  # B, 1, D, H, W
        "semantic": torch.from_numpy(semantic).long(),
        "a2": torch.from_numpy(a2).float(),
    }
