"""Validate a trained 3D U-Net against the GF-PA66 ground-truth labels.

License: CC BY-SA 4.0
DOI: 10.5281/zenodo.4587827
Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fiber_tracer.backends.unet3d import UNet3D


class SlidingWindowDataset(Dataset):
    """Extract overlapping 3D patches from a volume for inference."""

    def __init__(
        self,
        volume: np.ndarray,
        patch_size: tuple[int, int, int] = (64, 64, 64),
        stride: tuple[int, int, int] = (32, 32, 32),
    ) -> None:
        self.volume = volume.astype(np.float32)
        self.patch_size = patch_size
        self.stride = stride
        self.indices = self._compute_indices()

    def _compute_indices(self) -> list[tuple[int, int, int]]:
        d, h, w = self.volume.shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        coords = []
        for z in range(0, d - pd + 1, sd):
            for y in range(0, h - ph + 1, sh):
                for x in range(0, w - pw + 1, sw):
                    coords.append((z, y, x))
        # Ensure the final patches reach the boundary.
        last_z = max(0, d - pd)
        last_y = max(0, h - ph)
        last_x = max(0, w - pw)
        if (last_z, 0, 0) not in coords:
            for y in range(0, last_y + 1, sh):
                for x in range(0, last_x + 1, sw):
                    coords.append((last_z, y, x))
        if not any(c[1] == last_y for c in coords):
            for z in range(0, last_z + 1, sd):
                for x in range(0, last_x + 1, sw):
                    coords.append((z, last_y, x))
        if not any(c[2] == last_x for c in coords):
            for z in range(0, last_z + 1, sd):
                for y in range(0, last_y + 1, sh):
                    coords.append((z, y, last_x))
        return list(set(coords))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        z, y, x = self.indices[idx]
        patch = self.volume[
            z : z + self.patch_size[0], y : y + self.patch_size[1], x : x + self.patch_size[2]
        ]
        patch = (patch - patch.mean()) / (patch.std() + 1e-8)
        tensor = torch.from_numpy(patch[None, ...])
        return tensor, (z, y, x)


def _dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return float((2.0 * intersection + eps) / (union + eps))


def _iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def _load_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu")
    features = ckpt.get("features", (16, 32, 64, 128))
    norm = ckpt.get("norm", "batch")
    model = UNet3D(in_channels=1, out_channels=1, features=features, norm=norm)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def _sliding_window_inference(
    model: nn.Module,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
    batch_size: int = 1,
) -> np.ndarray:
    model.eval()
    d, h, w = volume.shape
    output = np.zeros((d, h, w), dtype=np.float32)
    counts = np.zeros((d, h, w), dtype=np.float32)

    dataset = SlidingWindowDataset(volume, patch_size, stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch, coords in tqdm(loader, desc="inference"):
            batch = batch.to(device)
            probs = model(batch).cpu().numpy()
            for i, (z, y, x) in enumerate(coords):
                pd, ph, pw = patch_size
                output[z : z + pd, y : y + ph, x : x + pw] += probs[i, 0]
                counts[z : z + pd, y : y + ph, x : x + pw] += 1.0

    output /= np.maximum(counts, 1.0)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate U-Net on GF-PA66")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/fiber_unet_v2_full.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/gfpa66/pa66_volumes.h5"))
    parser.add_argument("--image-key", default="pa66")
    parser.add_argument("--label-key", default="ground_truth")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--subsample", type=int, default=None, help="Evaluate on N central slices")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    if not args.data.is_file():
        print(f"ERROR: data file not found: {args.data}")
        return 1
    if not args.checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return 1

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = _load_checkpoint(args.checkpoint)
    model = model.to(device)
    patch_size = ckpt.get("patch_size", (args.patch_size, args.patch_size, args.patch_size))

    print(f"Loading GF-PA66 volume: {args.data}")
    with h5py.File(args.data, "r") as f:
        volume = f[args.image_key][()]
        labels = f[args.label_key][()]

    print(f"Volume shape: {volume.shape}, label shape: {labels.shape}")
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    if labels.ndim == 4 and labels.shape[0] == 1:
        labels = labels[0]

    # Binarize labels: GF-PA66 ground truth has classes 0,1,2.
    target = (labels > 0).astype(np.uint8)

    # Optional: evaluate on central slices for speed.
    z_slice = slice(None)
    if args.subsample:
        d = volume.shape[0]
        mid = d // 2
        half = args.subsample // 2
        z_start = max(0, mid - half)
        z_end = min(d, mid + half)
        volume = volume[z_start:z_end]
        target = target[z_start:z_end]
        z_slice = slice(z_start, z_end)
        print(f"Evaluating on slices {z_start}-{z_end}")

    print("Running sliding-window inference...")
    stride = (args.stride, args.stride, args.stride)
    probs = _sliding_window_inference(model, volume, patch_size, stride, device, args.batch_size)

    pred = (probs > args.threshold).astype(np.uint8)

    print("\nMetrics:")
    print(f"  foreground voxels (target): {target.sum():,}")
    print(f"  foreground voxels (pred):   {pred.sum():,}")
    print(f"  Dice:  {_dice_score(pred, target):.4f}")
    print(f"  IoU:   {_iou_score(pred, target):.4f}")
    print(f"  accuracy: {(pred == target).mean():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
