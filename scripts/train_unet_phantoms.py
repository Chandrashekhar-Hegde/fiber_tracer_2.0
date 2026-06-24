"""Train a lightweight 3D U-Net on synthetic fiber phantoms.

This script is self-contained: it generates its own ground-truth labels using
``fiber_tracer.validation.phantoms`` and writes a PyTorch checkpoint that can
be loaded by ``MLSegmentationBackend``.

Example
-------
    python scripts/train_unet_phantoms.py --epochs 30 --output models/fiber_unet.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.validation.phantoms import generate_fiber_phantom


class PhantomPatchDataset(Dataset):
    """Dataset that generates random 32³ patches from synthetic phantoms."""

    def __init__(
        self,
        n_phantoms: int,
        patches_per_phantom: int,
        patch_size: tuple[int, int, int] = (32, 32, 32),
        phantom_shape: tuple[int, int, int] = (64, 64, 64),
        seed: int | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.phantom_shape = phantom_shape
        self.patches_per_phantom = patches_per_phantom
        self.n_phantoms = n_phantoms
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_phantoms * self.patches_per_phantom

    def _generate_phantom(self) -> tuple[np.ndarray, np.ndarray]:
        n_fibers = self.rng.randint(3, 10)
        fiber_diameter_um = self.rng.uniform(2.0, 6.0)
        noise_std = self.rng.uniform(0.01, 0.05)
        seed = self.rng.randint(0, 2**31 - 1)
        phantom = generate_fiber_phantom(
            shape=self.phantom_shape,
            n_fibers=n_fibers,
            fiber_diameter_um=fiber_diameter_um,
            voxel_spacing_um=(1.0, 1.0, 1.0),
            noise_std=noise_std,
            seed=seed,
        )
        # Return grayscale volume and binary foreground mask.
        return phantom.volume.astype(np.float32), (phantom.labels > 0).astype(np.float32)

    def _random_patch(self, volume: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pd, ph, pw = self.patch_size
        d = self.np_rng.integers(0, volume.shape[0] - pd + 1)
        h = self.np_rng.integers(0, volume.shape[1] - ph + 1)
        w = self.np_rng.integers(0, volume.shape[2] - pw + 1)
        patch = volume[d : d + pd, h : h + ph, w : w + pw]
        target = mask[d : d + pd, h : h + ph, w : w + pw]
        return patch, target

    def _augment(self, patch: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Random axis flips.
        for axis in range(3):
            if self.rng.random() > 0.5:
                patch = np.flip(patch, axis=axis).copy()
                target = np.flip(target, axis=axis).copy()
        # Random 90-degree rotation around z-axis.
        k = self.rng.randint(0, 3)
        if k:
            patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()
        return patch, target

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Regenerate the same phantom for all patches belonging to it.
        phantom_idx = idx // self.patches_per_phantom
        patch_idx = idx % self.patches_per_phantom
        local_rng = random.Random(phantom_idx)

        phantom = generate_fiber_phantom(
            shape=self.phantom_shape,
            n_fibers=local_rng.randint(3, 10),
            fiber_diameter_um=local_rng.uniform(2.0, 6.0),
            voxel_spacing_um=(1.0, 1.0, 1.0),
            noise_std=local_rng.uniform(0.01, 0.05),
            seed=local_rng.randint(0, 2**31 - 1),
        )
        volume = phantom.volume.astype(np.float32)
        mask = (phantom.labels > 0).astype(np.float32)

        # Deterministically pick a patch based on patch_idx.
        pd, ph, pw = self.patch_size
        d = (patch_idx * 17) % (volume.shape[0] - pd + 1)
        h = (patch_idx * 31) % (volume.shape[1] - ph + 1)
        w = (patch_idx * 47) % (volume.shape[2] - pw + 1)
        patch = volume[d : d + pd, h : h + ph, w : w + pw]
        target = mask[d : d + pd, h : h + ph, w : w + pw]

        # Apply random augmentation with a per-patch seed.
        aug_rng = random.Random(idx)
        for axis in range(3):
            if aug_rng.random() > 0.5:
                patch = np.flip(patch, axis=axis).copy()
                target = np.flip(target, axis=axis).copy()
        k = aug_rng.randint(0, 3)
        if k:
            patch = np.rot90(patch, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(1, 2)).copy()

        return (
            torch.from_numpy(patch).unsqueeze(0),
            torch.from_numpy(target).unsqueeze(0),
        )


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return cast(torch.Tensor, (2.0 * intersection + eps) / (union + eps))  # type: ignore[no-any-return]


class BCEDiceLoss(nn.Module):
    """Combined binary cross-entropy and Dice loss."""

    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        dice = 1.0 - _dice_score(pred, target).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice  # type: ignore[no-any-return]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    criterion = BCEDiceLoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_dice += _dice_score(outputs, targets).mean().item()
            n_batches += 1
    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_dice / max(n_batches, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a 3D U-Net on synthetic fiber phantoms")
    parser.add_argument("--n-phantoms", type=int, default=200, help="Number of training phantoms")
    parser.add_argument("--patches-per-phantom", type=int, default=16, help="Patches per phantom")
    parser.add_argument("--patch-size", type=int, default=32, help="Cubic patch edge length")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction")
    parser.add_argument(
        "--output", type=Path, default=Path("models/fiber_unet.pt"), help="Checkpoint path"
    )
    parser.add_argument("--device", default="auto", help="torch device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    print(f"Training on device: {device}")

    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    full_dataset = PhantomPatchDataset(
        n_phantoms=args.n_phantoms,
        patches_per_phantom=args.patches_per_phantom,
        patch_size=patch_size,
        seed=args.seed,
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet3D(in_channels=1, out_channels=1, features=(8, 16, 32)).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        train_metrics = {"loss": epoch_loss / max(len(train_loader), 1)}
        val_metrics = _evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": (8, 16, 32),
                    "patch_size": patch_size,
                    "val_dice": best_dice,
                },
                args.output,
            )
            print(f"  -> Saved new best checkpoint to {args.output}")

    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
