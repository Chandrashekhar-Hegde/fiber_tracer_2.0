"""Train a 3D U-Net on mixed synthetic + real fiber XCT patches."""

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
from torch.utils.data import DataLoader
from tqdm import tqdm

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.training.dataset import FiberVolumeDataset


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.view(pred.size(0), -1)
    target = target.view(pred.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return cast(torch.Tensor, (2.0 * intersection + eps) / (union + eps))  # type: ignore[no-any-return]


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        bce = -(
            self.pos_weight * target * torch.log(pred + eps)
            + (1.0 - target) * torch.log(1.0 - pred + eps)
        ).mean()
        dice = 1.0 - _dice_score(pred, target).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice  # type: ignore[no-any-return]


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    foreground_threshold: float = 0.001,
) -> dict[str, float]:
    model.eval()
    criterion = BCEDiceLoss()
    total_loss = 0.0
    total_soft_dice = 0.0
    total_hard_dice = 0.0
    total_fg_hard_dice = 0.0
    n_batches = 0
    n_fg_batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_soft_dice += _dice_score(outputs, targets).mean().item()
            hard_pred = (outputs > 0.5).float()
            total_hard_dice += _dice_score(hard_pred, targets).mean().item()
            if targets.mean() > foreground_threshold:
                total_fg_hard_dice += _dice_score(hard_pred, targets).mean().item()
                n_fg_batches += 1
            n_batches += 1
    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": total_soft_dice / max(n_batches, 1),
        "hard_dice": total_hard_dice / max(n_batches, 1),
        "fg_hard_dice": total_fg_hard_dice / max(n_fg_batches, 1) if n_fg_batches else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train U-Net on mixed fiber XCT patches")
    parser.add_argument("--registry", type=Path, default=Path("data/processed/datasets.json"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("models/fiber_unet_v2.pt"))
    parser.add_argument("--features", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--norm", default="batch", choices=["batch", "instance"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Training on device: {device}")

    train_ds = FiberVolumeDataset(
        args.registry,
        args.processed_root,
        split="train",
        augment=True,
        seed=args.seed,
    )
    val_ds = FiberVolumeDataset(
        args.registry,
        args.processed_root,
        split="val",
        augment=False,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Estimate foreground ratio from the training set to weight BCE.
    fg_ratios = [targets.mean().item() for _, targets in train_ds]
    mean_fg = float(np.mean(fg_ratios)) if fg_ratios else 0.5
    pos_weight = float((1.0 - mean_fg) / (mean_fg + 1e-8))
    print(f"Training foreground ratio: {mean_fg:.4f}, BCE pos_weight: {pos_weight:.2f}")

    features = tuple(args.features)
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        features=features,
        dropout=args.dropout,
        norm=args.norm,
    ).to(device)
    criterion = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # AMP on MPS does not support ConvTranspose3D in fp16/bf16, so only
    # enable AMP on CUDA where the full 3D U-Net op set is supported.
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_fg_dice = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        scheduler.step()

        train_metrics = {"loss": epoch_loss / max(len(train_loader), 1)}
        val_metrics = _evaluate(model, val_loader, device, use_amp)
        print(
            f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_soft_dice={val_metrics['dice']:.4f} "
            f"val_hard_dice={val_metrics['hard_dice']:.4f} "
            f"val_fg_hard_dice={val_metrics['fg_hard_dice']:.4f}"
        )

        if val_metrics["fg_hard_dice"] > best_fg_dice:
            best_fg_dice = val_metrics["fg_hard_dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": features,
                    "patch_size": (args.patch_size, args.patch_size, args.patch_size),
                    "val_fg_hard_dice": best_fg_dice,
                    "norm": args.norm,
                },
                args.output,
            )
            print(f"  -> Saved new best checkpoint to {args.output}")

    print(f"Training complete. Best validation foreground hard Dice: {best_fg_dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
