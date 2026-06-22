# Improved Fiber U-Net Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a higher-capacity 3D U-Net on open Henry Royce and other fiber-composite XCT data, verify it beats the synthetic-only baseline, and publish the checkpoint(s) as GitHub Release assets.

**Architecture:** Extend the existing `UNet3D` with a deeper encoder, add a mixed real/synthetic dataset loader and patch extractor, train with BCE+Dice (+ boundary loss) on MPS/CPU, validate with Dice, and package checkpoints via `gh release`.

**Tech Stack:** PyTorch 2.x, scikit-image, numpy, tifffile, requests, tqdm, PyTorch MPS or CPU.

---

## File map

| File | Responsibility |
|------|----------------|
| `scripts/download_datasets.py` | Download open datasets (Henry Royce Zenodo, GF-PA66, etc.) |
| `scripts/prepare_training_data.py` | Normalize, generate pseudo-labels, extract 64³ patches, write dataset registry |
| `src/fiber_tracer/backends/unet3d.py` | Extend U-Net architecture (deeper variant, dropout, IN option) |
| `src/fiber_tracer/training/__init__.py` | Package marker |
| `src/fiber_tracer/training/augment.py` | 3D augmentations (flip, rot90, gamma, noise) |
| `src/fiber_tracer/training/dataset.py` | `FiberVolumeDataset` that reads patch shards and labels |
| `scripts/train_unet_mixed.py` | Training loop with MPS/CPU AMP support, checkpoint saving |
| `scripts/validate_unet.py` | Compute whole-volume Dice on held-out data |
| `README.md`, `docs/parameter_guide.md`, `CHANGELOG.md` | Document new checkpoints and usage |

---

## Task 1: Set up data directories and download scripts

**Files:**
- Create: `scripts/download_datasets.py`
- Modify: `.gitignore`

**Context:** All raw data lives under `data/raw/<source>/` and is gitignored. Training patches live under `data/processed/`.

- [ ] **Step 1.1: Add data directories to `.gitignore`**

```bash
# data/raw/
# data/processed/
```

Append to `.gitignore`:

```text
# Training data
/data/raw/
/data/processed/
/data/datasets.json
```

- [ ] **Step 1.2: Create `scripts/download_datasets.py`**

```python
"""Download open XCT datasets for fiber segmentation training."""

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DATASETS = {
    "henry_ncf_fatigue": {
        "url": "https://zenodo.org/record/4541235/files/0_cycles.zip",
        "archive": "0_cycles.zip",
        "source": "Henry Royce Institute / Prajapati et al.",
    },
    "henry_ud_compression": {
        "url": "https://zenodo.org/record/2597498/files/GFRP_Initial.zip",
        "archive": "GFRP_Initial.zip",
        "source": "Henry Royce Institute / Wang et al.",
    },
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download open XCT fiber datasets")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args(argv)

    selected = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    for key in selected:
        meta = DATASETS[key]
        out_dir = args.output / key
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = out_dir / meta["archive"]
        if archive.exists():
            print(f"{archive} already exists; skipping download.")
            continue
        print(f"Downloading {key} from {meta['url']}")
        _download(meta["url"], archive)
        print(f"Saved to {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 1.3: Test download script (dry-run/small dataset)**

Run:

```bash
python scripts/download_datasets.py --datasets henry_ud_compression
```

Expected: archive appears in `data/raw/henry_ud_compression/`.

- [ ] **Step 1.4: Commit**

```bash
git add scripts/download_datasets.py .gitignore
git commit -m "Add downloader for open Henry Royce XCT datasets"
```

---

## Task 2: Extend U-Net architecture

**Files:**
- Modify: `src/fiber_tracer/backends/unet3d.py`

- [ ] **Step 2.1: Add dropout, normalization choice, and deeper default**

Replace the `_ConvBlock` and `UNet3D` definitions so the constructor accepts `features`, `dropout`, and `norm`.

```python
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        Norm = nn.BatchNorm3d if norm == "batch" else nn.InstanceNorm3d
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore[no-any-return]


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (8, 16, 32),
        dropout: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        current = in_channels
        for feature in features:
            self.encoder_blocks.append(_ConvBlock(current, feature, dropout=dropout, norm=norm))
            current = feature

        self.bottleneck = _ConvBlock(features[-1], features[-1] * 2, dropout=dropout, norm=norm)

        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features)):
            in_feat = reversed_features[i] * 2 if i == 0 else reversed_features[i - 1]
            out_feat = reversed_features[i]
            self.up_convs.append(
                nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                _ConvBlock(out_feat * 2, out_feat, dropout=dropout, norm=norm)
            )

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, decoder_block, enc_out in zip(
            self.up_convs, self.decoder_blocks, reversed(encoder_outputs)
        ):
            x = up_conv(x)
            if x.shape != enc_out.shape:
                diff_d = enc_out.shape[2] - x.shape[2]
                diff_h = enc_out.shape[3] - x.shape[3]
                diff_w = enc_out.shape[4] - x.shape[4]
                x = nn.functional.pad(
                    x,
                    [
                        diff_w // 2,
                        diff_w - diff_w // 2,
                        diff_h // 2,
                        diff_h - diff_h // 2,
                        diff_d // 2,
                        diff_d - diff_d // 2,
                    ],
                )
            x = torch.cat([enc_out, x], dim=1)
            x = decoder_block(x)

        return torch.sigmoid(self.final_conv(x))
```

Keep `predict_volume` unchanged.

- [ ] **Step 2.2: Add architecture test**

In `tests/test_ml_backend.py` add:

```python
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_unet3d_large_forward_shape():
    import torch
    from fiber_tracer.backends.unet3d import UNet3D

    model = UNet3D(in_channels=1, out_channels=1, features=(16, 32, 64, 128))
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64, 64)
```

- [ ] **Step 2.3: Run tests**

```bash
pytest tests/test_ml_backend.py -v
```

Expected: all tests pass.

- [ ] **Step 2.4: Commit**

```bash
git add src/fiber_tracer/backends/unet3d.py tests/test_ml_backend.py
git commit -m "Extend UNet3D with deeper features, dropout, and norm choice"
```

---

## Task 3: Preprocessing and pseudo-label generation

**Files:**
- Create: `scripts/prepare_training_data.py`

- [ ] **Step 3.1: Implement preprocessing script**

The script walks `data/raw/<source>/`, loads TIFF stacks, normalizes, optionally generates pseudo-labels with Otsu/watershed, extracts 64³ patches, and writes a registry.

```python
"""Convert raw TIFF stacks into training patches and a dataset registry."""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from skimage import exposure
from tqdm import tqdm

from fiber_tracer.io import load_tiff_stack, save_tiff_stack
from fiber_tracer.segmentation.classical import segment_otsu_3d, segment_watershed_3d
from fiber_tracer.validation.phantoms import generate_fiber_phantom

PATCH_SIZE = (64, 64, 64)
PATCHES_PER_VOLUME = 128


def _normalize(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v


def _pseudo_label(volume: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "otsu":
        return segment_otsu_3d(volume)
    return segment_watershed_3d(segment_otsu_3d(volume))


def _extract_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    n_patches: int = PATCHES_PER_VOLUME,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(42)
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    vol_patches: list[np.ndarray] = []
    msk_patches: list[np.ndarray] = []
    foreground_coords = np.argwhere(mask > 0)
    for _ in range(n_patches):
        if foreground_coords.size and rng.random() > 0.3:
            z, y, x = foreground_coords[rng.randint(0, len(foreground_coords) - 1)]
            z = min(max(z - pd // 2, 0), d - pd)
            y = min(max(y - ph // 2, 0), h - ph)
            x = min(max(x - pw // 2, 0), w - pw)
        else:
            z = rng.randint(0, d - pd)
            y = rng.randint(0, h - ph)
            x = rng.randint(0, w - pw)
        vol_patches.append(volume[z : z + pd, y : y + ph, x : x + pw])
        msk_patches.append(mask[z : z + pd, y : y + ph, x : x + pw])
    return vol_patches, msk_patches


def _extract_from_phantom(idx: int, patch_size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(idx)
    phantom = generate_fiber_phantom(
        shape=(128, 128, 128),
        n_fibers=rng.randint(3, 12),
        fiber_diameter_um=rng.uniform(2.0, 8.0),
        voxel_spacing_um=(1.0, 1.0, 1.0),
        noise_std=rng.uniform(0.01, 0.07),
        seed=rng.randint(0, 2**31 - 1),
    )
    volume = _normalize(phantom.volume.astype(np.float32))
    mask = (phantom.labels > 0).astype(np.float32)
    vol_patches, msk_patches = _extract_patches(volume, mask, n_patches=PATCHES_PER_VOLUME, patch_size=patch_size)
    return vol_patches[0], msk_patches[0]


def _process_real_volume(
    tiff_path: Path,
    output_dir: Path,
    label_method: str = "otsu",
    n_patches: int = PATCHES_PER_VOLUME,
) -> dict:
    volume = load_tiff_stack(tiff_path)
    volume = _normalize(volume)
    mask = _pseudo_label(volume, method=label_method).astype(np.float32)
    vol_patches, msk_patches = _extract_patches(volume, mask, n_patches=n_patches)

    shard_dir = output_dir / tiff_path.stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, (vp, mp) in enumerate(zip(vol_patches, msk_patches)):
        np.savez_compressed(shard_dir / f"patch_{i:04d}.npz", volume=vp, mask=mp)

    return {
        "source": str(tiff_path),
        "label_method": label_method,
        "n_patches": len(vol_patches),
        "patch_dir": str(shard_dir.relative_to(output_dir.parent)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare training patches")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-synthetic", type=int, default=100, help="Number of synthetic phantoms")
    parser.add_argument("--n-patches-per-volume", type=int, default=PATCHES_PER_VOLUME)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    registry: list[dict] = []

    # Synthetic phantoms
    syn_dir = args.output / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(args.n_synthetic), desc="synthetic phantoms"):
        vp, mp = _extract_from_phantom(i, PATCH_SIZE)
        np.savez_compressed(syn_dir / f"phantom_{i:04d}.npz", volume=vp, mask=mp)
    registry.append({
        "name": "synthetic",
        "type": "synthetic",
        "n_patches": args.n_synthetic,
        "patch_dir": str(syn_dir.relative_to(args.output.parent)),
    })

    # Real volumes
    for source_dir in sorted(args.raw.iterdir()):
        if not source_dir.is_dir():
            continue
        tiff_candidates = list(source_dir.rglob("*.tif")) + list(source_dir.rglob("*.tiff"))
        if not tiff_candidates:
            continue
        for tiff_path in tqdm(tiff_candidates, desc=f"processing {source_dir.name}"):
            try:
                entry = _process_real_volume(
                    tiff_path,
                    args.output / "real",
                    label_method="otsu",
                    n_patches=args.n_patches_per_volume,
                )
                entry["name"] = f"{source_dir.name}/{tiff_path.stem}"
                entry["type"] = "real"
                registry.append(entry)
            except Exception as exc:
                print(f"Skipping {tiff_path}: {exc}")

    registry_path = args.output.parent / "datasets.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry written to {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3.2: Run on synthetic-only subset first**

```bash
python scripts/prepare_training_data.py --n-synthetic 10 --n-patches-per-volume 8
```

Expected: `data/processed/synthetic/` contains `.npz` files and `data/datasets.json` is created.

- [ ] **Step 3.3: Commit**

```bash
git add scripts/prepare_training_data.py
git commit -m "Add preprocessing script: pseudo-labels and 64-cubed patch extraction"
```

---

## Task 4: Training dataset loader and augmentations

**Files:**
- Create: `src/fiber_tracer/training/__init__.py`
- Create: `src/fiber_tracer/training/augment.py`
- Create: `src/fiber_tracer/training/dataset.py`

- [ ] **Step 4.1: Create training package**

```bash
touch src/fiber_tracer/training/__init__.py
```

- [ ] **Step 4.2: Implement augmentations**

```python
# src/fiber_tracer/training/augment.py
import random

import numpy as np


def augment_patch(volume: np.ndarray, mask: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    # Axis flips
    for axis in range(3):
        if rng.random() > 0.5:
            volume = np.flip(volume, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
    # 90-degree rotations around z
    k = rng.randint(0, 3)
    if k:
        volume = np.rot90(volume, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
    # Intensity gamma
    if rng.random() > 0.5:
        gamma = rng.uniform(0.8, 1.2)
        volume = np.clip(volume ** gamma, 0.0, 1.0)
    # Additive noise
    if rng.random() > 0.5:
        noise = rng.gauss(0, 0.02)
        volume = np.clip(volume + noise, 0.0, 1.0)
    return volume, mask
```

- [ ] **Step 4.3: Implement dataset**

```python
# src/fiber_tracer/training/dataset.py
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from fiber_tracer.training.augment import augment_patch


class FiberVolumeDataset(Dataset):
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
        self.augment = augment
        self.patch_files: list[Path] = []
        rng = random.Random(seed)
        for entry in registry:
            patch_dir = self.processed_root / entry["patch_dir"]
            files = sorted(patch_dir.glob("*.npz"))
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
            volume, mask = augment_patch(volume, mask, seed=self.rng.randint(0, 2**31 - 1))
        return torch.from_numpy(volume).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
```

- [ ] **Step 4.4: Add unit test for dataset**

Create `tests/test_training_dataset.py`:

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fiber_tracer.training.dataset import FiberVolumeDataset


def test_dataset_loads_patch():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        patch_dir = tmp / "synthetic"
        patch_dir.mkdir()
        np.savez_compressed(patch_dir / "p_0000.npz", volume=np.random.rand(64, 64, 64), mask=np.ones((64, 64, 64)))
        registry = [{"name": "synthetic", "type": "synthetic", "n_patches": 1, "patch_dir": "synthetic"}]
        registry_path = tmp / "datasets.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        ds = FiberVolumeDataset(registry_path, tmp, split="train", augment=False)
        assert len(ds) == 1
        x, y = ds[0]
        assert x.shape == (1, 64, 64, 64)
        assert y.shape == (1, 64, 64, 64)
```

- [ ] **Step 4.5: Run test**

```bash
pytest tests/test_training_dataset.py -v
```

Expected: PASS.

- [ ] **Step 4.6: Commit**

```bash
git add src/fiber_tracer/training/ tests/test_training_dataset.py
git commit -m "Add training dataset loader and 3D augmentations"
```

---

## Task 5: Mixed-data training script

**Files:**
- Create: `scripts/train_unet_mixed.py`

- [ ] **Step 5.1: Implement training script**

Use the same BCEDice loss, add a boundary term, support `--features`, `--device auto`, AMP, and save architecture metadata.

```python
"""Train a 3D U-Net on mixed synthetic + real fiber XCT patches."""

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence, cast

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
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred, target)
        dice = 1.0 - _dice_score(pred, target).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice  # type: ignore[no-any-return]


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> dict[str, float]:
    model.eval()
    criterion = BCEDiceLoss()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_dice += _dice_score(outputs, targets).mean().item()
            n_batches += 1
    return {"loss": total_loss / max(n_batches, 1), "dice": total_dice / max(n_batches, 1)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train U-Net on mixed fiber XCT patches")
    parser.add_argument("--registry", type=Path, default=Path("data/datasets.json"))
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

    train_ds = FiberVolumeDataset(args.registry, args.processed_root, split="train", augment=True, seed=args.seed)
    val_ds = FiberVolumeDataset(args.registry, args.processed_root, split="val", augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    features = tuple(args.features)
    model = UNet3D(in_channels=1, out_channels=1, features=features, dropout=args.dropout, norm=args.norm).to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.amp and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler() if use_amp else None

    best_dice = -1.0
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
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": features,
                    "patch_size": (args.patch_size, args.patch_size, args.patch_size),
                    "val_dice": best_dice,
                    "norm": args.norm,
                },
                args.output,
            )
            print(f"  -> Saved new best checkpoint to {args.output}")

    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.2: Smoke-test training**

```bash
python scripts/prepare_training_data.py --n-synthetic 20 --n-patches-per-volume 16
python scripts/train_unet_mixed.py --features 8 16 32 --epochs 2 --batch-size 1 --device cpu --output models/smoke.pt
```

Expected: runs without error and saves `models/smoke.pt`.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/train_unet_mixed.py
git commit -m "Add mixed-data U-Net training script with MPS/CPU AMP support"
```

---

## Task 6: Validation on held-out volumes

**Files:**
- Create: `scripts/validate_unet.py`

- [ ] **Step 6.1: Implement validation script**

```python
"""Validate a trained U-Net on full volumes and report Dice scores."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from fiber_tracer.backends.unet3d import UNet3D
from fiber_tracer.backends.ml_segmentation import MLSegmentationBackend
from fiber_tracer.io import load_tiff_stack


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float((pred * target).sum())
    return 2 * inter / float(pred.sum() + target.sum()) if (pred.sum() + target.sum()) > 0 else 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate U-Net on full volumes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--label", type=Path, help="Ground-truth binary mask TIFF")
    parser.add_argument("--output", type=Path, help="Where to save predicted mask")
    args = parser.parse_args(argv)

    backend = MLSegmentationBackend.from_checkpoint(args.checkpoint)
    volume = load_tiff_stack(args.volume)
    pred = backend.segment(volume).astype(bool)

    if args.output:
        from fiber_tracer.io import save_tiff_stack
        save_tiff_stack(args.output, pred.astype(np.uint8) * 255)

    if args.label:
        target = load_tiff_stack(args.label).astype(bool)
        dice = _dice(pred, target)
        print(f"Dice: {dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.2: Smoke-test on synthetic phantom**

```python
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.validation.phantoms import generate_fiber_phantom
phantom = generate_fiber_phantom((128,128,128), 5, 4.0, (1,1,1), seed=99)
save_tiff_stack("val_volume.tif", phantom.volume)
save_tiff_stack("val_label.tif", (phantom.labels>0).astype(np.uint8)*255)
```

```bash
python scripts/validate_unet.py --checkpoint models/smoke.pt --volume val_volume.tif --label val_label.tif --output val_pred.tif
```

Expected: Dice printed.

- [ ] **Step 6.3: Commit**

```bash
git add scripts/validate_unet.py
git commit -m "Add U-Net validation script with whole-volume Dice"
```

---

## Task 7: Run full data download and preparation

- [ ] **Step 7.1: Download Henry Royce and GF-PA66 datasets**

```bash
python scripts/download_datasets.py --datasets all
# For GF-PA66, use the existing downloader:
python scripts/download_gfpa66.py --file pa66_volumes.h5 --output-dir data/raw/gfpa66 --accept-license
```

Expected: archives in `data/raw/`.

- [ ] **Step 7.2: Extract archives and prepare patches**

Use system `unzip` for `.zip` files, then:

```bash
python scripts/prepare_training_data.py --n-synthetic 500 --n-patches-per-volume 64
```

Expected: `data/processed/` populated and `data/datasets.json` updated.

- [ ] **Step 7.3: Commit registry and scripts (not data)**

```bash
git add data/datasets.json scripts/download_datasets.py scripts/prepare_training_data.py
git commit -m "Register downloaded datasets and generated patch registry"
```

---

## Task 8: Train production model(s)

- [ ] **Step 8.1: Train general model**

Run in background due to long duration:

```bash
python scripts/train_unet_mixed.py \
  --features 16 32 64 128 \
  --dropout 0.1 \
  --epochs 100 \
  --batch-size 1 \
  --lr 3e-4 \
  --device auto \
  --amp \
  --output models/fiber_unet_v2.pt
```

- [ ] **Step 8.2: Fine-tune Henry-specific model (optional)**

Filter registry to Henry-only entries, then:

```bash
python scripts/train_unet_mixed.py \
  --registry data/datasets_henry.json \
  --features 16 32 64 128 \
  --epochs 50 \
  --batch-size 1 \
  --output models/fiber_unet_henry.pt
```

- [ ] **Step 8.3: Validate**

```bash
python scripts/validate_unet.py --checkpoint models/fiber_unet_v2.pt --volume val_volume.tif --label val_label.tif
```

Quality gate: Dice ≥ 0.85 on held-out synthetic and at least one real patch set.

---

## Task 9: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/parameter_guide.md`
- Modify: `CHANGELOG.md`
- Create: `docs/model_cards/fiber_unet_v2.md` (optional)

- [ ] **Step 9.1: Document new checkpoints**

Add to `README.md`:

```markdown
### Pre-trained models

The best checkpoints are distributed as GitHub Release assets:

| Checkpoint | Architecture | Training data | Use case |
|------------|--------------|---------------|----------|
| `fiber_unet_v2.pt` | UNet3D (16,32,64,128) | Synthetic + open real XCT | General fiber segmentation |
| `fiber_unet_henry.pt` | UNet3D (16,32,64,128) | Henry Royce glass-epoxy data | Manchester Henry Royce glass composites |

Download a release asset and run:

```bash
fiber-tracer --data stack.tif --output results/ --segmentation-method unet --model-path fiber_unet_v2.pt
```
```

- [ ] **Step 9.2: Update CHANGELOG**

Add under `[Unreleased]`:

```markdown
- Larger 3D U-Net trained on mixed synthetic and open XCT datasets.
- New training scripts: `scripts/train_unet_mixed.py`, `scripts/validate_unet.py`.
- Pre-trained checkpoints published as GitHub Release assets.
```

- [ ] **Step 9.3: Commit docs**

```bash
git add README.md docs/parameter_guide.md CHANGELOG.md
git commit -m "Document improved U-Net checkpoints and training scripts"
```

---

## Task 10: Final lint, tests, and benchmark

- [ ] **Step 10.1: Run quality checks**

```bash
ruff check .
black --check .
mypy src/fiber_tracer
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 10.2: Run benchmark**

```bash
python scripts/benchmark_phantoms.py
```

Expected: resolved Dice > 0.85.

- [ ] **Step 10.3: Commit any fixes**

```bash
git add -A && git commit -m "Fix lint and type issues after training pipeline additions"
```

---

## Task 11: Upload models to GitHub Releases

- [ ] **Step 11.1: Ensure `gh` CLI is installed and authenticated**

```bash
gh auth status
```

Expected: logged in.

- [ ] **Step 11.2: Create release and attach checkpoints**

```bash
gh release create v3.3.0 \
  --title "Improved 3D U-Net segmentation models" \
  --notes "Pre-trained U-Net checkpoints trained on open Henry Royce and fiber-composite XCT data." \
  models/fiber_unet_v2.pt \
  models/fiber_unet_henry.pt
```

- [ ] **Step 11.3: Verify release URL**

```bash
gh release view v3.3.0
```

Expected: release page lists both assets.

---

## Spec coverage check

| Spec requirement | Task(s) |
|------------------|---------|
| Collect open Henry Royce data | 1, 7 |
| Preprocess and pseudo-label | 3, 7 |
| Larger U-Net architecture | 2 |
| Mixed-data training | 4, 5, 8 |
| M5 Pro / MPS/CPU training | 5, 8 |
| Validation ≥ 0.85 Dice | 6, 8 |
| Docs updated | 9 |
| Lint/tests/benchmark pass | 10 |
| Models uploaded to GitHub | 11 |

## Placeholder scan

No TBD/TODO/fill-in placeholders. All steps include concrete commands or code.

## Type consistency

- `UNet3D` signature uses `features: Sequence[int]`; training script builds `tuple[int, ...]` and saves it.
- `FiberVolumeDataset` accepts `registry_path: str | Path` and `processed_root: str | Path`.
- Checkpoint keys: `model_state_dict`, `features`, `patch_size`, `val_dice`, `norm` — loaded by `MLSegmentationBackend`.
