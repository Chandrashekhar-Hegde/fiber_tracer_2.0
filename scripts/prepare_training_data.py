"""Convert raw XCT stacks into training patches and a dataset registry."""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from fiber_tracer.io import load_tiff_stack
from fiber_tracer.segmentation.classical import segment_otsu_3d
from fiber_tracer.validation.phantoms import generate_fiber_phantom

PATCH_SIZE = (64, 64, 64)
PATCHES_PER_VOLUME = 64


def _normalize(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32)
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v


def _pseudo_label(volume: np.ndarray) -> np.ndarray:
    """Generate a binary foreground mask for unlabeled real data."""
    return segment_otsu_3d(volume)


def _maybe_pad(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pd, ph, pw = patch_size
    d, h, w = volume.shape
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d or pad_h or pad_w:
        volume = np.pad(
            volume,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
        )
        mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant")
    return volume, mask


def _extract_patches(
    volume: np.ndarray,
    mask: np.ndarray,
    n_patches: int = PATCHES_PER_VOLUME,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = random.Random(seed)
    volume, mask = _maybe_pad(volume, mask, patch_size)
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    vol_patches: list[np.ndarray] = []
    msk_patches: list[np.ndarray] = []
    foreground_coords = np.argwhere(mask > 0)
    for i in range(n_patches):
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


def _extract_from_phantom(
    idx: int,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
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
    vol_patches, msk_patches = _extract_patches(
        volume, mask, n_patches=1, patch_size=patch_size, seed=idx
    )
    return vol_patches[0], msk_patches[0]


def _load_tiff_volume(tiff_path: Path) -> np.ndarray:
    return load_tiff_stack(tiff_path)


def _load_h5_volume(h5_path: Path) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as f:
        # Try common dataset names.
        for name in ("data", "image", "volume", "XCT", "pa66"):
            if name in f:
                ds = f[name]
                if isinstance(ds, h5py.Dataset):
                    return np.asarray(ds)
        # Otherwise take the first float/integer dataset.
        for key in f.keys():
            ds = f[key]
            if isinstance(ds, h5py.Dataset) and ds.ndim == 3:
                return np.asarray(ds)
    raise ValueError(f"Could not find a 3D dataset in {h5_path}")


def _find_volumes(source_dir: Path) -> list[Path]:
    """Return list of loadable volume files under *source_dir*."""
    candidates: list[Path] = []
    for ext in ("*.tif", "*.tiff", "*.h5", "*.hdf5"):
        candidates.extend(source_dir.rglob(ext))
    return candidates


def _process_real_volume(
    volume_path: Path,
    output_dir: Path,
    n_patches: int = PATCHES_PER_VOLUME,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> Optional[dict]:
    try:
        if volume_path.suffix.lower() in (".h5", ".hdf5"):
            volume = _load_h5_volume(volume_path)
        else:
            volume = _load_tiff_volume(volume_path)
    except Exception as exc:
        print(f"Skipping {volume_path}: {exc}")
        return None

    if volume.ndim != 3:
        print(f"Skipping {volume_path}: expected 3D, got {volume.ndim}D")
        return None

    volume = _normalize(volume)
    mask = _pseudo_label(volume).astype(np.float32)
    seed = hash(volume_path.name) % 2**31
    vol_patches, msk_patches = _extract_patches(
        volume, mask, n_patches=n_patches, patch_size=patch_size, seed=seed
    )

    shard_dir = output_dir / f"{volume_path.parent.name}_{volume_path.stem}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, (vp, mp) in enumerate(zip(vol_patches, msk_patches)):
        np.savez_compressed(shard_dir / f"patch_{i:04d}.npz", volume=vp, mask=mp)

    return {
        "name": f"{volume_path.parent.name}/{volume_path.stem}",
        "type": "real",
        "source": str(volume_path),
        "n_patches": len(vol_patches),
        "patch_dir": str(shard_dir.relative_to(output_dir.parent)),
    }


def _extract_zip_archives(source_dir: Path) -> None:
    for archive in source_dir.rglob("*.zip"):
        extract_dir = archive.with_suffix("")
        if extract_dir.exists():
            continue
        print(f"Extracting {archive} ...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare training patches")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--n-patches-per-volume", type=int, default=PATCHES_PER_VOLUME)
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    registry: list[dict] = []

    # Synthetic phantoms.
    syn_dir = args.output / "synthetic"
    syn_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(args.n_synthetic), desc="synthetic phantoms"):
        vp, mp = _extract_from_phantom(i, PATCH_SIZE)
        np.savez_compressed(syn_dir / f"phantom_{i:04d}.npz", volume=vp, mask=mp)
    registry.append(
        {
            "name": "synthetic",
            "type": "synthetic",
            "n_patches": args.n_synthetic,
            "patch_dir": str(syn_dir.relative_to(args.output)),
        }
    )

    if args.skip_real:
        print("--skip-real set; only synthetic patches generated.")
    else:
        # Extract any zip archives found in raw data.
        for source_dir in sorted(args.raw.iterdir()):
            if source_dir.is_dir():
                _extract_zip_archives(source_dir)

        # Process real volumes.
        real_output = args.output / "real"
        real_output.mkdir(parents=True, exist_ok=True)
        for source_dir in sorted(args.raw.iterdir()):
            if not source_dir.is_dir():
                continue
            for volume_path in tqdm(
                _find_volumes(source_dir), desc=f"processing {source_dir.name}"
            ):
                entry = _process_real_volume(
                    volume_path,
                    real_output,
                    n_patches=args.n_patches_per_volume,
                )
                if entry is not None:
                    registry.append(entry)

    registry_path = args.output / "datasets.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Registry written to {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
