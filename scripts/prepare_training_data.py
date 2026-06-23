"""Convert raw XCT stacks into training patches and a dataset registry."""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
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


def _otsu_threshold_from_sample(values: np.ndarray) -> float:
    """Compute an Otsu threshold from a 1-D array of sample intensities."""
    from skimage import filters

    return float(filters.threshold_otsu(values))


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
    min_foreground_ratio: float = 0.0005,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract foreground-biased 3D patches.

    Patches are rejected and re-sampled if their foreground ratio is below
    *min_foreground_ratio*.  Thin fibers occupy a tiny volume fraction, so the
    default threshold is low (0.05%) to keep fiber-containing patches while
    still discarding purely background crops.
    """
    rng = random.Random(seed)
    volume, mask = _maybe_pad(volume, mask, patch_size)
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    vol_patches: list[np.ndarray] = []
    msk_patches: list[np.ndarray] = []
    foreground_coords = np.argwhere(mask > 0)
    if not foreground_coords.size:
        # No foreground at all; return random background patches.
        for _ in range(n_patches):
            z = rng.randint(0, d - pd)
            y = rng.randint(0, h - ph)
            x = rng.randint(0, w - pw)
            vol_patches.append(volume[z : z + pd, y : y + ph, x : x + pw])
            msk_patches.append(mask[z : z + pd, y : y + ph, x : x + pw])
        return vol_patches, msk_patches

    # Phase 1: try to get n_patches with the desired foreground ratio.
    attempts = 0
    max_attempts = n_patches * 200
    while len(vol_patches) < n_patches and attempts < max_attempts:
        attempts += 1
        if rng.random() > 0.1:
            z, y, x = foreground_coords[rng.randint(0, len(foreground_coords) - 1)]
            z = min(max(z - pd // 2, 0), d - pd)
            y = min(max(y - ph // 2, 0), h - ph)
            x = min(max(x - pw // 2, 0), w - pw)
        else:
            z = rng.randint(0, d - pd)
            y = rng.randint(0, h - ph)
            x = rng.randint(0, w - pw)
        patch_mask = mask[z : z + pd, y : y + ph, x : x + pw]
        if patch_mask.mean() < min_foreground_ratio:
            continue
        vol_patches.append(volume[z : z + pd, y : y + ph, x : x + pw])
        msk_patches.append(patch_mask)

    # Phase 2: relax constraint and keep trying foreground-biased patches.
    relaxed_ratio = min_foreground_ratio / 5.0
    attempts = 0
    while len(vol_patches) < n_patches and attempts < max_attempts:
        attempts += 1
        z, y, x = foreground_coords[rng.randint(0, len(foreground_coords) - 1)]
        z = min(max(z - pd // 2, 0), d - pd)
        y = min(max(y - ph // 2, 0), h - ph)
        x = min(max(x - pw // 2, 0), w - pw)
        patch_mask = mask[z : z + pd, y : y + ph, x : x + pw]
        if patch_mask.mean() < relaxed_ratio:
            continue
        vol_patches.append(volume[z : z + pd, y : y + ph, x : x + pw])
        msk_patches.append(patch_mask)

    # Phase 3: last resort, random patches.
    while len(vol_patches) < n_patches:
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
    modes = [
        "random",
        "aligned",
        "in_plane",
        "orthogonal",
        "woven",
        "twill",
    ]
    mode = modes[idx % len(modes)]
    phantom = generate_fiber_phantom(
        shape=(128, 128, 128),
        n_fibers=rng.randint(3, 16),
        fiber_diameter_um=rng.uniform(1.5, 10.0),
        fiber_length_um=rng.uniform(20.0, 180.0),
        voxel_spacing_um=(1.0, 1.0, 1.0),
        noise_std=rng.uniform(0.01, 0.08),
        orientation_mode=mode,
        broken_fraction=rng.uniform(0.0, 0.25),
        n_broken_pieces=rng.randint(2, 4),
        porosity=rng.uniform(0.0, 0.005),
        seed=rng.randint(0, 2**31 - 1),
    )
    volume = _normalize(phantom.volume.astype(np.float32))
    mask = (phantom.labels > 0).astype(np.float32)
    vol_patches, msk_patches = _extract_patches(
        volume, mask, n_patches=1, patch_size=patch_size, seed=idx
    )
    return vol_patches[0], msk_patches[0]


def _is_h5(path: Path) -> bool:
    return path.suffix.lower() in (".h5", ".hdf5")


def _h5_data_path(sub_key: Optional[str]) -> str:
    return f"{sub_key}/data" if sub_key else "data"


def _h5_label_path(sub_key: Optional[str]) -> str:
    return f"{sub_key}/ground_truth" if sub_key else "ground_truth"


def _volume_shape(path: Path, sub_key: Optional[str] = None) -> tuple[int, int, int]:
    """Return (D, H, W) without loading the full volume."""
    if _is_h5(path):
        import h5py

        with h5py.File(path, "r") as f:
            ds = f[_h5_data_path(sub_key)]
            return tuple(int(s) for s in ds.shape)
    if path.is_dir():
        files = sorted(path.glob("*.tif*"))
        if not files:
            raise FileNotFoundError(f"No TIFF files in {path}")
        sample = tifffile.imread(files[0])
        return (len(files), *sample.shape)
    # Single TIFF file.
    return load_tiff_stack(path).shape


def _sample_intensities(
    path: Path,
    sub_key: Optional[str] = None,
    max_voxels: int = 10_000_000,
) -> np.ndarray:
    """Sample voxel intensities for Otsu threshold estimation."""
    shape = _volume_shape(path, sub_key)
    total = int(np.prod(shape))
    if total <= max_voxels:
        # Small enough to load entirely.
        roi = _load_roi(path, sub_key, (0, shape[0]), (0, shape[1]), (0, shape[2]))
        return roi.astype(np.float32).ravel()

    # Sample a regular grid that covers the volume.
    d, h, w = shape
    # Determine stride so that we get roughly max_voxels samples.
    stride = int(np.ceil((total / max_voxels) ** (1 / 3)))
    z = np.arange(0, d, stride)
    y = np.arange(0, h, stride)
    x = np.arange(0, w, stride)
    roi = _load_roi(path, sub_key, (int(z[0]), int(z[-1]) + 1), (0, h), (0, w))
    sampled = roi[z - z[0], :, :][:, y, :][:, :, x]
    return sampled.astype(np.float32).ravel()


def _load_roi(
    path: Path,
    sub_key: Optional[str],
    z_range: tuple[int, int],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
) -> np.ndarray:
    """Load a 3D ROI [z0:z1, y0:y1, x0:x1] efficiently."""
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    if _is_h5(path):
        import h5py

        with h5py.File(path, "r") as f:
            ds = f[_h5_data_path(sub_key)]
            return np.asarray(ds[z0:z1, y0:y1, x0:x1])
    if path.is_dir():
        files = sorted(path.glob("*.tif*"))
        selected = files[z0:z1]
        slices = [tifffile.imread(f) for f in selected]
        stack = np.stack(slices, axis=0)
        return stack[:, y0:y1, x0:x1]
    # Single TIFF file: load whole thing and crop.
    return load_tiff_stack(path)[z0:z1, y0:y1, x0:x1]


def _load_label_roi(
    path: Path,
    sub_key: Optional[str],
    z_range: tuple[int, int],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
) -> np.ndarray:
    """Load a binary foreground ROI from a ground-truth HDF5 dataset."""
    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[_h5_label_path(sub_key)]
        return (np.asarray(ds[z0:z1, y0:y1, x0:x1]) > 0).astype(np.float32)


def _has_ground_truth(path: Path, sub_key: Optional[str] = None) -> bool:
    """Check whether an HDF5 file contains a ground-truth label dataset."""
    if not _is_h5(path):
        return False
    import h5py

    with h5py.File(path, "r") as f:
        return _h5_label_path(sub_key) in f


def _extract_patches_streaming(
    path: Path,
    sub_key: Optional[str],
    n_patches: int,
    patch_size: tuple[int, int, int],
    seed: int,
    threshold: Optional[float] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    """Extract patches from a potentially large volume without loading it whole.

    Returns ``(volume_patches, mask_patches, label_source)``.
    """
    rng = random.Random(seed)
    shape = _volume_shape(path, sub_key)
    d, h, w = shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        # Load whole small volume and use the in-memory extractor.
        volume = _normalize(_load_roi(path, sub_key, (0, d), (0, h), (0, w)).astype(np.float32))
        if _has_ground_truth(path, sub_key):
            mask = _load_label_roi(path, sub_key, (0, d), (0, h), (0, w))
            label_source = "ground_truth"
        else:
            mask = (
                (volume > threshold).astype(np.float32)
                if threshold is not None
                else _pseudo_label(volume).astype(np.float32)
            )
            label_source = "otsu"
        vol_patches, msk_patches = _extract_patches(
            volume, mask, n_patches=n_patches, patch_size=patch_size, seed=seed
        )
        return vol_patches, msk_patches, label_source

    # Determine patch origins.
    origins: list[tuple[int, int, int]] = []
    for _ in range(n_patches):
        z = rng.randint(0, d - pd)
        y = rng.randint(0, h - ph)
        x = rng.randint(0, w - pw)
        origins.append((z, y, x))

    has_gt = _has_ground_truth(path, sub_key)
    label_source = "ground_truth" if has_gt else "otsu"
    vol_patches: list[np.ndarray] = []
    msk_patches: list[np.ndarray] = []
    for z, y, x in origins:
        volume_roi = _load_roi(path, sub_key, (z, z + pd), (y, y + ph), (x, x + pw))
        volume_roi = _normalize(volume_roi.astype(np.float32))
        if has_gt:
            mask_roi = _load_label_roi(path, sub_key, (z, z + pd), (y, y + ph), (x, x + pw))
        else:
            mask_roi = (
                (volume_roi > threshold).astype(np.float32)
                if threshold is not None
                else _pseudo_label(volume_roi).astype(np.float32)
            )
        vol_patches.append(volume_roi)
        msk_patches.append(mask_roi)
    return vol_patches, msk_patches, label_source


def _find_volumes(source_dir: Path) -> list[tuple[Path, Optional[str]]]:
    """Return list of loadable volume references under *source_dir*.

    TIFF *directories* containing many slices are treated as a single 3D
    volume. Standalone multi-page TIFF files are also accepted. HDF5 files
    may yield multiple references if they contain several volumes.
    """
    candidates: list[tuple[Path, Optional[str]]] = []

    # Discover TIFF directories first.
    tiff_dirs: set[Path] = set()
    for ext in ("*.tif", "*.tiff"):
        for path in source_dir.rglob(ext):
            tiff_dirs.add(path.parent)

    # Treat directories with many slices as volume references.
    for tiff_dir in sorted(tiff_dirs):
        n_tiffs = len(list(tiff_dir.glob("*.tif*")))
        if n_tiffs >= 10:
            candidates.append((tiff_dir, None))

    # Add standalone TIFF files that are not inside an already-registered
    # TIFF directory.
    registered_dirs = {ref[0] for ref in candidates}
    for ext in ("*.tif", "*.tiff"):
        for path in source_dir.rglob(ext):
            if path.parent not in registered_dirs:
                candidates.append((path, None))

    for ext in ("*.h5", "*.hdf5"):
        for path in source_dir.rglob(ext):
            try:
                import h5py

                with h5py.File(path, "r") as f:
                    for key in f.keys():
                        obj = f[key]
                        if isinstance(obj, h5py.Group) and "data" in obj:
                            candidates.append((path, key))
                        elif isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                            candidates.append((path, None))
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
    return candidates


def _process_real_volume(
    volume_path: Path,
    output_dir: Path,
    sub_key: Optional[str] = None,
    n_patches: int = PATCHES_PER_VOLUME,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> Optional[dict]:
    try:
        shape = _volume_shape(volume_path, sub_key)
    except Exception as exc:
        print(f"Skipping {volume_path}: {exc}")
        return None

    if len(shape) != 3:
        print(f"Skipping {volume_path}: expected 3D, got {len(shape)}D")
        return None

    seed = hash(f"{volume_path.name}_{sub_key}") % 2**31
    threshold: Optional[float] = None
    if not _has_ground_truth(volume_path, sub_key):
        # Estimate Otsu threshold from a sample so we can stream patches.
        sample = _sample_intensities(volume_path, sub_key)
        threshold = _otsu_threshold_from_sample(sample)

    try:
        vol_patches, msk_patches, label_source = _extract_patches_streaming(
            volume_path,
            sub_key,
            n_patches=n_patches,
            patch_size=patch_size,
            seed=seed,
            threshold=threshold,
        )
    except Exception as exc:
        print(f"Skipping {volume_path}: {exc}")
        return None

    shard_name = f"{volume_path.parent.name}_{volume_path.stem}"
    if sub_key:
        shard_name += f"_{sub_key.replace('/', '_')}"
    shard_dir = output_dir / shard_name
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, (vp, mp) in enumerate(zip(vol_patches, msk_patches)):
        np.savez_compressed(shard_dir / f"patch_{i:04d}.npz", volume=vp, mask=mp)

    return {
        "name": f"{volume_path.parent.name}/{volume_path.stem}"
        + (f"/{sub_key}" if sub_key else ""),
        "type": "real",
        "source": str(volume_path),
        "label_source": label_source,
        "n_patches": len(vol_patches),
        "patch_dir": str(shard_dir.relative_to(output_dir.parent)),
    }


def _extract_zip_archives(source_dir: Path) -> None:
    for archive in source_dir.rglob("*.zip"):
        extract_dir = archive.with_suffix("")
        if extract_dir.exists():
            continue
        print(f"Extracting {archive} ...")
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            print(f"  Warning: skipping incomplete/corrupt archive {archive}: {exc}")


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
            for volume_path, sub_key in tqdm(
                _find_volumes(source_dir), desc=f"processing {source_dir.name}"
            ):
                entry = _process_real_volume(
                    volume_path,
                    real_output,
                    sub_key=sub_key,
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
