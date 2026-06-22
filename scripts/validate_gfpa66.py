"""Validate RAFA against the GF-PA66 3D XCT dataset.

License: CC BY-SA 4.0
DOI: 10.5281/zenodo.4587827
Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229
"""

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline

DEFAULT_DATASET_NAMES: Sequence[str] = (
    "pa66_volumes.h5",
    "GF-PA66_3D_XCT.h5",
)


def find_dataset_path(hint: Optional[str] = None) -> Path:
    """Locate the GF-PA66 HDF5 dataset.

    If ``hint`` is provided, return it as a Path. Otherwise search the current
    working directory for the default dataset filenames.
    """
    if hint is not None:
        return Path(hint)

    for name in DEFAULT_DATASET_NAMES:
        candidate = Path(name)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect GF-PA66 dataset. "
        f"Looked for {list(DEFAULT_DATASET_NAMES)} in the current directory. "
        "Use --data to specify the path to pa66_volumes.h5."
    )


def load_hdf5_volume(path: str, dataset: Optional[str] = None):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install h5py to validate GF-PA66: pip install h5py") from exc

    with h5py.File(path, "r") as f:
        if dataset:
            return f[dataset][()]
        # Try common dataset names
        for candidate in ["data", "image", "volume", "XCT"]:
            if candidate in f:
                return f[candidate][()]
        # If only one dataset exists, use it
        keys = list(f.keys())
        if len(keys) == 1:
            return f[keys[0]][()]
        raise ValueError(f"Could not auto-detect HDF5 dataset. Keys: {keys}. Use --dataset.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate RAFA on GF-PA66")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to GF-PA66 HDF5 file (default: auto-detect pa66_volumes.h5)",
    )
    parser.add_argument("--dataset", default=None, help="HDF5 dataset name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--fiber-diameter", type=float, default=10.0)
    parser.add_argument(
        "--regime", choices=["auto", "resolved", "marginal", "subvoxel"], default="auto"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    data_path = find_dataset_path(args.data)
    if not data_path.is_file():
        print(f"ERROR: Dataset not found at {data_path}")
        print(
            "Download it with: python scripts/download_gfpa66.py "
            "--file pa66_volumes.h5 --accept-license"
        )
        return 1

    volume = load_hdf5_volume(str(data_path), args.dataset)

    # The RAFA pipeline loads data via load_tiff_stack. Convert the HDF5 volume
    # to a temporary multi-page TIFF stack and point Config.data_path at it.
    with tempfile.TemporaryDirectory(prefix="gfpa66_") as tmpdir:
        temp_tiff = Path(tmpdir) / "volume.tif"
        save_tiff_stack(temp_tiff, volume)

        config = Config(
            data_path=str(temp_tiff),
            output_dir=args.output,
            voxel_spacing_um=VoxelSpacing(*args.voxel_spacing),
            fiber_diameter_um=args.fiber_diameter,
            regime=args.regime,
        )
        config.validate()
        pipeline = FiberAnalysisPipeline(config)
        summary = pipeline.run()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
