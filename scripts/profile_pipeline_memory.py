"""Profile peak memory usage of the resolved-regime pipeline on synthetic phantoms."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.io import save_tiff_stack
from fiber_tracer.pipeline import FiberAnalysisPipeline
from fiber_tracer.validation.phantoms import generate_fiber_phantom


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports kilobytes.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage.ru_maxrss / divisor


def profile(shape: tuple, n_fibers: int = 5, fiber_diameter_um: float = 6.0):
    phantom = generate_fiber_phantom(
        shape=shape,
        n_fibers=n_fibers,
        fiber_diameter_um=fiber_diameter_um,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        seed=42,
    )
    tmp_dir = Path("profile_tmp")
    tmp_dir.mkdir(exist_ok=True)
    data_path = tmp_dir / f"phantom_{shape[0]}_{shape[1]}_{shape[2]}.tif"
    out_dir = tmp_dir / f"out_{shape[0]}_{shape[1]}_{shape[2]}"
    save_tiff_stack(data_path, phantom.volume)

    config = Config(
        data_path=str(data_path),
        output_dir=str(out_dir),
        voxel_spacing_um=VoxelSpacing(1.0, 1.0, 1.0),
        fiber_diameter_um=fiber_diameter_um,
        regime="resolved",
    )

    start = time.time()
    pipeline = FiberAnalysisPipeline(config)
    summary = pipeline.run()
    elapsed = time.time() - start

    return {
        "shape": shape,
        "n_voxels": int(np.prod(shape)),
        "peak_memory_mb": peak_memory_mb(),
        "elapsed_s": elapsed,
        "n_labels": summary.get("n_labels", 0),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Profile pipeline memory usage")
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=int,
        default=[64, 128],
        help="Cube edge lengths to test",
    )
    parser.add_argument("--n-fibers", type=int, default=5)
    parser.add_argument("--fiber-diameter", type=float, default=6.0)
    args = parser.parse_args(argv)

    print("shape,n_voxels,peak_memory_mb,elapsed_s,n_labels")
    for edge in args.shapes:
        shape = (edge, edge, edge)
        result = profile(shape, args.n_fibers, args.fiber_diameter)
        print(
            f"{result['shape']},{result['n_voxels']},"
            f"{result['peak_memory_mb']:.2f},{result['elapsed_s']:.2f},"
            f"{result['n_labels']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
