"""Command-line interface for fiber_tracer."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.pipeline import FiberAnalysisPipeline


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="RAFA fiber analysis")
    parser.add_argument("--data", required=True, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Path to YAML/JSON config")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, metavar=("Z", "Y", "X"))
    parser.add_argument("--fiber-diameter", type=float)
    parser.add_argument("--regime", choices=["auto", "resolved", "marginal", "subvoxel"], default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config(data_path=args.data, output_dir=args.output)

    if args.voxel_spacing:
        config.voxel_spacing_um = VoxelSpacing(*args.voxel_spacing)
    if args.fiber_diameter:
        config.fiber_diameter_um = args.fiber_diameter
    if args.regime:
        config.regime = args.regime

    config.validate()
    pipeline = FiberAnalysisPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
