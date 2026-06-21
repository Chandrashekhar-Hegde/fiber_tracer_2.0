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
    parser.add_argument("--data", required=False, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=False, help="Output directory")
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
        file_config = Config.from_file(args.config)
    else:
        file_config = Config()

    data_path = args.data or file_config.data_path
    output_dir = args.output or file_config.output_dir

    if not data_path:
        raise ValueError("data_path must be provided via --data or the configuration file")
    if not output_dir:
        raise ValueError("output_dir must be provided via --output or the configuration file")

    config = Config(
        data_path=data_path,
        output_dir=output_dir,
        voxel_spacing_um=file_config.voxel_spacing_um,
        fiber_diameter_um=file_config.fiber_diameter_um,
        regime=file_config.regime,
        processing=file_config.processing,
        segmentation=file_config.segmentation,
        orientation=file_config.orientation,
        analysis=file_config.analysis,
    )

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
