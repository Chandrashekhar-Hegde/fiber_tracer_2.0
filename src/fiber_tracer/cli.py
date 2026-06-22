"""Command-line interface for fiber_tracer."""

import argparse
import logging
import sys
from typing import Any, Optional

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.pipeline import FiberAnalysisPipeline


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add RAFA pipeline arguments to *parser*."""
    parser.add_argument("--data", required=False, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=False, help="Output directory")
    parser.add_argument("--config", help="Path to YAML/JSON config")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, metavar=("Z", "Y", "X"))
    parser.add_argument("--fiber-diameter", type=float)
    parser.add_argument(
        "--regime", choices=["auto", "resolved", "marginal", "subvoxel"], default="auto"
    )
    parser.add_argument("--segmentation-method", choices=["otsu", "watershed", "unet"])
    parser.add_argument("--model-path", help="Path to a PyTorch checkpoint for method='unet'")


def _add_view_args(parser: argparse.ArgumentParser) -> None:
    """Add visualization arguments to *parser*."""
    parser.add_argument("--data", required=True, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=True, help="Output directory")


def _build_report_viz_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("report-viz", help="Generate interactive Plotly visualizations")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--output", required=True, help="Output HTML report")
    parser.set_defaults(func=_run_report_viz)


def _build_batch_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("batch", help="Process multiple volumes from a batch config")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON batch config")
    parser.add_argument(
        "--aggregate-csv", default="batch_summary.csv", help="Aggregate CSV output path"
    )
    parser.set_defaults(func=_run_batch)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAFA fiber analysis")
    parser.add_argument("--log-level", default="INFO")

    # Keep top-level pipeline arguments for backward compatibility.
    _add_pipeline_args(parser)

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the RAFA pipeline", aliases=["analyze"])
    _add_pipeline_args(run_parser)

    view_parser = subparsers.add_parser("view", help="Visualize results in napari")
    _add_view_args(view_parser)

    _build_report_viz_parser(subparsers)
    _build_batch_parser(subparsers)

    return parser


def _run_pipeline(args: argparse.Namespace) -> int:
    """Run the RAFA pipeline from parsed CLI arguments."""
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
    if args.segmentation_method:
        config.segmentation.method = args.segmentation_method
    if args.model_path:
        config.segmentation.model_path = args.model_path

    config.validate()
    pipeline = FiberAnalysisPipeline(config)
    pipeline.run()
    return 0


def _run_view(args: argparse.Namespace) -> int:
    """Launch the napari viewer with RAFA results."""
    from fiber_tracer.viz.napari_viewer import run_napari_viewer

    return run_napari_viewer(args.data, args.output)


def _run_report_viz(args: argparse.Namespace) -> int:
    """Generate an interactive Plotly report from a summary.json."""
    import json

    from fiber_tracer.viz.plotly_plots import generate_interactive_report

    with open(args.summary) as f:
        summary = json.load(f)
    generate_interactive_report(summary, args.output)
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    """Process multiple volumes from a batch config."""
    from fiber_tracer.batch import process_batch

    process_batch(args.config, aggregate_csv=args.aggregate_csv)
    return 0


def main(argv: Optional[list] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command is None or args.command in ("run", "analyze"):
        return _run_pipeline(args)
    if args.command == "view":
        return _run_view(args)
    if args.command == "report-viz":
        return _run_report_viz(args)
    if args.command == "batch":
        return _run_batch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
