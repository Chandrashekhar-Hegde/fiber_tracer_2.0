"""Command-line interface for fiber_tracer."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fiber_tracer.config import Config, VoxelSpacing
from fiber_tracer.pipeline import FiberAnalysisPipeline

DEFAULT_UNET_FEATURES = (8, 16, 32)


def _add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add RAFA pipeline arguments to *parser*."""
    parser.add_argument("--data", required=False, help="Path to TIFF stack or directory")
    parser.add_argument("--output", required=False, help="Output directory")
    parser.add_argument("--config", help="Path to YAML/JSON config")
    parser.add_argument("--voxel-spacing", nargs=3, type=float, metavar=("Z", "Y", "X"))
    parser.add_argument("--fiber-diameter", type=float)
    parser.add_argument(
        "--regime", choices=["auto", "resolved", "marginal", "subvoxel"], default=None
    )
    parser.add_argument("--segmentation-method", choices=["otsu", "watershed", "unet"])
    parser.add_argument("--model-path", help="Path to a PyTorch checkpoint for method='unet'")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Inference batch size for U-Net backend"
    )


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


def _build_model_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("model", help="Manage segmentation models")
    model_sub = parser.add_subparsers(dest="model_command", required=True)

    list_p = model_sub.add_parser("list", help="List registered models")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")
    list_p.set_defaults(func=_run_model_list)

    add_p = model_sub.add_parser("add", help="Add a local model")
    add_p.add_argument("--model-id", required=True)
    add_p.add_argument("--name", required=True)
    add_p.add_argument("--path", required=True)
    add_p.add_argument("--architecture", default="unet3d")
    add_p.add_argument("--version", default="unknown")
    add_p.add_argument("--description", default="")
    add_p.set_defaults(func=_run_model_add)

    remove_p = model_sub.add_parser("remove", help="Remove a model")
    remove_p.add_argument("model_id")
    remove_p.set_defaults(func=_run_model_remove)

    default_p = model_sub.add_parser("set-default", help="Set the default model")
    default_p.add_argument("model_id")
    default_p.set_defaults(func=_run_model_set_default)


def _run_model_list(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list_models()
    if args.json:
        print(json.dumps([asdict(m) for m in models]))
        return 0
    default = registry.get_default()
    for m in models:
        marker = " (default)" if default and m.model_id == default.model_id else ""
        print(f"{m.model_id}: {m.name} [{m.source}]{marker}")
    return 0


def _run_model_add(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    try:
        registry.add_model(
            model_id=args.model_id,
            name=args.name,
            path=args.path,
            architecture=args.architecture,
            version=args.version,
            description=args.description,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Added model {args.model_id}")
    return 0


def _run_model_remove(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    try:
        registry.remove_model(args.model_id)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Removed model {args.model_id}")
    return 0


def _run_model_set_default(args: argparse.Namespace) -> int:
    from fiber_tracer.models.registry import ModelRegistry

    registry = ModelRegistry()
    try:
        registry.set_default(args.model_id)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Default model set to {args.model_id}")
    return 0


def _build_experiment_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("experiment", help="Manage experiments")
    exp_sub = parser.add_subparsers(dest="experiment_command", required=True)

    list_p = exp_sub.add_parser("list", help="List experiments")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")
    list_p.set_defaults(func=_run_experiment_list)

    show_p = exp_sub.add_parser("show", help="Show experiment details")
    show_p.add_argument("experiment_id")
    show_p.set_defaults(func=_run_experiment_show)

    compare_p = exp_sub.add_parser("compare", help="Compare experiments by metric")
    compare_p.add_argument("experiment_ids", nargs="+")
    compare_p.add_argument("--metric", default="val_dice")
    compare_p.set_defaults(func=_run_experiment_compare)


def _run_experiment_list(args: argparse.Namespace) -> int:
    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    experiments = store.list_experiments()
    if args.json:
        print(json.dumps([asdict(e) for e in experiments]))
        return 0
    for exp in experiments:
        print(f"{exp.id} {exp.name} {exp.status}")
    return 0


def _run_experiment_show(args: argparse.Namespace) -> int:
    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    exp = store.get_experiment(args.experiment_id)
    if exp is None:
        print(f"Experiment {args.experiment_id} not found", file=sys.stderr)
        return 1
    print(json.dumps(asdict(exp), indent=2))
    return 0


def _run_experiment_compare(args: argparse.Namespace) -> int:
    from fiber_tracer.experiments.store import ExperimentStore

    store = ExperimentStore()
    result = store.compare(args.experiment_ids, metric=args.metric)
    print(json.dumps(result, indent=2))
    return 0


def _validate_train_args(args: argparse.Namespace) -> str | None:
    """Return an error message if train arguments are invalid, or None."""
    if args.epochs <= 0:
        return "--epochs must be greater than 0"
    if args.batch_size <= 0:
        return "--batch-size must be greater than 0"
    if args.lr <= 0:
        return "--lr must be greater than 0"
    if not 0 < args.val_fraction < 1:
        return "--val-fraction must be between 0 and 1 (exclusive)"
    return None


def _build_train_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("train", help="Train a segmentation model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-id", default="unet-v3.2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--features", nargs="+", type=int, default=DEFAULT_UNET_FEATURES)
    parser.add_argument("--name", default=None, help="Experiment name")
    parser.set_defaults(func=_run_train)


def _upsert_registry_model(
    registry: Any,
    model_id: str,
    checkpoint_path: Path,
) -> None:
    """Add or update a registry entry pointing at *checkpoint_path*."""
    existing = registry.get_model(model_id)
    if existing is not None:
        registry.remove_model(model_id)
    registry.add_model(
        model_id=model_id,
        name=model_id,
        path=str(checkpoint_path),
        architecture="unet3d",
        version="unknown",
        description=(
            f"Checkpoint produced by fiber-tracer train" f" ({datetime.now(UTC).isoformat()})"
        ),
    )


def _resolve_model_path(registry: Any, model_path: str) -> str:
    """Return the registered path if *model_path* is a registry ID, else itself."""
    if not Path(model_path).is_file():
        entry = registry.get_model(model_path)
        if entry is not None:
            return str(entry.path)
    return model_path


def _run_train(args: argparse.Namespace) -> int:
    from fiber_tracer.experiments.store import ExperimentStore
    from fiber_tracer.models.registry import ModelRegistry
    from fiber_tracer.training.trainer import UNetTrainer

    error = _validate_train_args(args)
    if error:
        print(f"Invalid argument: {error}", file=sys.stderr)
        return 1

    store = ExperimentStore()
    exp = store.create(
        name=args.name or f"train-{args.model_id}",
        type="train",
        model_id=args.model_id,
        dataset=args.dataset_dir,
        config_snapshot={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
            "val_fraction": args.val_fraction,
            "features": args.features,
        },
        artifact_dir=args.output_dir,
    )

    features = tuple(args.features)
    trainer = UNetTrainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        device=args.device,
        features=features,
    )
    try:
        trainer.train(experiment_id=exp.id)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    registry = ModelRegistry()
    checkpoint_path = Path(args.output_dir) / "checkpoint.pt"
    try:
        _upsert_registry_model(registry, args.model_id, checkpoint_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Failed to register checkpoint: {exc}", file=sys.stderr)
        return 1

    print(f"Registered checkpoint for model {args.model_id} at {checkpoint_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAFA fiber analysis")
    parser.add_argument(
        "--version",
        action="version",
        version=f"fiber-tracer {importlib.metadata.version('fiber-tracer')}",
    )
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
    _build_model_parser(subparsers)
    _build_experiment_parser(subparsers)
    _build_train_parser(subparsers)

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
    if args.regime is not None:
        config.regime = args.regime
    if args.segmentation_method:
        config.segmentation.method = args.segmentation_method
    if args.model_path:
        from fiber_tracer.models.registry import ModelRegistry

        registry = ModelRegistry()
        config.segmentation.model_path = _resolve_model_path(registry, args.model_path)
    if args.batch_size is not None:
        config.segmentation.batch_size = args.batch_size

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


def main(argv: list | None = None) -> int:
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
    if args.command == "model":
        return int(args.func(args))
    if args.command == "experiment":
        return int(args.func(args))
    if args.command == "train":
        return int(args.func(args))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
