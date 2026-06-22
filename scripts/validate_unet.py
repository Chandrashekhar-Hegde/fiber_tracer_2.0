"""Validate a trained U-Net on full volumes and report Dice scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fiber_tracer.backends.ml_segmentation import MLSegmentationBackend
from fiber_tracer.io import load_tiff_stack, save_tiff_stack


def _dice(pred: np.ndarray, target: np.ndarray) -> float:
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return 2 * inter / denom if denom > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
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
        save_tiff_stack(args.output, pred.astype(np.uint8) * 255)

    if args.label:
        target = load_tiff_stack(args.label).astype(bool)
        dice = _dice(pred, target)
        print(f"Dice: {dice:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
