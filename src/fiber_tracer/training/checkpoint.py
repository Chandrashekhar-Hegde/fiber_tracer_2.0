"""Checkpoint helpers with metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model state dict and metadata to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint produced by ``save_checkpoint``."""
    return torch.load(path, map_location="cpu", weights_only=False)
