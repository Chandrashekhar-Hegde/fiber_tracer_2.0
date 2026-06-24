"""Cross-platform path helpers."""
from __future__ import annotations

import os
from pathlib import Path


def get_config_dir() -> str:
    """Return the fiber-tracer configuration directory.

    Uses ``FIBER_TRACER_CONFIG_DIR`` if set, otherwise ``~/.config/fiber-tracer``
    on POSIX or ``~/AppData/Roaming/fiber-tracer`` on Windows.
    """
    if env_dir := os.environ.get("FIBER_TRACER_CONFIG_DIR", "").strip():
        path = Path(env_dir)
    elif os.name == "nt":
        path = Path.home() / "AppData" / "Roaming" / "fiber-tracer"
    else:
        path = Path.home() / ".config" / "fiber-tracer"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
