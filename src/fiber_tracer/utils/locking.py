"""Cross-process file locking helpers."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

if sys.platform == "win32":
    _POSIX = False
else:
    _POSIX = True
    import fcntl


@contextmanager
def file_lock(path: str | Path) -> Generator[None, None, None]:
    """Acquire a cross-process advisory lock on ``path``.

    On POSIX systems this uses ``fcntl.flock``.  On Windows this is a no-op
    context manager because cross-process locking is not implemented; this is
    acceptable for local single-user usage.
    """
    path = Path(path)
    if not _POSIX:
        yield
        return

    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
