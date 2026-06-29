"""Recover files from a ZIP that is missing its central directory.

This is intentionally minimal: it scans local file headers and extracts the
payload between headers (or from the last header to EOF).  It is meant for
internal, non-shareable data only.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

LOCAL_HEADER = b"PK\x03\x04"


def _find_local_headers(path: Path) -> list[int]:
    """Return byte offsets of all local file headers in *path*."""
    offsets: list[int] = []
    with open(path, "rb") as f:
        # Read in chunks to avoid loading multi-GB files.
        chunk_size = 8 * 1024 * 1024
        overlap = 4
        prev_tail = b""
        position = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buf = prev_tail + chunk
            start = 0
            while True:
                idx = buf.find(LOCAL_HEADER, start)
                if idx == -1:
                    break
                abs_idx = position - len(prev_tail) + idx
                # Avoid duplicates at chunk boundaries.
                if not offsets or offsets[-1] != abs_idx:
                    offsets.append(abs_idx)
                start = idx + 1
            prev_tail = chunk[-overlap:] if len(chunk) >= overlap else b""
            position += len(chunk)
    return offsets


def _read_header(path: Path, offset: int) -> tuple[str, int]:
    """Return (file_name, data_offset) for the header at *offset*."""
    with open(path, "rb") as f:
        f.seek(offset)
        hdr = f.read(30)
    if len(hdr) < 30:
        raise ValueError(f"Incomplete header at {offset}")
    (
        _sig,
        _version,
        _flags,
        _method,
        _time,
        _date,
        _crc,
        _comp_size,
        _uncomp_size,
        name_len,
        extra_len,
    ) = struct.unpack("<4sHHHHHIIIHH", hdr)
    with open(path, "rb") as f:
        f.seek(offset + 30)
        name = f.read(name_len).decode("utf-8", errors="replace")
    data_offset = offset + 30 + name_len + extra_len
    return name, data_offset


def recover(input_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = _find_local_headers(input_path)
    if not offsets:
        raise RuntimeError("No local file headers found")

    file_size = input_path.stat().st_size
    extracted: list[Path] = []

    for i, offset in enumerate(offsets):
        name, data_offset = _read_header(input_path, offset)
        safe_name = name.replace("..", "_").lstrip("/")
        dest = output_dir / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        # End of this file's data is the next header, or EOF.
        next_offset = offsets[i + 1] if i + 1 < len(offsets) else file_size
        data_length = next_offset - data_offset

        # Heuristic: if the header says the data is stored and a data
        # descriptor may be present, trim a trailing 16-byte descriptor.
        if data_length > 16:
            data_length -= 16

        with open(input_path, "rb") as f:
            f.seek(data_offset)
            data = f.read(data_length)
        dest.write_bytes(data)
        extracted.append(dest)
        print(f"Extracted {dest} ({len(data):,} bytes)")

    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover files from a truncated ZIP")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    recover(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
