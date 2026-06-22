"""GF-PA66 3D XCT dataset download helper.

License: CC BY-SA 4.0
DOI: 10.5281/zenodo.4587827
Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229

This script does not redistribute the data. It downloads the file directly from
Zenodo after the user accepts the dataset license.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

ZENODO_RECORD_URL = "https://zenodo.org/api/records/4587827"


def fetch_record_metadata(record_url: str = ZENODO_RECORD_URL, timeout: int = 30) -> dict:
    """Fetch Zenodo record metadata."""
    response = requests.get(record_url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def list_files(metadata: dict) -> list[dict]:
    """Return a list of downloadable files with keys, sizes, and links."""
    files = []
    for f in metadata.get("files", []):
        files.append(
            {
                "key": f["key"],
                "size": f["size"],
                "url": f["links"]["self"],
            }
        )
    return files


def download_file(
    url: str,
    dest: Path,
    expected_size: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> None:
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = expected_size or int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(dest, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download the GF-PA66 3D XCT dataset")
    parser.add_argument("--output-dir", default=".", help="Directory to save the file")
    parser.add_argument(
        "--file",
        default=None,
        help=("Filename to download (e.g. pa66_volumes.h5). " "If omitted, list available files."),
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm acceptance of CC BY-SA 4.0",
    )
    parser.add_argument("--list", action="store_true", help="List available files and exit")
    args = parser.parse_args(argv)

    print("GF-PA66 3D XCT validation dataset")
    print("  Zenodo record:", ZENODO_RECORD_URL.replace("/api/", "/"))
    print("  License: CC BY-SA 4.0")
    print("  Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229")
    print()

    metadata = fetch_record_metadata()
    files = list_files(metadata)

    if args.list or args.file is None:
        print("Available files:")
        for f in files:
            print(f"  {f['key']} ({f['size'] / (1024**3):.2f} GB)")
        return 0

    if not args.accept_license:
        print("ERROR: You must accept the CC BY-SA 4.0 license with --accept-license to download.")
        return 1

    selected = next((f for f in files if f["key"] == args.file), None)
    if selected is None:
        print(f"ERROR: File '{args.file}' not found in Zenodo record.")
        print("Available files:", [f["key"] for f in files])
        return 1

    dest = Path(args.output_dir) / selected["key"]
    if dest.exists() and dest.stat().st_size == selected["size"]:
        print(f"File already exists and has correct size: {dest}")
        return 0

    print(f"Downloading {selected['key']} ({selected['size'] / (1024**3):.2f} GB) to {dest}")
    download_file(selected["url"], dest, expected_size=selected["size"])
    print("Download complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
