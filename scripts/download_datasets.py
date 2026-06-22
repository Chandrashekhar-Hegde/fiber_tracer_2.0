"""Download open XCT datasets for fiber segmentation training.

Currently covers Henry Royce Institute datasets hosted on Zenodo. More
open composite-CT datasets can be registered in ``DATASETS`` as they are
identified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests  # type: ignore[import-untyped]
from tqdm import tqdm

DATASETS = {
    "henry_ncf_fatigue": {
        "url": "https://zenodo.org/record/4541235/files/0_cycles.zip",
        "archive": "0_cycles.zip",
        "source": "Henry Royce Institute / Prajapati et al.",
    },
    "henry_ud_compression": {
        "url": "https://zenodo.org/record/2597498/files/GFRP_Initial.zip",
        "archive": "GFRP_Initial.zip",
        "source": "Henry Royce Institute / Wang et al.",
    },
}


def _download(url: str, dest: Path) -> None:
    """Stream *url* to *dest* with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download open XCT fiber datasets")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
    )
    args = parser.parse_args(argv)

    selected = list(DATASETS.keys()) if "all" in args.datasets else args.datasets
    for key in selected:
        meta = DATASETS[key]
        out_dir = args.output / key
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = out_dir / meta["archive"]
        if archive.exists():
            print(f"{archive} already exists; skipping download.")
            continue
        print(f"Downloading {key} from {meta['url']}")
        _download(meta["url"], archive)
        print(f"Saved to {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
