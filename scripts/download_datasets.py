"""Download open XCT datasets for fiber segmentation training.

Currently covers Henry Royce Institute datasets hosted on Zenodo. More
open composite-CT datasets can be registered in ``DATASETS`` as they are
identified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import unquote

import requests  # type: ignore[import-untyped]
from tqdm import tqdm

# Each dataset maps to one or more files to download.  The ``files`` value
# is a list of (url, filename) tuples.  ``filename`` is the local name; when
# omitted the basename is derived from the URL.
DATASETS: dict[str, dict[str, object]] = {
    "henry_ncf_fatigue": {
        "source": "Henry Royce Institute / Prajapati et al.",
        "files": [
            (
                "https://zenodo.org/record/4541235/files/0_cycles.zip",
                "0_cycles.zip",
            ),
        ],
    },
    "henry_ud_compression": {
        "source": "Henry Royce Institute / Wang et al.",
        "files": [
            (
                "https://zenodo.org/record/2597498/files/GFRP_Initial.zip",
                "GFRP_Initial.zip",
            ),
        ],
    },
    "henry_ud_compression_static": {
        "source": "Henry Royce Institute / Wang et al.",
        "files": [
            (
                "https://zenodo.org/api/records/13348028/files/GFRP_Static_200N.zip/content",
                "GFRP_Static_200N.zip",
            ),
            (
                "https://zenodo.org/api/records/13348028/files/GFRP_Static_600N.zip/content",
                "GFRP_Static_600N.zip",
            ),
        ],
    },
    "henry_benchmark_gfrp_cfrp": {
        "source": "Henry Royce Institute / MAX IV Laboratory",
        "files": [
            (
                "https://zenodo.org/api/records/12743155/files/Glass%201.tif/content",
                "Glass_1.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%202.tif/content",
                "Glass_2.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%203.tif/content",
                "Glass_3.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%204.tif/content",
                "Glass_4.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%201-Sub.tif/content",
                "Glass_1_Sub.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%202-Sub.tif/content",
                "Glass_2_Sub.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%203-Sub.tif/content",
                "Glass_3_Sub.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Glass%204-Sub.tif/content",
                "Glass_4_Sub.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Carbon%201.tif/content",
                "Carbon_1.tif",
            ),
            (
                "https://zenodo.org/api/records/12743155/files/Carbon%201-Sub.tif/content",
                "Carbon_1_Sub.tif",
            ),
        ],
    },
    "dtu_pultruded_cfrp": {
        "source": "Technical University of Denmark / Mageira et al.",
        "files": [
            (
                "https://zenodo.org/api/records/18364215/files/x_ray_ct_scan_sample_a.h5/content",
                "x_ray_ct_scan_sample_a.h5",
            ),
        ],
    },
    "ivw_short_gfrp": {
        "source": "Leibniz-Institut fuer Verbundwerkstoffe / Boos et al.",
        "files": [
            (
                "https://zenodo.org/api/records/15852957/files/sGFRP_stitched_images_3x4.zip/content",
                "sGFRP_stitched_images_3x4.zip",
            ),
            (
                "https://zenodo.org/api/records/15852957/files/sGFRP_metadata.json/content",
                "sGFRP_metadata.json",
            ),
        ],
    },
    "ivw_carbon_twill_weave": {
        "source": "Leibniz-Institut fuer Verbundwerkstoffe / Boos et al.",
        "files": [
            (
                "https://zenodo.org/api/records/14946081/files/CF_weave_stitched_images_2x2.zip/content",
                "CF_weave_stitched_images_2x2.zip",
            ),
            (
                "https://zenodo.org/api/records/14946081/files/8L-G-weave_meta.json/content",
                "8L-G-weave_meta.json",
            ),
        ],
    },
    "ivw_recycled_cfrp": {
        "source": "Leibniz-Institut fuer Verbundwerkstoffe / Boos et al.",
        "files": [
            (
                "https://zenodo.org/api/records/14945398/files/rCF_stitched_images_2x2.zip/content",
                "rCF_stitched_images_2x2.zip",
            ),
        ],
    },
}


def _derive_filename(url: str) -> str:
    """Return a safe local filename from a Zenodo content URL."""
    # URLs from Zenodo API end in ``/files/<name>/content``.
    parts = Path(unquote(url)).parts
    if len(parts) >= 2 and parts[-1] == "content":
        return parts[-2].replace(" ", "_")
    return Path(unquote(url)).name


def _download(url: str, dest: Path) -> None:
    """Stream *url* to *dest* with a progress bar, resuming if possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {}
    if dest.exists():
        headers["Range"] = f"bytes={dest.stat().st_size}-"
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code == 416:
        # Already fully downloaded.
        print(f"{dest.name} already complete.")
        return
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    mode = "ab" if resp.status_code == 206 else "wb"
    with (
        open(dest, mode) as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def _files_for(meta: dict[str, object]) -> list[tuple[str, str]]:
    """Normalize dataset metadata to a list of (url, filename)."""
    raw_files = meta.get("files")
    if isinstance(raw_files, list):
        result: list[tuple[str, str]] = []
        for item in raw_files:
            if isinstance(item, tuple):
                url, name = item
            else:
                url = str(item)
                name = _derive_filename(url)
            result.append((url, name))
        return result
    # Backward compatibility with single-archive datasets.
    url = str(meta["url"])
    name = str(meta.get("archive", _derive_filename(url)))
    return [(url, name)]


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
        print(f"Downloading {key} ({meta.get('source', 'unknown source')})")
        for url, filename in _files_for(meta):
            dest = out_dir / filename
            if dest.exists():
                print(f"  {dest.name} already exists; resuming if needed.")
            _download(url, dest)
            print(f"  Saved {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
