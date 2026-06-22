# Third-Party Licenses and Attributions

## Core dependencies

- **NumPy** — BSD-3-Clause — https://numpy.org/
- **SciPy** — BSD-3-Clause — https://scipy.org/
- **scikit-image** — BSD-3-Clause — https://scikit-image.org/
- **pandas** — BSD-3-Clause — https://pandas.pydata.org/
- **Matplotlib** — PSF-based license — https://matplotlib.org/
- **tifffile** — BSD-3-Clause — https://pypi.org/project/tifffile/
- **PyYAML** — MIT — https://pyyaml.org/
- **tqdm** — MPL-2.0 AND MIT — https://tqdm.github.io/

## Optional dependencies

- **structure-tensor** — MIT — https://github.com/Skielex/structure-tensor
- **skan** — BSD-3-Clause — https://github.com/jni/skan
- **torch** — BSD-3-Clause — https://pytorch.org/
- **torchvision** — BSD-3-Clause — https://pytorch.org/vision/
- **scikit-learn** — BSD-3-Clause — https://scikit-learn.org/
- **nnunetv2** — Apache-2.0 — Linux-only extra due to transitive build constraints — https://github.com/MIC-DKFZ/nnUNet
- **gudhi** — MIT (Python modules ≥3.9.0) — https://gudhi.inria.fr/
- **plotly** — MIT — https://plotly.com/python/
- **napari** — BSD-3-Clause — https://napari.org/
- **zarr** — MIT — https://zarr.dev/
- **dask** — BSD-3-Clause — https://www.dask.org/
- **h5py** — BSD-3-Clause — https://www.h5py.org/

## Development / test dependencies

- **pytest** — MIT — https://pytest.org/
- **pytest-cov** — MIT — https://pytest-cov.readthedocs.io/
- **black** — MIT — https://black.readthedocs.io/
- **ruff** — MIT — https://docs.astral.sh/ruff/
- **mypy** — MIT — https://mypy-lang.org/

## Datasets used for validation

- **GF-PA66 3D XCT** — CC BY-SA 4.0 — DOI:10.5281/zenodo.4587827 — Bertoldo et al., Front. Mater. 2021.

## Notes

`pip install -e ".[all]"` no longer installs `ripser` because its transitive
dependency `hopcroftkarp` is GPLv3. Users who need Ripser persistence must
install it separately and comply with its license.

See `CITATIONS.md` for academic citations.
