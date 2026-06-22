"""GF-PA66 3D XCT dataset download helper.

License: CC BY-SA 4.0
DOI: 10.5281/zenodo.4587827
Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229

This script does not redistribute the data. Visit the Zenodo record below and
download the HDF5 file(s) manually.
"""

ZENODO_RECORD_URL = "https://zenodo.org/records/4587827"
EXPECTED_FILES = [
    "GF-PA66_3D_XCT.h5",
]


def main():
    print("GF-PA66 3D XCT validation dataset")
    print(f"  Zenodo record: {ZENODO_RECORD_URL}")
    print("  License: CC BY-SA 4.0")
    print("  Citation: Bertoldo et al., Front. Mater. 2021, DOI:10.3389/fmats.2021.761229")
    print()
    print("Please download the HDF5 file(s) from the Zenodo record and place one of")
    print(f"the following files in your working directory: {EXPECTED_FILES}")
    print()
    print("Then run:")
    print("  python scripts/validate_gfpa66.py --data GF-PA66_3D_XCT.h5 --output results/gfpa66")


if __name__ == "__main__":
    main()
