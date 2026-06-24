"""Shared citation and caveat metadata for reports."""

from __future__ import annotations

CITATIONS = [
    (
        "Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and "
        "predict fiber orientation in short fiber composites. Journal of Rheology, 31(8), "
        "751–784."
    ),
    (
        "Jeppesen, N., et al. (2021). Quantifying effects of manufacturing methods on "
        "fiber orientation in unidirectional composites using structure tensor analysis. "
        "Composites Part A, 149, 106541."
    ),
    ("van der Walt et al. (2014). scikit-image: Image processing in Python. PeerJ, " "2, e453."),
]

REGIME_CAVEATS = {
    "resolved": (
        "Resolved-regime results depend on successful segmentation and skeletonization. "
        "Overlapping or sub-voxel fibers may be misclassified."
    ),
    "marginal": (
        "Marginal-regime results are computed from a local structure-tensor field. "
        "Accuracy degrades when the fiber diameter is close to the voxel size."
    ),
    "subvoxel": (
        "Subvoxel-regime results aggregate orientations over large windows because "
        "individual fibers are not resolved. Only population-level orientation statistics "
        "are reliable."
    ),
}
