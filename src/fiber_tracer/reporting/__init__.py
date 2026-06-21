"""Reporting exporters and shared report metadata."""

CITATIONS = [
    "Advani, S. G., & Tucker III, C. L. (1987). The use of tensors to describe and predict fiber orientation in short fiber composites. Journal of Rheology, 31(8), 751–784.",
    "Jeppesen, N., et al. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. Composites Part A, 149, 106541.",
    "van der Walt et al. (2014). scikit-image: Image processing in Python. PeerJ, 2, e453.",
]

from fiber_tracer.reporting.json import write_json_report
from fiber_tracer.reporting.csv import write_csv_report
from fiber_tracer.reporting.html import write_html_report

__all__ = [
    "write_json_report",
    "write_csv_report",
    "write_html_report",
    "CITATIONS",
]
