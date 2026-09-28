"""Minimal single-cell research platform.

Stable identifiers for cells, genes and sample batches; traceable QC;
immutable raw data; versioned, parameter-derived analysis results.
"""

from .models import CountsDataset, QcParams, NormalizationParams
from .qc import ValidationReport, validate_dataset, run_qc, QcResult
from .normalize import run_normalization, NormalizationResult
from .store import ResultStore
from .io import build_dataset

__all__ = [
    "CountsDataset",
    "QcParams",
    "NormalizationParams",
    "ValidationReport",
    "validate_dataset",
    "run_qc",
    "QcResult",
    "run_normalization",
    "NormalizationResult",
    "ResultStore",
    "build_dataset",
]
