"""单细胞数据研究平台：质控筛选、归一化、版本化结果保存。"""

from .io import load_matrix_csv, write_matrix_csv
from .model import CountMatrix
from .qc import QCParams, QCReport, filter_cells
from .compare import CellChange, QCComparison, compare_filters
from .normalize import NormalizeParams, normalize
from .store import ImmutableResultError, ResultStore

__all__ = [
    "CountMatrix",
    "load_matrix_csv",
    "write_matrix_csv",
    "QCParams",
    "QCReport",
    "filter_cells",
    "CellChange",
    "QCComparison",
    "compare_filters",
    "NormalizeParams",
    "normalize",
    "ResultStore",
    "ImmutableResultError",
]
