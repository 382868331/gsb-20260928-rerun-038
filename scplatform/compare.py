"""并行比较多套筛选参数：逐个细胞说明进入/离开及原因。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import CountMatrix
from .qc import CellVerdict, QCParams, filter_cells


@dataclass(frozen=True)
class CellChange:
    cell_id: str
    batch: str
    total_counts: int
    detected_genes: int
    change: str  # "entered" | "left" | "kept" | "out_both"
    reasons_before: list[str]
    reasons_after: list[str]

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "batch": self.batch,
            "total_counts": self.total_counts,
            "detected_genes": self.detected_genes,
            "change": self.change,
            "reasons_before": self.reasons_before,
            "reasons_after": self.reasons_after,
        }


@dataclass
class QCComparison:
    label_before: str
    label_after: str
    params_before: dict
    params_after: dict
    cells: list[CellChange] = field(default_factory=list)
    empty_batches_before: list[str] = field(default_factory=list)
    empty_batches_after: list[str] = field(default_factory=list)

    @property
    def entered(self) -> list[CellChange]:
        return [c for c in self.cells if c.change == "entered"]

    @property
    def left(self) -> list[CellChange]:
        return [c for c in self.cells if c.change == "left"]

    def to_dict(self) -> dict:
        return {
            "label_before": self.label_before,
            "label_after": self.label_after,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "empty_batches_before": self.empty_batches_before,
            "empty_batches_after": self.empty_batches_after,
            "entered": [c.to_dict() for c in self.entered],
            "left": [c.to_dict() for c in self.left],
            "all_cells": [c.to_dict() for c in self.cells],
        }

    def format_text(self) -> str:
        lines = [
            f"筛选参数并行比较: {self.label_before} -> {self.label_after}",
            f"  进入细胞 ({len(self.entered)}):",
        ]
        for c in self.entered:
            lines.append(
                f"    + {c.cell_id} [{c.batch}] total={c.total_counts} "
                f"detected={c.detected_genes} 旧规则原因: {c.reasons_before or ['-']}"
            )
        lines.append(f"  离开细胞 ({len(self.left)}):")
        for c in self.left:
            lines.append(
                f"    - {c.cell_id} [{c.batch}] total={c.total_counts} "
                f"detected={c.detected_genes} 新规则原因: {c.reasons_after or ['-']}"
            )
        if self.empty_batches_after:
            lines.append(f"  新规则下全空批次: {self.empty_batches_after}")
        return "\n".join(lines)


def _verdict_index(verdicts: list[CellVerdict]) -> dict[str, CellVerdict]:
    return {v.cell_id: v for v in verdicts}


def compare_filters(
    matrix: CountMatrix,
    params_before: dict[str, QCParams],
    params_after: dict[str, QCParams],
    *,
    label_before: str = "before",
    label_after: str = "after",
) -> QCComparison:
    """用两套参数并行筛选同一原始矩阵，逐细胞比较去留。"""
    filtered_before, report_before = filter_cells(matrix, params_before)
    filtered_after, report_after = filter_cells(matrix, params_after)
    before = _verdict_index(report_before.verdicts)
    after = _verdict_index(report_after.verdicts)

    changes: list[CellChange] = []
    for cell_id in matrix.cell_ids:
        vb, va = before[cell_id], after[cell_id]
        if not vb.kept and va.kept:
            change = "entered"
        elif vb.kept and not va.kept:
            change = "left"
        elif vb.kept and va.kept:
            change = "kept"
        else:
            change = "out_both"
        changes.append(
            CellChange(
                cell_id=cell_id,
                batch=va.batch,
                total_counts=va.total_counts,
                detected_genes=va.detected_genes,
                change=change,
                reasons_before=list(vb.reasons),
                reasons_after=list(va.reasons),
            )
        )

    return QCComparison(
        label_before=label_before,
        label_after=label_after,
        params_before={b: p.to_dict() for b, p in params_before.items()},
        params_after={b: p.to_dict() for b, p in params_after.items()},
        cells=changes,
        empty_batches_before=list(report_before.empty_batches),
        empty_batches_after=list(report_after.empty_batches),
    )
