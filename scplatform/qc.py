"""质量筛选：每个细胞的去留与原因全部可追溯。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .model import CountMatrix


@dataclass(frozen=True)
class QCParams:
    """单个批次的筛选规则。不同批次可使用不同规则。""" 

    min_total_counts: int = 500
    min_detected_genes: int = 200

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CellVerdict:
    cell_id: str
    batch: str
    kept: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class QCReport:
    """筛选报告：保留/剔除明细 + 空样本与全零细胞警告。""" 

    verdicts: list[CellVerdict]
    empty_batches: list[str]
    all_zero_cells: list[str]
    source_fingerprint: str = ""

    @property
    def kept_ids(self) -> list[str]:
        return [v.cell_id for v in self.verdicts if v.kept]

    @property
    def dropped_ids(self) -> list[str]:
        return [v.cell_id for v in self.verdicts if not v.kept]

    def to_dict(self) -> dict:
        return {
            "kept": self.kept_ids,
            "dropped": [
                {"cell_id": v.cell_id, "batch": v.batch, "reasons": v.reasons}
                for v in self.verdicts
                if not v.kept
            ],
            "empty_batches": self.empty_batches,
            "all_zero_cells": self.all_zero_cells,
            "source_fingerprint": self.source_fingerprint,
        }


def filter_cells(
    matrix: CountMatrix, params_by_batch: dict[str, QCParams]
) -> tuple[CountMatrix, QCReport]:
    """按批次各自的规则筛选细胞，返回 (筛选后矩阵, 报告)。""" 
    missing = set(matrix.batches) - set(params_by_batch)
    if missing:
        raise ValueError(f"以下批次缺少筛选规则: {sorted(missing)}")

    verdicts: list[CellVerdict] = []
    all_zero: list[str] = []
    keep_idx: list[int] = []

    for i, cell_id in enumerate(matrix.cell_ids):
        batch = matrix.batch_of_cell[i]
        params = params_by_batch[batch]
        total = matrix.cell_total(i)
        detected = int((matrix.counts[i] > 0).sum())

        reasons: list[str] = []
        if total == 0:
            all_zero.append(cell_id)
            reasons.append("all_zero_counts")
        if total < params.min_total_counts:
            reasons.append(f"total_counts<{params.min_total_counts}")
        if detected < params.min_detected_genes:
            reasons.append(f"detected_genes<{params.min_detected_genes}")

        kept = not reasons
        if kept:
            keep_idx.append(i)
        verdicts.append(CellVerdict(cell_id=cell_id, batch=batch, kept=kept, reasons=reasons))

    kept_batches = {matrix.batch_of_cell[i] for i in keep_idx}
    empty_batches = [b for b in matrix.batches if b not in kept_batches]

    report = QCReport(
        verdicts=verdicts,
        empty_batches=empty_batches,
        all_zero_cells=all_zero,
        source_fingerprint=matrix.content_fingerprint(),
    )
    return matrix.subset_cells(keep_idx), report
