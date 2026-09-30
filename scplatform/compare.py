"""筛选参数的并行比较：同一原始矩阵跑多套规则，逐细胞给出去留变化。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import CountMatrix
from .qc import QCParams, QCReport, filter_cells


@dataclass
class CellChange:
    cell_id: str
    batch: str
    before_kept: bool
    after_kept: bool
    before_reasons: list[str] = field(default_factory=list)
    after_reasons: list[str] = field(default_factory=list)

    @property
    def direction(self) -> str:
        if not self.before_kept and self.after_kept:
            return "entered"
        if self.before_kept and not self.after_kept:
            return "left"
        return "unchanged"


@dataclass
class QCComparison:
    """两套筛选参数在同一原始矩阵上的并行比较结果。"""

    baseline_label: str
    candidate_label: str
    baseline_report: QCReport
    candidate_report: QCReport
    changes: list[CellChange]

    @property
    def entered_ids(self) -> list[str]:
        return [c.cell_id for c in self.changes if c.direction == "entered"]

    @property
    def left_ids(self) -> list[str]:
        return [c.cell_id for c in self.changes if c.direction == "left"]

    def to_dict(self) -> dict:
        def verdict_map(report: QCReport) -> dict[str, dict]:
            return {
                v.cell_id: {"kept": v.kept, "reasons": list(v.reasons)}
                for v in report.verdicts
            }

        before_map = verdict_map(self.baseline_report)
        after_map = verdict_map(self.candidate_report)
        all_ids = [v.cell_id for v in self.baseline_report.verdicts]
        return {
            "baseline": self.baseline_label,
            "candidate": self.candidate_label,
            "entered": self.entered_ids,
            "left": self.left_ids,
            "cells": [
                {
                    "cell_id": cell_id,
                    "change": next(c.direction for c in self.changes if c.cell_id == cell_id),
                    "baseline": before_map[cell_id],
                    "candidate": after_map[cell_id],
                }
                for cell_id in all_ids
            ],
            "empty_batches": {
                "baseline": list(self.baseline_report.empty_batches),
                "candidate": list(self.candidate_report.empty_batches),
            },
        }


def compare_qc(
    matrix: CountMatrix,
    baseline_params: dict[str, QCParams],
    candidate_params: dict[str, QCParams],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> QCComparison:
    """在同一份原始矩阵上并行跑两套筛选规则，逐细胞比较去留及原因。"""
    _, baseline_report = filter_cells(matrix, baseline_params)
    _, candidate_report = filter_cells(matrix, candidate_params)

    before = {v.cell_id: v for v in baseline_report.verdicts}
    after = {v.cell_id: v for v in candidate_report.verdicts}
    if set(before) != set(after):
        raise ValueError("两次筛选覆盖的细胞身份不一致，无法比较")

    changes: list[CellChange] = []
    for cell_id, verdict in before.items():
        other = after[cell_id]
        changes.append(
            CellChange(
                cell_id=cell_id,
                batch=verdict.batch,
                before_kept=verdict.kept,
                after_kept=other.kept,
                before_reasons=list(verdict.reasons),
                after_reasons=list(other.reasons),
            )
        )
    return QCComparison(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        changes=changes,
    )
