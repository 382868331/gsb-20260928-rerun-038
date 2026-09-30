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

    @property
    def intersection_ids(self) -> list[str]:
        """两套方案都保留的细胞（交集）。"""
        return [c.cell_id for c in self.changes if c.before_kept and c.after_kept]

    @property
    def only_baseline_ids(self) -> list[str]:
        """仅 baseline 方案保留、candidate 排除的细胞。"""
        return [c.cell_id for c in self.changes if c.before_kept and not c.after_kept]

    @property
    def only_candidate_ids(self) -> list[str]:
        """仅 candidate 方案保留、baseline 排除的细胞。"""
        return [c.cell_id for c in self.changes if not c.before_kept and c.after_kept]

    def merge_view(self) -> dict:
        """合并比较视图：交集、仅某方案保留的细胞及各自排除原因。

        身份来自同一份原始矩阵（compare_qc 已校验两次筛选覆盖一致），
        同名但不同来源的细胞不会在此被合并。
        """
        def detail(cell_id: str) -> dict:
            change = next(c for c in self.changes if c.cell_id == cell_id)
            return {
                "cell_id": cell_id,
                "batch": change.batch,
                self.baseline_label + "_reasons": list(change.before_reasons),
                self.candidate_label + "_reasons": list(change.after_reasons),
            }

        return {
            "intersection": [detail(c) for c in self.intersection_ids],
            "only_" + self.baseline_label: [detail(c) for c in self.only_baseline_ids],
            "only_" + self.candidate_label: [detail(c) for c in self.only_candidate_ids],
        }

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
            "intersection": self.intersection_ids,
            "merge": self.merge_view(),
            "source_fingerprint": self.baseline_report.source_fingerprint,
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
    merge_reports(baseline_report, candidate_report)

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


def merge_reports(baseline_report: QCReport, candidate_report: QCReport) -> None:
    """校验两份筛选报告来自同一来源，可以安全合并比较。

    来源指纹不同（同名细胞可能来自不同矩阵）或任一侧存在重复身份时
    拒绝，绝不把相同名称但不同来源的细胞合并成一条记录。
    """
    if baseline_report.source_fingerprint != candidate_report.source_fingerprint:
        raise ValueError(
            "两份筛选结果来源指纹不同，拒绝合并比较："
            "相同名称的细胞可能来自不同矩阵，不能合并"
        )
    for label, report in (("baseline", baseline_report), ("candidate", candidate_report)):
        ids = [v.cell_id for v in report.verdicts]
        dupes = sorted({c for c in ids if ids.count(c) > 1})
        if dupes:
            raise ValueError(f"{label} 结果含重复细胞身份，拒绝合并比较: {dupes}")
