"""筛选参数的并行比较：同一原始矩阵跑多套规则，逐细胞给出去留变化。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import CountMatrix
from .qc import QCParams, QCReport, filter_cells


@dataclass(frozen=True)
class CellRef:
    """细胞的全局身份：(来源指纹, 细胞 ID)。

    相同细胞名若来自不同矩阵（指纹不同），是两个不同对象，
    比较与合并时绝不按名字混为一谈。
    """

    source_fingerprint: str
    cell_id: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_fingerprint, self.cell_id)


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
    source_fingerprint: str = ""

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
            "source_fingerprint": self.source_fingerprint,
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

    fingerprint = matrix.content_fingerprint()
    before = {CellRef(fingerprint, v.cell_id).key: v for v in baseline_report.verdicts}
    after = {CellRef(fingerprint, v.cell_id).key: v for v in candidate_report.verdicts}
    if set(before) != set(after):
        raise ValueError("两次筛选覆盖的细胞身份不一致，无法比较")

    changes: list[CellChange] = []
    for (_, cell_id), verdict in before.items():
        other = after[(fingerprint, cell_id)]
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
        source_fingerprint=matrix.content_fingerprint(),
    )


@dataclass
class MergeResult:
    """多套筛选方案在指定来源上的合并比较。

    - intersection：所有方案都保留的细胞（交集）；
    - only_kept[scheme]：仅该方案保留的细胞；
    - excluded[cell]：该细胞被哪些方案以什么原因排除；
    - name_collisions：同名但来源不同的细胞（永远不合并，单独列出）。
    """

    schemes: dict[str, CountMatrix]
    params: dict[str, dict[str, QCParams]]
    reports: dict[str, QCReport]
    intersection: list[CellRef]
    only_kept: dict[str, list[CellRef]]
    excluded: dict[CellRef, dict[str, list[str]]]
    name_collisions: list[dict]

    def intersection_ids(self) -> list[str]:
        return [ref.cell_id for ref in self.intersection]

    def only_kept_ids(self, scheme: str) -> list[str]:
        return [ref.cell_id for ref in self.only_kept[scheme]]

    def to_dict(self) -> dict:
        def ref(ref_obj: CellRef) -> dict:
            return {
                "source_fingerprint": ref_obj.source_fingerprint,
                "cell_id": ref_obj.cell_id,
            }

        return {
                "schemes": {
                    name: matrix.content_fingerprint()[:12]
                    for name, matrix in self.schemes.items()
                },
                "intersection": [ref(r) for r in self.intersection],
                "only_kept": {
                    name: [ref(r) for r in refs]
                    for name, refs in self.only_kept.items()
                },
                "excluded": [
                    {
                        **ref(cell_ref),
                        "exclusions": {
                            scheme: reasons for scheme, reasons in by_scheme.items()
                        },
                    }
                    for cell_ref, by_scheme in self.excluded.items()
                ],
                "name_collisions": self.name_collisions,
            }


def merge_qc(
    schemes: dict[str, CountMatrix],
    params_by_scheme: dict[str, dict[str, QCParams]],
) -> MergeResult:
    """合并比较多套筛选方案，明确交集、仅某方案保留的细胞与各自排除原因。

    ``schemes`` 为方案名 -> 该方案的来源矩阵；矩阵可以不同，但细胞身份按
    (来源指纹, 细胞 ID) 限定，**相同名称但不同来源的细胞不会被合并**，
    这类同名碰撞会在 ``name_collisions`` 中明确列出。
    """
    if set(schemes) != set(params_by_scheme):
        raise ValueError("方案名与参数键必须一一对应")
    if not schemes:
        raise ValueError("至少需要一个筛选方案")

    reports: dict[str, QCReport] = {}
    kept: dict[str, set[tuple[str, str]]] = {}
    verdict_by_ref: dict[tuple[str, str], dict[str, list[str]]] = {}

    for name, matrix in schemes.items():
        _, report = filter_cells(matrix, params_by_scheme[name])
        reports[name] = report
        fp = matrix.content_fingerprint()
        kept[name] = set()
        for v in report.verdicts:
            ref_key = (fp, v.cell_id)
            kept[name].add(ref_key) if v.kept else None
            verdict_by_ref.setdefault(ref_key, {})
            verdict_by_ref[ref_key][name] = list(v.reasons)

    common = set.intersection(*kept.values()) if kept else set()
    all_refs = set().union(*kept.values())
    # 被至少一个方案判过的细胞（含剔除），用于报告排除原因
    judged = set(verdict_by_ref)

    def source_of(fp: str) -> str:
        return next(name for name, m in schemes.items() if m.content_fingerprint() == fp)

    # 同名不同源：相同 cell_id 出现在不同来源指纹下
    by_cell_name: dict[str, set[str]] = {}
    for fp, cell_id in judged:
        by_cell_name.setdefault(cell_id, set()).add(fp)
    name_collisions = [
        {
            "cell_id": cell_id,
            "sources": [
                {"scheme": source_of(fp), "source_fingerprint": fp}
                for fp in sorted(fps)
            ],
        }
        for cell_id, fps in sorted(by_cell_name.items())
        if len(fps) > 1
    ]

    ref_objects = {key: CellRef(key[0], key[1]) for key in judged}
    order = {
        (m.content_fingerprint(), cid): i
        for m in schemes.values()
        for i, cid in enumerate(m.cell_ids)
    }

    intersection = [ref_objects[k] for k in sorted(common, key=lambda k: order.get(k, 1 << 30))]
    only_kept = {
        name: [
            ref_objects[k]
            for k in sorted(kept[name] - common, key=lambda k: order.get(k, 1 << 30))
        ]
        for name in schemes
    }
    excluded = {}
    for ref_key in sorted(judged - common, key=lambda k: order.get(k, 1 << 30)):
        by_scheme = {
            name: reasons
            for name, reasons in verdict_by_ref[ref_key].items()
            if reasons
        }
        if by_scheme:
            excluded[ref_objects[ref_key]] = by_scheme

    return MergeResult(
        schemes=dict(schemes),
        params=params_by_scheme,
        reports=reports,
        intersection=intersection,
        only_kept=only_kept,
        excluded=excluded,
        name_collisions=name_collisions,
    )
