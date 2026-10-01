"""跨批次联合分析集：基因可重叠但不完全相同，缺测绝不当作零表达。

- 每个值都带 measured 掩码：True=真实测量，False=该批次未测此基因；
  缺测位置以 NaN 存储，任何统计都只在 measured 范围内进行。
- 两种联合模式：common（只用共同基因）/ union（保留全部基因并明确
  每个批次缺测哪些基因）；compare_join_modes 比较两种选择对保留
  细胞与质量判断的影响。
- 细胞显示名相同仍按来源区分：内部身份为 "批次::细胞ID"，
  显示名仅用于展示，绝不作为合并依据。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import CountMatrix
from .qc import QCParams

JOIN_MODES = ("common", "union")


def cell_key(batch: str, cell_id: str) -> str:
    """联合集中的细胞内部身份：来源批次 + 显示名，同名不同源绝不合并。"""
    return f"{batch}::{cell_id}"


@dataclass(frozen=True)
class JointMatrix:
    """联合分析集：values 中缺测为 NaN，measured 掩码标明每个值是否真实测量。"""

    values: np.ndarray  # cells x genes，缺测为 NaN，绝不用 0 冒充
    measured: np.ndarray  # cells x genes 布尔掩码，True=该批次真实测量
    cell_keys: tuple[str, ...]  # "批次::细胞ID" 内部身份
    display_names: tuple[str, ...]  # 原始细胞 ID，仅展示用
    batch_of_cell: tuple[str, ...]
    gene_ids: tuple[str, ...]
    mode: str
    source_fingerprints: dict[str, str]  # 批次 -> 来源矩阵内容指纹

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        measured = np.asarray(self.measured, dtype=bool)
        if values.shape != measured.shape:
            raise ValueError("values 与 measured 掩码形状不一致")
        n_cells, n_genes = values.shape
        if len(self.cell_keys) != n_cells or len(self.display_names) != n_cells:
            raise ValueError("细胞身份数量与矩阵行数不一致")
        if len(self.batch_of_cell) != n_cells:
            raise ValueError("批次归属数量与细胞数不一致")
        if len(self.gene_ids) != n_genes:
            raise ValueError("基因身份数量与矩阵列数不一致")
        if len(set(self.cell_keys)) != n_cells:
            raise ValueError("联合集含重复细胞来源身份，拒绝构造")
        if len(set(self.gene_ids)) != n_genes:
            raise ValueError("联合集含重复基因身份，拒绝构造")
        if self.mode not in JOIN_MODES:
            raise ValueError(f"未知联合模式: {self.mode!r}，可选 {JOIN_MODES}")
        # 缺测必须是 NaN、测量位置不得是 NaN：缺测与零表达严格区分
        if np.isnan(values[measured]).any():
            raise ValueError("测量位置出现 NaN，缺测掩码与数值不一致")
        if not np.isnan(values[~measured]).all():
            raise ValueError("缺测位置必须存储为 NaN，不能用 0 或其他值冒充")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "measured", measured)

    @property
    def n_cells(self) -> int:
        return self.values.shape[0]

    @property
    def n_genes(self) -> int:
        return self.values.shape[1]

    @property
    def batches(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.batch_of_cell)))

    def missing_genes_by_batch(self) -> dict[str, list[str]]:
        """每个批次未测量的基因（缺测范围说明，仅 union 模式非空）。"""
        result: dict[str, list[str]] = {}
        for batch in self.batches:
            rows = [i for i, b in enumerate(self.batch_of_cell) if b == batch]
            missing = [
                self.gene_ids[j]
                for j in range(self.n_genes)
                if not self.measured[rows, j].any()
            ]
            result[batch] = missing
        return result

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "gene_ids": list(self.gene_ids),
            "batches": list(self.batches),
            "missing_genes_by_batch": self.missing_genes_by_batch(),
            "source_fingerprints": dict(self.source_fingerprints),
            "cells": [
                {
                    "cell_key": self.cell_keys[i],
                    "display_name": self.display_names[i],
                    "batch": self.batch_of_cell[i],
                    "n_measured_genes": int(self.measured[i].sum()),
                }
                for i in range(self.n_cells)
            ],
        }


def join_batches(matrices: dict[str, CountMatrix], mode: str) -> JointMatrix:
    """把若干批次的原始矩阵联合成一个分析集。

    matrices: 批次名 -> 该批次的 CountMatrix。同名细胞在不同批次中
    各自保留，以 "批次::细胞ID" 区分，绝不按显示名合并。
    """
    if mode not in JOIN_MODES:
        raise ValueError(f"未知联合模式: {mode!r}，可选 {JOIN_MODES}")
    if not matrices:
        raise ValueError("至少需要一个批次才能联合")
    for batch, matrix in matrices.items():
        foreign = sorted(set(matrix.batch_of_cell) - {batch})
        if foreign:
            raise ValueError(
                f"批次 {batch} 的矩阵混入了其他批次的细胞: {foreign}，拒绝联合"
            )

    gene_sets = {b: set(m.gene_ids) for b, m in matrices.items()}
    if mode == "common":
        common = set.intersection(*gene_sets.values())
        # 保持第一个批次的基因顺序，保证联合结果基因顺序可核对
        first = next(iter(matrices.values()))
        genes = tuple(g for g in first.gene_ids if g in common)
        if not genes:
            raise ValueError("各批次没有共同基因，common 模式无法联合")
    else:  # union：保留全部基因，顺序为各批次首次出现顺序
        seen: dict[str, None] = {}
        for matrix in matrices.values():
            for gene in matrix.gene_ids:
                seen.setdefault(gene)
        genes = tuple(seen)

    gene_col = {g: j for j, g in enumerate(genes)}
    n_rows = sum(m.n_cells for m in matrices.values())
    values = np.full((n_rows, len(genes)), np.nan)
    measured = np.zeros(values.shape, dtype=bool)
    cell_keys: list[str] = []
    display_names: list[str] = []
    batch_of_cell: list[str] = []
    fingerprints: dict[str, str] = {}

    row = 0
    for batch, matrix in matrices.items():
        fingerprints[batch] = matrix.content_fingerprint()
        src_col = {g: matrix.gene_ids.index(g) for g in matrix.gene_ids if g in gene_col}
        for i, cid in enumerate(matrix.cell_ids):
            cell_keys.append(cell_key(batch, cid))
            display_names.append(cid)
            batch_of_cell.append(batch)
            for gene, jsrc in src_col.items():
                j = gene_col[gene]
                values[row, j] = matrix.counts[i, jsrc]
                measured[row, j] = True
            row += 1

    return JointMatrix(
        values=values,
        measured=measured,
        cell_keys=tuple(cell_keys),
        display_names=tuple(display_names),
        batch_of_cell=tuple(batch_of_cell),
        gene_ids=genes,
        mode=mode,
        source_fingerprints=fingerprints,
    )


def _qc_on_joint(joint: JointMatrix, params_by_batch: dict[str, QCParams]) -> dict[str, list[str]]:
    """在联合集上按批次规则做质量判断，统计只落在 measured 范围内。"""
    missing = set(joint.batches) - set(params_by_batch)
    if missing:
        raise ValueError(f"以下批次缺少筛选规则: {sorted(missing)}")
    verdicts: dict[str, list[str]] = {}
    for i, key in enumerate(joint.cell_keys):
        params = params_by_batch[joint.batch_of_cell[i]]
        mask = joint.measured[i]
        total = float(joint.values[i][mask].sum()) if mask.any() else 0.0
        detected = int((joint.values[i][mask] > 0).sum())
        reasons: list[str] = []
        if total == 0:
            reasons.append("all_zero_counts")
        if total < params.min_total_counts:
            reasons.append(f"total_counts<{params.min_total_counts}")
        if detected < params.min_detected_genes:
            reasons.append(f"detected_genes<{params.min_detected_genes}")
        verdicts[key] = reasons
    return verdicts


@dataclass
class JoinModeComparison:
    """common 与 union 两种联合选择的对比：保留细胞与质量判断差异。"""

    common: JointMatrix
    union: JointMatrix
    common_verdicts: dict[str, list[str]]
    union_verdicts: dict[str, list[str]]

    @property
    def common_kept(self) -> list[str]:
        return [k for k, r in self.common_verdicts.items() if not r]

    @property
    def union_kept(self) -> list[str]:
        return [k for k, r in self.union_verdicts.items() if not r]

    @property
    def only_common_kept(self) -> list[str]:
        return sorted(set(self.common_kept) - set(self.union_kept))

    @property
    def only_union_kept(self) -> list[str]:
        return sorted(set(self.union_kept) - set(self.common_kept))

    def to_dict(self) -> dict:
        return {
            "common": {
                "n_genes": self.common.n_genes,
                "gene_ids": list(self.common.gene_ids),
                "kept": self.common_kept,
            },
            "union": {
                "n_genes": self.union.n_genes,
                "gene_ids": list(self.union.gene_ids),
                "kept": self.union_kept,
                "missing_genes_by_batch": self.union.missing_genes_by_batch(),
            },
            "only_common_kept": self.only_common_kept,
            "only_union_kept": self.only_union_kept,
            "verdict_diffs": [
                {
                    "cell_key": key,
                    "common_reasons": self.common_verdicts[key],
                    "union_reasons": self.union_verdicts[key],
                }
                for key in sorted(set(self.common_verdicts) & set(self.union_verdicts))
                if self.common_verdicts[key] != self.union_verdicts[key]
            ],
        }


def compare_join_modes(
    matrices: dict[str, CountMatrix], params_by_batch: dict[str, QCParams]
) -> JoinModeComparison:
    """同批来源分别按 common / union 联合，比较保留细胞与质量判断差异。"""
    common = join_batches(matrices, "common")
    union = join_batches(matrices, "union")
    return JoinModeComparison(
        common=common,
        union=union,
        common_verdicts=_qc_on_joint(common, params_by_batch),
        union_verdicts=_qc_on_joint(union, params_by_batch),
    )
