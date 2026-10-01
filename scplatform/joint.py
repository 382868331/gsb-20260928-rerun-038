"""联合分析集：把基因集合可重叠但不完全相同的多个采集批次整理成一个联合矩阵。

核心约定：

- 缺测（某批次没有测量某基因）在 values 中以 NaN 表示、observed 掩码为
  False，绝不把缺测解释成零表达；零表达必须是真实测量到的 0。
- 细胞显示名称相同也按来源区分：联合键为 "<batch>::<cell_id>"，
  不同批次的同名细胞绝不合并。
- 支持两种模式：intersection 只使用所有批次共同测量的基因；
  union 保留全部出现过的基因并明确记录每个批次的缺测范围。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .model import CountMatrix
from .qc import QCParams

JOIN_MODES = ("intersection", "union")


@dataclass(frozen=True)
class JointMatrix:
    """联合矩阵：values 中 NaN 表示该批次未测量，observed 标记真实测量。

    cell_keys 是来源区分的稳定身份（"<batch>::<cell_id>"），display_names
    是各批次内部的原始显示名（可跨批次重复，绝不据此合并）。
    """

    values: np.ndarray
    observed: np.ndarray
    cell_keys: tuple[str, ...]
    display_names: tuple[str, ...]
    gene_ids: tuple[str, ...]
    batch_of_cell: tuple[str, ...]
    mode: str
    source_fingerprints: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        observed = np.asarray(self.observed, dtype=bool)
        if values.ndim != 2 or observed.shape != values.shape:
            raise ValueError("values 与 observed 必须是同形状的二维矩阵")
        n_cells, n_genes = values.shape
        if len(self.cell_keys) != n_cells or len(self.display_names) != n_cells:
            raise ValueError("细胞身份数量与矩阵行数不一致")
        if len(self.batch_of_cell) != n_cells:
            raise ValueError("批次归属数量与细胞数不一致")
        if len(self.gene_ids) != n_genes:
            raise ValueError("基因身份数量与矩阵列数不一致")
        if len(set(self.cell_keys)) != n_cells:
            raise ValueError("存在重复联合细胞键，拒绝构造")
        if len(set(self.gene_ids)) != n_genes:
            raise ValueError("存在重复基因 ID，拒绝构造")
        if self.mode not in JOIN_MODES:
            raise ValueError(f"未知联合模式: {self.mode!r}")
        # 缺测必须同时是 NaN 且 observed=False；观测值不得是 NaN。
        if np.isnan(values[observed]).any():
            raise ValueError("observed=True 的位置不允许出现 NaN")
        if not np.isnan(values[~observed]).all():
            raise ValueError("缺测位置必须以 NaN 表示，不能用 0 或其他数值冒充")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(
            self, "source_fingerprints", tuple(tuple(p) for p in self.source_fingerprints)
        )

    @property
    def n_cells(self) -> int:
        return self.values.shape[0]

    @property
    def n_genes(self) -> int:
        return self.values.shape[1]

    @property
    def batches(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.batch_of_cell)))

    def missing_by_batch(self) -> dict[str, list[str]]:
        """每个批次缺测的基因清单（该批次所有细胞都未观测到的基因）。"""
        result: dict[str, list[str]] = {}
        for batch in self.batches:
            rows = [i for i, b in enumerate(self.batch_of_cell) if b == batch]
            missing = [
                self.gene_ids[j]
                for j in range(self.n_genes)
                if not self.observed[rows, j].any()
            ]
            result[batch] = missing
        return result

    def content_fingerprint(self) -> str:
        """绑定联合内容（值、缺测掩码、身份、模式、来源指纹）的整体指纹。"""
        payload = {
            "mode": self.mode,
            "values": np.where(self.observed, self.values, np.nan).tobytes().hex(),
            "observed": self.observed.tobytes().hex(),
            "cell_keys": list(self.cell_keys),
            "display_names": list(self.display_names),
            "gene_ids": list(self.gene_ids),
            "batch_of_cell": list(self.batch_of_cell),
            "source_fingerprints": [list(p) for p in self.source_fingerprints],
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def join_batches(batches: dict[str, CountMatrix], mode: str = "union") -> JointMatrix:
    """把若干批次矩阵整理成一个联合分析集。

    mode="intersection" 只保留所有批次共同测量的基因；
    mode="union" 保留全部出现过的基因，缺测位置为 NaN + observed=False。
    任一批次构造失败（重复身份等）直接抛错，不产出混杂的联合结果。
    """
    if mode not in JOIN_MODES:
        raise ValueError(f"未知联合模式: {mode!r}，可选 {JOIN_MODES}")
    if not batches:
        raise ValueError("至少需要一个批次才能联合")
    for name in batches:
        if not name or "::" in name:
            raise ValueError(f"非法批次名: {name!r}（不能为空且不得包含 '::'）")

    names = sorted(batches)
    gene_sets = [set(batches[n].gene_ids) for n in names]
    if mode == "intersection":
        common = set.intersection(*gene_sets)
        # 共同基因按第一个批次（名字典序最小）的原始顺序排列
        gene_ids = tuple(g for g in batches[names[0]].gene_ids if g in common)
        if not gene_ids:
            raise ValueError("各批次没有共同测量的基因，无法使用 intersection 模式")
    else:
        seen: dict[str, None] = {}
        for n in names:
            for g in batches[n].gene_ids:
                seen.setdefault(g)
        gene_ids = tuple(seen)

    cell_keys: list[str] = []
    display_names: list[str] = []
    batch_of_cell: list[str] = []
    rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    gene_col = {g: j for j, g in enumerate(gene_ids)}

    for name in names:
        matrix = batches[name]
        for i, cell_id in enumerate(matrix.cell_ids):
            cell_keys.append(f"{name}::{cell_id}")
            display_names.append(cell_id)
            batch_of_cell.append(name)
            row = np.full(len(gene_ids), np.nan, dtype=np.float64)
            obs = np.zeros(len(gene_ids), dtype=bool)
            for j, gene in enumerate(matrix.gene_ids):
                col = gene_col.get(gene)
                if col is None:
                    continue  # intersection 模式下该基因不在共同集合中
                row[col] = float(matrix.counts[i, j])
                obs[col] = True
            rows.append(row)
            obs_rows.append(obs)

    fingerprints = tuple(
        (n, batches[n].content_fingerprint()) for n in names
    )
    return JointMatrix(
        values=np.vstack(rows) if rows else np.zeros((0, len(gene_ids))),
        observed=np.vstack(obs_rows) if obs_rows else np.zeros((0, len(gene_ids)), dtype=bool),
        cell_keys=tuple(cell_keys),
        display_names=tuple(display_names),
        gene_ids=gene_ids,
        batch_of_cell=tuple(batch_of_cell),
        mode=mode,
        source_fingerprints=fingerprints,
    )


@dataclass
class JointCellVerdict:
    cell_key: str
    display_name: str
    batch: str
    kept: bool
    reasons: list[str] = field(default_factory=list)
    n_missing_genes: int = 0


@dataclass
class JointQCReport:
    """联合矩阵上的质量判断：只统计真实测量值，缺测不参与计数。""" 

    mode: str
    verdicts: list[JointCellVerdict]

    @property
    def kept_ids(self) -> list[str]:
        return [v.cell_key for v in self.verdicts if v.kept]

    @property
    def dropped_ids(self) -> list[str]:
        return [v.cell_key for v in self.verdicts if not v.kept]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "kept": self.kept_ids,
            "dropped": [
                {
                    "cell_key": v.cell_key,
                    "display_name": v.display_name,
                    "batch": v.batch,
                    "reasons": list(v.reasons),
                    "n_missing_genes": v.n_missing_genes,
                }
                for v in self.verdicts
                if not v.kept
            ],
        }


def qc_on_joint(joint: JointMatrix, params_by_batch: dict[str, QCParams]) -> JointQCReport:
    """在联合矩阵上按批次规则做质量判断。

    总计数与检出基因数只统计 observed=True 的位置；缺测基因既不当作
    零表达计入检出，也不计入总计数。
    """
    missing = set(joint.batches) - set(params_by_batch)
    if missing:
        raise ValueError(f"以下批次缺少筛选规则: {sorted(missing)}")

    verdicts: list[JointCellVerdict] = []
    for i, key in enumerate(joint.cell_keys):
        batch = joint.batch_of_cell[i]
        params = params_by_batch[batch]
        obs = joint.observed[i]
        measured = joint.values[i][obs]
        total = float(measured.sum()) if measured.size else 0.0
        detected = int((measured > 0).sum())
        n_missing = int((~obs).sum())

        reasons: list[str] = []
        if total == 0:
            reasons.append("all_zero_counts")
        if total < params.min_total_counts:
            reasons.append(f"total_counts<{params.min_total_counts}")
        if detected < params.min_detected_genes:
            reasons.append(f"detected_genes<{params.min_detected_genes}")

        verdicts.append(
            JointCellVerdict(
                cell_key=key,
                display_name=joint.display_names[i],
                batch=batch,
                kept=not reasons,
                reasons=reasons,
                n_missing_genes=n_missing,
            )
        )
    return JointQCReport(mode=joint.mode, verdicts=verdicts)


@dataclass
class JoinModeComparison:
    """intersection 与 union 两种联合选择的并行比较。

    细胞以来源区分的联合键标识；entered/left 表示从 intersection 换到
    union 时保留集合的变化，missing_by_batch 记录 union 模式下每个批次
    的缺测范围。
    """

    intersection_report: JointQCReport
    union_report: JointQCReport
    missing_by_batch: dict[str, list[str]]
    n_genes_intersection: int
    n_genes_union: int

    @property
    def entered_ids(self) -> list[str]:
        """union 保留但 intersection 剔除的细胞。"""
        inter = set(self.intersection_report.kept_ids)
        return [k for k in self.union_report.kept_ids if k not in inter]

    @property
    def left_ids(self) -> list[str]:
        """intersection 保留但 union 剔除的细胞。"""
        union = set(self.union_report.kept_ids)
        return [k for k in self.intersection_report.kept_ids if k not in union]

    @property
    def intersection_ids(self) -> list[str]:
        """两种选择都保留的细胞。"""
        union = set(self.union_report.kept_ids)
        return [k for k in self.intersection_report.kept_ids if k in union]

    def to_dict(self) -> dict:
        return {
            "n_genes": {
                "intersection": self.n_genes_intersection,
                "union": self.n_genes_union,
            },
            "kept": {
                "intersection": self.intersection_report.kept_ids,
                "union": self.union_report.kept_ids,
            },
            "both_kept": self.intersection_ids,
            "only_union_kept": self.entered_ids,
            "only_intersection_kept": self.left_ids,
            "missing_by_batch": {k: list(v) for k, v in self.missing_by_batch.items()},
            "intersection_qc": self.intersection_report.to_dict(),
            "union_qc": self.union_report.to_dict(),
        }


def compare_join_modes(
    batches: dict[str, CountMatrix],
    params_by_batch: dict[str, QCParams],
) -> JoinModeComparison:
    """并行比较两种联合选择对保留细胞与质量判断的影响。"""
    joint_inter = join_batches(batches, mode="intersection")
    joint_union = join_batches(batches, mode="union")
    return JoinModeComparison(
        intersection_report=qc_on_joint(joint_inter, params_by_batch),
        union_report=qc_on_joint(joint_union, params_by_batch),
        missing_by_batch=joint_union.missing_by_batch(),
        n_genes_intersection=joint_inter.n_genes,
        n_genes_union=joint_union.n_genes,
    )
