"""计数矩阵数据模型：细胞、基因、批次均有稳定身份。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CountMatrix:
    """cells x genes 的原始计数矩阵。

    cell_ids / gene_ids 是稳定身份，筛选只取子集，不改写身份。
    重复基因、重复细胞等在构造时直接报错，绝不悄悄吞掉。
    """

    counts: np.ndarray
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    batch_of_cell: tuple[str, ...]

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts)
        if counts.ndim != 2:
            raise ValueError("counts 必须是二维矩阵")
        n_cells, n_genes = counts.shape
        if len(self.cell_ids) != n_cells:
            raise ValueError("cell_ids 数量与矩阵行数不一致")
        if len(self.gene_ids) != n_genes:
            raise ValueError("gene_ids 数量与矩阵列数不一致")
        if len(self.batch_of_cell) != n_cells:
            raise ValueError("batch_of_cell 数量与细胞数不一致")
        if len(set(self.cell_ids)) != n_cells:
            raise ValueError("存在重复细胞 ID，拒绝载入")
        if len(set(self.gene_ids)) != n_genes:
            dupes = sorted({g for g in self.gene_ids if self.gene_ids.count(g) > 1})
            raise ValueError(f"存在重复基因 ID，拒绝载入: {dupes}")
        if (counts < 0).any():
            raise ValueError("计数矩阵含负值")
        object.__setattr__(self, "counts", counts.astype(np.int64))

    @property
    def n_cells(self) -> int:
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    @property
    def batches(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.batch_of_cell)))

    def cell_total(self, i: int) -> int:
        return int(self.counts[i].sum())

    def subset_cells(self, keep_indices: list[int]) -> "CountMatrix":
        idx = np.asarray(keep_indices, dtype=int)
        return CountMatrix(
            counts=self.counts[idx],
            cell_ids=tuple(self.cell_ids[i] for i in idx),
            gene_ids=self.gene_ids,
            batch_of_cell=tuple(self.batch_of_cell[i] for i in idx),
        )

    @staticmethod
    def _digest(payload: object) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def identity_digests(self) -> dict[str, str]:
        """分别给出原始计数、细胞身份、基因身份、批次归属的摘要。

        摘要只取决于内容本身，与文件名、调用方给的标记无关：
        任何一部分被替换，对应摘要必然改变。
        """
        counts_payload = {
            "dtype": str(self.counts.dtype),
            "shape": list(self.counts.shape),
            "data": self.counts.tobytes().hex(),
        }
        return {
            "counts": self._digest(counts_payload),
            "cell_ids": self._digest(list(self.cell_ids)),
            "gene_ids": self._digest(list(self.gene_ids)),
            "batch_of_cell": self._digest(list(self.batch_of_cell)),
        }

    def content_fingerprint(self) -> str:
        """绑定原始矩阵内容与全部身份的整体指纹（平台自行计算）。"""
        return self._digest(self.identity_digests())
