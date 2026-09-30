"""矩阵文件载入：来源文件名随矩阵一起记录，供结果追溯。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import CountMatrix


def load_matrix_csv(path: str | Path) -> CountMatrix:
    """读取细胞 x 基因计数表。

    格式（带表头）::

        cell_id,batch,GeneA,GeneB,...
        cell_a1,batchA,600,300,...

    只依赖标准库，保证示例数据可以脱离第三方包被检查。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 3:
            raise ValueError(f"{path} 缺少表头，需包含 cell_id,batch 与至少一个基因列")
        gene_ids = tuple(g.strip() for g in header[2:])
        cell_ids: list[str] = []
        batches: list[str] = []
        rows: list[list[int]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row or all(cell == "" for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"{path}:{line_no} 列数与表头不一致")
            cell_ids.append(row[0].strip())
            batches.append(row[1].strip())
            try:
                rows.append([int(value) for value in row[2:]])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} 含非整数计数: {exc}") from exc

    if not rows:
        raise ValueError(f"{path} 没有任何细胞行（全空批次输入不允许静默通过）")

    counts = np.asarray(rows, dtype=np.int64)
    return CountMatrix(
        counts=counts,
        cell_ids=tuple(cell_ids),
        gene_ids=gene_ids,
        batch_of_cell=tuple(batches),
    )


def write_matrix_csv(path: str | Path, matrix: CountMatrix) -> None:
    """把矩阵写成与 :func:`load_matrix_csv` 对称的 CSV，便于 demo 造数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "batch", *matrix.gene_ids])
        for i, cell_id in enumerate(matrix.cell_ids):
            writer.writerow([cell_id, matrix.batch_of_cell[i], *map(int, matrix.counts[i])])
