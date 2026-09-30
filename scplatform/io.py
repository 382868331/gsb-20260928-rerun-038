"""从文件加载原始计数矩阵，使平台可独立于内置 toy 数据使用。

矩阵 CSV：首行表头为基因 ID，首列为细胞 ID；批次 CSV 两列：cell_id,batch。
指纹只与文件内容绑定，与文件名无关——同名文件内容改变即视为新输入。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .model import CountMatrix


def load_matrix(counts_csv: str | Path, batches_csv: str | Path | None = None) -> CountMatrix:
    """加载原始计数矩阵；未提供批次表时所有细胞归入 default 批次。"""
    counts_path = Path(counts_csv)
    with counts_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"计数文件为空: {counts_path}")

    header = rows[0]
    if len(header) < 2:
        raise ValueError("计数文件首行必须包含细胞 ID 列名与至少一个基因列")
    gene_ids = tuple(header[1:])
    cell_ids: list[str] = []
    data: list[list[int]] = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        if len(row) != len(header):
            raise ValueError(f"细胞 {row[0]!r} 的列数与表头不一致")
        cell_ids.append(row[0])
        data.append([int(value) for value in row[1:]])

    counts = np.asarray(data, dtype=np.int64) if data else np.zeros((0, len(gene_ids)), dtype=np.int64)
    if counts.size == 0 or counts.shape[0] == 0:
        raise ValueError("计数文件不含任何细胞行（全空批次请保留至少一个细胞行）")

    if batches_csv is None:
        batch_of_cell = tuple("default" for _ in cell_ids)
    else:
        batch_map: dict[str, str] = {}
        with Path(batches_csv).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                batch_map[row["cell_id"]] = row["batch"]
        missing = [cid for cid in cell_ids if cid not in batch_map]
        if missing:
            raise ValueError(f"以下细胞缺少批次归属: {missing}")
        extra = sorted(set(batch_map) - set(cell_ids))
        if extra:
            raise ValueError(f"批次表含矩阵中不存在的细胞: {extra}")
        batch_of_cell = tuple(batch_map[cid] for cid in cell_ids)

    return CountMatrix(
        counts=counts,
        cell_ids=tuple(cell_ids),
        gene_ids=gene_ids,
        batch_of_cell=batch_of_cell,
    )
