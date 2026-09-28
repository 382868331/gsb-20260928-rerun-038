"""内置人工矩阵：数值已知，用于核对筛选数量与归一化结果。"""

from __future__ import annotations

import numpy as np

from .model import CountMatrix


def make_toy_matrix() -> CountMatrix:
    """构造 6 细胞 x 4 基因、两个批次的人工数据。

    批次 batchA（严格规则）: cell_a1, cell_a2, cell_a3(低计数), cell_a4(全零)
    批次 batchB（宽松规则）: cell_b1, cell_b2(低计数但按 B 规则可过)
    """
    counts = np.array(
        [
            [600, 300, 100, 0],
            [800, 100, 50, 50],
            [10, 5, 0, 0],
            [0, 0, 0, 0],
            [40, 30, 20, 10],
            [30, 20, 0, 0],
        ],
        dtype=np.int64,
    )
    return CountMatrix(
        counts=counts,
        cell_ids=("cell_a1", "cell_a2", "cell_a3", "cell_a4", "cell_b1", "cell_b2"),
        gene_ids=("GeneA", "GeneB", "GeneC", "GeneD"),
        batch_of_cell=("batchA", "batchA", "batchA", "batchA", "batchB", "batchB"),
    )
