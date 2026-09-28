"""归一化：文库大小缩放 + log1p，参数可按批次区分。"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .model import CountMatrix


@dataclass(frozen=True)
class NormalizeParams:
    target_sum: float = 1e4
    log1p: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(
    matrix: CountMatrix, params_by_batch: dict[str, NormalizeParams]
) -> tuple[np.ndarray, dict[str, str]]:
    """逐细胞归一化，细胞所属批次决定使用哪套参数。

    返回 (归一化矩阵, cell_id -> 批次规则名 映射)，
    映射随结果保存，保证两个批次规则不混用且可审计。
    """
    missing = set(matrix.batches) - set(params_by_batch)
    if missing:
        raise ValueError(f"以下批次缺少归一化参数: {sorted(missing)}")

    out = np.zeros_like(matrix.counts, dtype=np.float64)
    rule_used: dict[str, str] = {}
    for i, cell_id in enumerate(matrix.cell_ids):
        batch = matrix.batch_of_cell[i]
        params = params_by_batch[batch]
        total = matrix.cell_total(i)
        if total == 0:
            raise ValueError(f"细胞 {cell_id} 总计数为 0，无法归一化（应在质控阶段剔除）")
        row = matrix.counts[i] / total * params.target_sum
        if params.log1p:
            row = np.log1p(row)
        out[i] = row
        rule_used[cell_id] = batch
    return out, rule_used
