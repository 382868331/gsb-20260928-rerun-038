"""平台行为核对：筛选数量、归一化数值、批次规则隔离、版本不覆盖。"""

import json

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    NormalizeParams,
    QCParams,
    ResultStore,
    filter_cells,
    normalize,
)
from scplatform.toydata import make_toy_matrix

QC = {
    "batchA": QCParams(min_total_counts=500, min_detected_genes=3),
    "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
}
NORM = {
    "batchA": NormalizeParams(target_sum=1e4, log1p=True),
    "batchB": NormalizeParams(target_sum=1e3, log1p=True),
}


def test_qc_keeps_expected_cells():
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    assert report.kept_ids == ["cell_a1", "cell_a2", "cell_b1", "cell_b2"]
    assert report.dropped_ids == ["cell_a3", "cell_a4"]
    assert report.all_zero_cells == ["cell_a4"]
    assert report.empty_batches == []
    # 稳定身份：筛选后细胞 ID 是原始 ID 的子集，顺序保持
    assert filtered.cell_ids == ("cell_a1", "cell_a2", "cell_b1", "cell_b2")
    assert filtered.gene_ids == matrix.gene_ids


def test_empty_batch_is_reported_not_swallowed():
    matrix = make_toy_matrix()
    qc = dict(QC, batchB=QCParams(min_total_counts=10**9, min_detected_genes=2))
    _, report = filter_cells(matrix, qc)
    assert report.empty_batches == ["batchB"]


def test_duplicate_genes_rejected():
    with pytest.raises(ValueError, match="重复基因"):
        CountMatrix(
            counts=np.ones((1, 2), dtype=int),
            cell_ids=("c1",),
            gene_ids=("G", "G"),
            batch_of_cell=("b",),
        )


def test_normalization_values_exact():
    matrix = make_toy_matrix()
    filtered, _ = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)

    # cell_a1: [600,300,100,0]/1000*1e4 -> log1p([6000,3000,1000,0])
    expected_a1 = np.log1p([6000.0, 3000.0, 1000.0, 0.0])
    np.testing.assert_allclose(normed[0], expected_a1)
    # cell_b1 用 batchB 的 target_sum=1e3: [40,30,20,10]/100*1e3
    expected_b1 = np.log1p([400.0, 300.0, 200.0, 100.0])
    np.testing.assert_allclose(normed[2], expected_b1)
    # 批次规则不混用
    assert rule_used == {
        "cell_a1": "batchA",
        "cell_a2": "batchA",
        "cell_b1": "batchB",
        "cell_b2": "batchB",
    }


def test_versions_never_overwrite(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)

    common = dict(
        qc_report=report.to_dict(),
        normalized=normed,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
        input_fingerprint="toy",
    )
    d1 = store.save_run(params={"qc": "strict"}, **common)
    d2 = store.save_run(params={"qc": "relaxed"}, **common)
    assert d1 != d2 and d1.exists() and d2.exists()
    # 旧版本内容未被改动
    saved = json.loads((d1 / "params.json").read_text(encoding="utf-8"))
    assert saved == {"qc": "strict"}
    assert len(store.list_versions()) == 2
