"""平台行为核对：筛选、归一化、批次隔离、并行比较与版本不可变。"""

import json

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    ImmutableResultError,
    NormalizeParams,
    QCParams,
    ResultStore,
    compare_filters,
    filter_cells,
    load_matrix_csv,
    normalize,
    write_matrix_csv,
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


def _save(store, matrix, *, params=None, qc=None, norm=None):
    qc = qc or QC
    norm = norm or NORM
    filtered, report = filter_cells(matrix, qc)
    normed, rule_used = normalize(filtered, norm)
    return store.save_run(
        raw_matrix=matrix,
        source_name="toy_matrix.csv",
        params=params or {"qc": "strict"},
        qc_report=report.to_dict(),
        normalized=normed,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
    )


def test_qc_keeps_expected_cells():
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    assert report.kept_ids == ["cell_a1", "cell_a2", "cell_b1", "cell_b2"]
    assert report.dropped_ids == ["cell_a3", "cell_a4"]
    assert report.all_zero_cells == ["cell_a4"]
    assert report.empty_batches == []
    assert filtered.cell_ids == ("cell_a1", "cell_a2", "cell_b1", "cell_b2")
    assert filtered.gene_ids == matrix.gene_ids
    verdict_a3 = next(v for v in report.verdicts if v.cell_id == "cell_a3")
    assert verdict_a3.total_counts == 15
    assert verdict_a3.detected_genes == 2


def test_all_empty_batch_is_reported_not_swallowed():
    """批次所有细胞都被滤掉（含全零批次）时必须显式报出。"""
    matrix = make_toy_matrix()
    qc = dict(QC, batchB=QCParams(min_total_counts=10**9, min_detected_genes=2))
    filtered, report = filter_cells(matrix, qc)
    assert report.empty_batches == ["batchB"]
    assert "cell_b1" in report.dropped_ids and "cell_b2" in report.dropped_ids
    assert filtered.batches == ("batchA",)


def test_batch_containing_only_zero_cells_is_empty():
    """真正的全空批次：整批细胞总计数为 0。"""
    matrix = CountMatrix(
        counts=np.array([[0, 0], [10, 20]], dtype=np.int64),
        cell_ids=("c1", "c2"),
        gene_ids=("G1", "G2"),
        batch_of_cell=("emptyBatch", "liveBatch"),
    )
    _, report = filter_cells(
        matrix,
        {
            "emptyBatch": QCParams(min_total_counts=1, min_detected_genes=1),
            "liveBatch": QCParams(min_total_counts=1, min_detected_genes=1),
        },
    )
    assert report.all_zero_cells == ["c1"]
    assert report.empty_batches == ["emptyBatch"]


def test_empty_csv_file_is_rejected(tmp_path):
    path = tmp_path / "matrix.csv"
    path.write_text("cell_id,batch,G1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="没有任何细胞行"):
        load_matrix_csv(path)


def test_duplicate_gene_and_cell_identities_rejected():
    with pytest.raises(ValueError, match="重复基因"):
        CountMatrix(
            counts=np.ones((1, 2), dtype=int),
            cell_ids=("c1",),
            gene_ids=("G", "G"),
            batch_of_cell=("b",),
        )
    with pytest.raises(ValueError, match="重复细胞"):
        CountMatrix(
            counts=np.ones((2, 1), dtype=int),
            cell_ids=("c1", "c1"),
            gene_ids=("G",),
            batch_of_cell=("b", "b"),
        )


def test_csv_round_trip_preserves_identities(tmp_path):
    matrix = make_toy_matrix()
    path = tmp_path / "toy_matrix.csv"
    write_matrix_csv(path, matrix)
    loaded = load_matrix_csv(path)
    assert loaded.cell_ids == matrix.cell_ids
    assert loaded.gene_ids == matrix.gene_ids
    assert loaded.batch_of_cell == matrix.batch_of_cell
    np.testing.assert_array_equal(loaded.counts, matrix.counts)
    assert loaded.fingerprint() == matrix.fingerprint()


def test_compare_filters_shows_entered_and_left_cells_with_reasons():
    matrix = make_toy_matrix()
    strict = dict(QC)
    relaxed = dict(QC, batchA=QCParams(min_total_counts=10, min_detected_genes=2))

    comparison = compare_filters(
        matrix, strict, relaxed, label_before="strict", label_after="relaxed"
    )
    entered = {c.cell_id: c for c in comparison.entered}
    assert set(entered) == {"cell_a3"}
    assert entered["cell_a3"].reasons_before == [
        "total_counts<500",
        "detected_genes<3",
    ]
    assert comparison.left == []

    stricter = dict(QC, batchB=QCParams(min_total_counts=200, min_detected_genes=2))
    reverse = compare_filters(matrix, strict, stricter)
    left = {c.cell_id: c for c in reverse.left}
    assert set(left) == {"cell_b1", "cell_b2"}
    assert left["cell_b1"].reasons_after == ["total_counts<200"]
    assert reverse.empty_batches_after == ["batchB"]
    payload = reverse.to_dict()
    assert {"entered", "left", "all_cells"} <= payload.keys()


def test_normalization_values_exact():
    matrix = make_toy_matrix()
    filtered, _ = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    expected_a1 = np.log1p([6000.0, 3000.0, 1000.0, 0.0])
    np.testing.assert_allclose(normed[0], expected_a1)
    expected_b1 = np.log1p([400.0, 300.0, 200.0, 100.0])
    np.testing.assert_allclose(normed[2], expected_b1)
    assert rule_used == {
        "cell_a1": "batchA",
        "cell_a2": "batchA",
        "cell_b1": "batchB",
        "cell_b2": "batchB",
    }


def test_fingerprint_changes_with_content_but_not_markers_only():
    matrix = make_toy_matrix()
    changed = CountMatrix(
        counts=matrix.counts.copy(),
        cell_ids=matrix.cell_ids,
        gene_ids=matrix.gene_ids,
        batch_of_cell=matrix.batch_of_cell,
    )
    assert changed.fingerprint() == matrix.fingerprint()
    changed.counts[0, 0] += 1
    assert changed.fingerprint() != matrix.fingerprint()
    renamed = CountMatrix(
        counts=matrix.counts,
        cell_ids=tuple("x_" + c for c in matrix.cell_ids),
        gene_ids=matrix.gene_ids,
        batch_of_cell=matrix.batch_of_cell,
    )
    assert renamed.fingerprint() != matrix.fingerprint()


def test_same_file_name_changed_content_creates_distinct_version(tmp_path):
    """文件名保持不变、矩阵内容被改过 -> 必须是不同版本。"""
    store = ResultStore(tmp_path)
    source = tmp_path / "toy_matrix.csv"

    write_matrix_csv(source, make_toy_matrix())
    matrix_v1 = load_matrix_csv(source)
    dir1 = _save(store, matrix_v1, params={"qc": "strict"})

    altered_counts = matrix_v1.counts.copy()
    altered_counts[0, 0] += 1
    matrix_v2 = CountMatrix(
        counts=altered_counts,
        cell_ids=matrix_v1.cell_ids,
        gene_ids=matrix_v1.gene_ids,
        batch_of_cell=matrix_v1.batch_of_cell,
    )
    write_matrix_csv(source, matrix_v2)
    assert load_matrix_csv(source).fingerprint() == matrix_v2.fingerprint()

    dir2 = _save(store, matrix_v2, params={"qc": "strict"})
    assert dir1 != dir2
    assert len(store.list_versions()) == 2
    identity1 = json.loads((dir1 / "identity.json").read_text(encoding="utf-8"))
    identity2 = json.loads((dir2 / "identity.json").read_text(encoding="utf-8"))
    assert identity1["source_name"] == identity2["source_name"] == "toy_matrix.csv"
    assert identity1["input_fingerprint"] != identity2["input_fingerprint"]
    assert identity1["version_hash"] != identity2["version_hash"]


def test_identical_run_is_idempotent_not_duplicated(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    dir1 = _save(store, matrix, params={"qc": "strict"})
    dir2 = _save(store, matrix, params={"qc": "strict"})
    assert dir1 == dir2
    assert len(store.list_versions()) == 1


def test_parameter_change_creates_distinct_version(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    dir1 = _save(store, matrix, params={"qc": "strict"})
    dir2 = _save(store, matrix, params={"qc": "relaxed"})
    assert dir1 != dir2
    saved = json.loads((dir1 / "params.json").read_text(encoding="utf-8"))
    assert saved == {"qc": "strict"}


def test_published_version_cannot_be_overwritten(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    version_dir = _save(store, matrix, params={"qc": "strict"})
    original_params = (version_dir / "params.json").read_text(encoding="utf-8")
    store.publish(version_dir)
    assert store.is_published(version_dir)

    # 后来的试验试图向已发布版本追加/覆盖产物，一律拒绝
    with pytest.raises(ImmutableResultError, match="已发布"):
        store._atomic_write(version_dir, "later_experiment.json", "{}")
    with pytest.raises(ImmutableResultError, match="结果文件已存在"):
        store._atomic_write(version_dir, "params.json", "{}")
    with pytest.raises(ImmutableResultError):
        store._atomic_write(version_dir, "published.json", '{"hacked": true}')
    assert (version_dir / "params.json").read_text(encoding="utf-8") == original_params
    # 重复发布幂等，不产生第二个标记
    again = store.publish(version_dir)
    assert again == version_dir


def test_caller_cannot_supply_fingerprint(tmp_path):
    """save_run 不再接受调用方指纹，参数传错直接 TypeError。"""
    store = ResultStore(tmp_path)
    with pytest.raises(TypeError):
        store.save_run(input_fingerprint="whatever")  # type: ignore[call-arg]


def test_saving_foreign_identity_is_rejected(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    with pytest.raises(ValueError, match="不属于原始矩阵"):
        store.save_run(
            raw_matrix=matrix,
            params={"qc": "strict"},
            qc_report=report.to_dict(),
            normalized=normed,
            cell_ids=["ghost_cell", *filtered.cell_ids[1:]],
            gene_ids=list(filtered.gene_ids),
            batch_of_cell=list(filtered.batch_of_cell),
            rule_used=rule_used,
        )


def test_raw_matrix_is_archived_with_version(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    version_dir = _save(store, matrix)
    archived = np.load(version_dir / "raw.npz")
    np.testing.assert_array_equal(archived["raw_counts"], matrix.counts)
