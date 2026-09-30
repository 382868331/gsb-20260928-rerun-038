"""平台行为核对：筛选数量、归一化数值、批次规则隔离、版本不覆盖。"""

import json

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    NormalizeParams,
    PublishRejectedError,
    QCParams,
    ResultStore,
    compare_qc,
    filter_cells,
    load_matrix,
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


def test_duplicate_cells_rejected():
    with pytest.raises(ValueError, match="重复细胞"):
        CountMatrix(
            counts=np.ones((2, 1), dtype=int),
            cell_ids=("c1", "c1"),
            gene_ids=("G",),
            batch_of_cell=("b", "b"),
        )


def test_all_zero_batch_is_flagged(tmp_path):
    """整个批次全部零计数：不崩溃、不悄悄消失，批次被标记为空。"""
    counts_csv = tmp_path / "counts.csv"
    batches_csv = tmp_path / "batches.csv"
    counts_csv.write_text("cell,G1,G2\nc1,5,5\nc2,0,0\n", encoding="utf-8")
    batches_csv.write_text("cell_id,batch\nc1,batchGood\nc2,batchEmpty\n", encoding="utf-8")
    matrix = load_matrix(counts_csv, batches_csv)
    assert matrix.batches == ("batchEmpty", "batchGood")
    _, report = filter_cells(
        matrix,
        {
            "batchGood": QCParams(min_total_counts=1, min_detected_genes=1),
            "batchEmpty": QCParams(min_total_counts=1, min_detected_genes=1),
        },
    )
    assert report.all_zero_cells == ["c2"]
    assert report.empty_batches == ["batchEmpty"]


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
        source_matrix=matrix,
        qc_report=report.to_dict(),
        normalized=normed,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
    )
    d1 = store.save_run(params={"qc": "strict"}, **common)
    d2 = store.save_run(params={"qc": "relaxed"}, **common)
    assert d1 != d2 and d1.exists() and d2.exists()
    # 旧版本内容未被改动
    saved = json.loads((d1 / "params.json").read_text(encoding="utf-8"))
    assert saved == {"qc": "strict"}
    assert len(store.list_versions()) == 2


def _save_kwargs(matrix, filtered, report, normed, rule_used):
    return dict(
        source_matrix=matrix,
        qc_report=report.to_dict(),
        normalized=normed,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
    )


def test_fingerprint_covers_content_and_identities():
    matrix = make_toy_matrix()
    digests = matrix.identity_digests()
    # 同样的行、不同的基因名 -> 基因身份摘要不同，整体指纹不同
    renamed = CountMatrix(
        counts=matrix.counts.copy(),
        cell_ids=matrix.cell_ids,
        gene_ids=tuple(f"X{i}" for i in range(matrix.n_genes)),
        batch_of_cell=matrix.batch_of_cell,
    )
    assert renamed.identity_digests()["gene_ids"] != digests["gene_ids"]
    assert renamed.content_fingerprint() != matrix.content_fingerprint()
    # 只改一个计数值 -> counts 摘要改变
    changed_counts = matrix.counts.copy()
    changed_counts[0, 0] += 1
    changed = CountMatrix(
        counts=changed_counts,
        cell_ids=matrix.cell_ids,
        gene_ids=matrix.gene_ids,
        batch_of_cell=matrix.batch_of_cell,
    )
    assert changed.identity_digests()["counts"] != digests["counts"]
    # 相同内容独立构造 -> 指纹一致（与文件名、标记无关）
    rebuilt = CountMatrix(
        counts=matrix.counts.copy(),
        cell_ids=tuple(matrix.cell_ids),
        gene_ids=tuple(matrix.gene_ids),
        batch_of_cell=tuple(matrix.batch_of_cell),
    )
    assert rebuilt.content_fingerprint() == matrix.content_fingerprint()


def test_changed_content_same_label_gets_new_version(tmp_path):
    """文件名/run_name 不变，仅内容改变：必须隔离成新版本，且互不覆盖。"""
    store = ResultStore(tmp_path)
    params = {"qc": {b: q.to_dict() for b, q in QC.items()},
              "normalize": {b: n.to_dict() for b, n in NORM.items()}}

    matrix1 = make_toy_matrix()
    f1, r1 = filter_cells(matrix1, QC)
    n1, rule1 = normalize(f1, NORM)
    d1 = store.save_run(params=params, **_save_kwargs(matrix1, f1, r1, n1, rule1))

    # 内容改变（一个计数变化），沿用完全相同的参数与外部标记
    changed_counts = matrix1.counts.copy()
    changed_counts[2, 0] = 200
    matrix2 = CountMatrix(
        counts=changed_counts,
        cell_ids=matrix1.cell_ids,
        gene_ids=matrix1.gene_ids,
        batch_of_cell=matrix1.batch_of_cell,
    )
    f2, r2 = filter_cells(matrix2, QC)
    n2, rule2 = normalize(f2, NORM)
    d2 = store.save_run(params=params, **_save_kwargs(matrix2, f2, r2, n2, rule2))

    assert d1 != d2
    identity1 = json.loads((d1 / "input_identity.json").read_text(encoding="utf-8"))
    identity2 = json.loads((d2 / "input_identity.json").read_text(encoding="utf-8"))
    assert identity1["input_fingerprint"] != identity2["input_fingerprint"]
    assert identity1["digests"]["counts"] != identity2["digests"]["counts"]
    assert len(store.list_versions()) == 2


def test_published_version_is_idempotent_and_readonly(tmp_path):
    store = ResultStore(tmp_path)
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    params = {"qc": "strict"}
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)

    d1 = store.save_run(params=params, **kwargs)
    d2 = store.save_run(params=params, **kwargs)
    assert d1 == d2
    assert len(store.list_versions()) == 1
    # 已发布文件只读，后来的试验无法覆盖
    with pytest.raises(PermissionError):
        (d1 / "params.json").write_text('{"qc": "tampered"}', encoding="utf-8")


def test_compare_qc_shows_entered_left_and_reasons():
    matrix = make_toy_matrix()
    relaxed = dict(QC, batchA=QCParams(min_total_counts=10, min_detected_genes=1))
    comparison = compare_qc(matrix, QC, relaxed, baseline_label="strict",
                            candidate_label="relaxed")
    # cell_a3(total=15) 在严格规则下被剔除、宽松规则下进入；全零 cell_a4 永不进入
    assert comparison.entered_ids == ["cell_a3"]
    assert comparison.left_ids == []
    change_a3 = next(c for c in comparison.changes if c.cell_id == "cell_a3")
    assert not change_a3.before_kept and change_a3.after_kept
    assert any("total_counts<500" in r for r in change_a3.before_reasons)
    assert change_a3.after_reasons == []
    a4 = next(c for c in comparison.changes if c.cell_id == "cell_a4")
    assert a4.before_kept is False and a4.after_kept is False
    assert "all_zero_counts" in a4.after_reasons

    # 反向比较：细胞离开并带原因
    reverse = compare_qc(matrix, relaxed, QC, baseline_label="relaxed",
                         candidate_label="strict")
    assert reverse.left_ids == ["cell_a3"]
    assert reverse.entered_ids == []
    payload = comparison.to_dict()
    assert payload["entered"] == ["cell_a3"]
    cell_payload = next(c for c in payload["cells"] if c["cell_id"] == "cell_a3")
    assert cell_payload["change"] == "entered"


def test_compare_flags_become_empty_batch():
    matrix = make_toy_matrix()
    kill_b = dict(QC, batchB=QCParams(min_total_counts=10**9, min_detected_genes=2))
    comparison = compare_qc(matrix, QC, kill_b, baseline_label="normal",
                            candidate_label="kill-b")
    assert comparison.to_dict()["empty_batches"] == {
        "baseline": [],
        "candidate": ["batchB"],
    }
    assert set(comparison.left_ids) == {"cell_b1", "cell_b2"}


def test_load_matrix_rejects_duplicate_identity(tmp_path):
    counts_csv = tmp_path / "counts.csv"
    counts_csv.write_text("cell,G1\nc1,1\nc1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复细胞"):
        load_matrix(counts_csv)


def test_load_matrix_and_pipeline_roundtrip(tmp_path):
    counts_csv = tmp_path / "counts.csv"
    batches_csv = tmp_path / "batches.csv"
    counts_csv.write_text(
        "cell,G1,G2,G3\nc1,600,300,100\nc2,40,30,20\n", encoding="utf-8"
    )
    batches_csv.write_text("cell_id,batch\nc1,batchA\nc2,batchB\n", encoding="utf-8")
    matrix = load_matrix(counts_csv, batches_csv)
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    store = ResultStore(tmp_path / "results")
    params = {"qc": {b: q.to_dict() for b, q in QC.items()}}
    out = store.save_run(
        params=params, **_save_kwargs(matrix, filtered, report, normed, rule_used)
    )
    identity = json.loads((out / "input_identity.json").read_text(encoding="utf-8"))
    assert identity["cell_ids"] == ["c1", "c2"]
    assert identity["gene_ids"] == ["G1", "G2", "G3"]
    assert store.find_version(matrix, params) == out


def _pipeline_parts():
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    return matrix, filtered, report, normed, rule_used


def test_save_rejects_unknown_cell_and_leaves_no_partial_version(tmp_path):
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    kwargs["cell_ids"] = list(filtered.cell_ids) + ["ghost_cell"]
    kwargs["batch_of_cell"] = list(filtered.batch_of_cell) + ["batchA"]
    kwargs["normalized"] = np.vstack([normed, np.zeros((1, normed.shape[1]))])
    with pytest.raises(PublishRejectedError, match="不存在的细胞身份") as exc:
        store.save_run(params={"qc": "x"}, **kwargs)
    assert any("ghost_cell" in r for r in exc.value.reasons)
    # 整次拒绝：不产生任何版本，也不留暂存残骸
    assert store.list_versions() == []
    assert list((tmp_path / "versions").glob(".staging-*")) == []


def test_save_rejects_duplicate_and_illegal_identities(tmp_path):
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    kwargs["cell_ids"] = ["cell_a1", "cell_a1", "cell_b1"]
    kwargs["batch_of_cell"] = ["batchA", "batchA", "batchB"]
    kwargs["normalized"] = normed[:3]
    with pytest.raises(PublishRejectedError, match="重复细胞身份") as exc:
        store.save_run(params={"qc": "x"}, **kwargs)
    assert any("cell_a1" in r for r in exc.value.reasons)
    assert store.list_versions() == []


def test_save_rejects_batch_mismatch(tmp_path):
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    bad_batches = list(filtered.batch_of_cell)
    bad_batches[0] = "batchB"  # cell_a1 在来源中属于 batchA
    kwargs["batch_of_cell"] = bad_batches
    with pytest.raises(PublishRejectedError, match="批次") as exc:
        store.save_run(params={"qc": "x"}, **kwargs)
    assert any("cell_a1" in r for r in exc.value.reasons)
    assert store.list_versions() == []


def test_save_rejects_gene_mismatch(tmp_path):
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    kwargs["gene_ids"] = ["GeneA", "GeneB", "GeneC", "NotInSource"]
    with pytest.raises(PublishRejectedError, match="基因身份与来源不一致"):
        store.save_run(params={"qc": "x"}, **kwargs)
    assert store.list_versions() == []


def test_reordered_result_keeps_source_identity_and_verifies(tmp_path):
    """结果顺序调整不改变来源身份：保存后重开仍能逐细胞核对。"""
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    order = [2, 0, 3, 1]  # 打乱结果行顺序
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    kwargs["cell_ids"] = [filtered.cell_ids[i] for i in order]
    kwargs["batch_of_cell"] = [filtered.batch_of_cell[i] for i in order]
    kwargs["normalized"] = normed[order]
    out = store.save_run(params={"qc": "reordered"}, **kwargs)
    # 来源指纹与顺序无关
    identity = json.loads((out / "input_identity.json").read_text(encoding="utf-8"))
    assert identity["input_fingerprint"] == matrix.content_fingerprint()
    # 重开核对：每个保留细胞的身份与批次归属都能对上来源
    verified = store.verify_version(out, matrix)
    assert verified["kept_cell_ids"] == kwargs["cell_ids"]
    assert verified["batch_of_cell"] == kwargs["batch_of_cell"]


def test_verify_version_detects_foreign_source(tmp_path):
    store = ResultStore(tmp_path)
    matrix, filtered, report, normed, rule_used = _pipeline_parts()
    out = store.save_run(
        params={"qc": "x"}, **_save_kwargs(matrix, filtered, report, normed, rule_used)
    )
    other = CountMatrix(
        counts=matrix.counts.copy(),
        cell_ids=matrix.cell_ids,
        gene_ids=tuple(f"Z{i}" for i in range(matrix.n_genes)),
        batch_of_cell=matrix.batch_of_cell,
    )
    with pytest.raises(PublishRejectedError):
        store.verify_version(out, other)


def test_merge_view_intersection_only_and_reasons():
    matrix = make_toy_matrix()
    relaxed = dict(QC, batchA=QCParams(min_total_counts=10, min_detected_genes=1))
    comparison = compare_qc(matrix, QC, relaxed, baseline_label="strict",
                            candidate_label="relaxed")
    assert set(comparison.intersection_ids) == {"cell_a1", "cell_a2", "cell_b1", "cell_b2"}
    assert comparison.only_candidate_ids == ["cell_a3"]
    assert comparison.only_baseline_ids == []
    merge = comparison.to_dict()["merge"]
    only_relaxed = merge["only_relaxed"]
    assert only_relaxed[0]["cell_id"] == "cell_a3"
    assert any("total_counts<500" in r for r in only_relaxed[0]["strict_reasons"])
    assert only_relaxed[0]["relaxed_reasons"] == []
    assert comparison.to_dict()["source_fingerprint"] == matrix.content_fingerprint()


def test_merge_reports_rejects_different_sources():
    from scplatform.compare import merge_reports

    matrix = make_toy_matrix()
    _, report1 = filter_cells(matrix, QC)
    other = CountMatrix(
        counts=matrix.counts.copy(),
        cell_ids=matrix.cell_ids,  # 相同名称、不同来源
        gene_ids=matrix.gene_ids,
        batch_of_cell=matrix.batch_of_cell,
    )
    changed = other.counts.copy()
    changed[0, 0] += 1
    other = CountMatrix(counts=changed, cell_ids=other.cell_ids,
                        gene_ids=other.gene_ids, batch_of_cell=other.batch_of_cell)
    _, report2 = filter_cells(other, QC)
    with pytest.raises(ValueError, match="来源指纹不同"):
        merge_reports(report1, report2)
