"""平台行为核对：筛选数量、归一化数值、批次规则隔离、版本不覆盖。"""

import json

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    PublishRejected,
    NormalizeParams,
    QCParams,
    ResultStore,
    compare_qc,
    filter_cells,
    load_matrix,
    merge_qc,
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


# ---------------------------------------------------------------------------
# 第二轮加固：发布前身份核对、原子拒绝、重开核对、多方案合并比较
# ---------------------------------------------------------------------------


def _publish_pieces(tmp_path, **overrides):
    """跑一遍合法流程，允许覆盖任意结果片段以构造非法发布。"""
    store = ResultStore(tmp_path / "results")
    matrix = make_toy_matrix()
    filtered, report = filter_cells(matrix, QC)
    normed, rule_used = normalize(filtered, NORM)
    kwargs = _save_kwargs(matrix, filtered, report, normed, rule_used)
    kwargs["params"] = {"qc": {b: q.to_dict() for b, q in QC.items()}}
    kwargs.update(overrides)
    return store, kwargs


def test_save_rejects_illegal_cell_not_in_source(tmp_path):
    """原矩阵中不存在的细胞身份不能挂在原输入指纹下发布。"""
    store, kwargs = _publish_pieces(tmp_path)
    kwargs["cell_ids"] = [*kwargs["cell_ids"], "cell_ghost"]
    kwargs["batch_of_cell"] = [*kwargs["batch_of_cell"], "batchA"]
    kwargs["rule_used"] = {**kwargs["rule_used"], "cell_ghost": "batchA"}
    kwargs["normalized"] = np.vstack([kwargs["normalized"], np.zeros((1, 4))])
    before = set(p.name for p in store.list_versions())
    with pytest.raises(PublishRejected) as excinfo:
        store.save_run(**kwargs)
    joined = "\n".join(excinfo.value.errors)
    assert "非法身份" in joined and "cell_ghost" in joined
    # 原子拒绝：没有留下任何部分版本或暂存目录
    assert set(p.name for p in store.list_versions()) == before
    assert not list(store.versions_dir.glob(".staging-*"))


def test_save_rejects_duplicate_and_batch_mismatch(tmp_path):
    """重复身份与批次错配同时存在时，两类说明都要保留。"""
    store, kwargs = _publish_pieces(tmp_path)
    ids = list(kwargs["cell_ids"])
    ids[2] = ids[0]  # 制造重复细胞身份（cell_b1 位置换成 cell_a1）
    kwargs["cell_ids"] = ids
    batches = list(kwargs["batch_of_cell"])
    batches[0] = "batchB"  # cell_a1 被错配到 batchB
    kwargs["batch_of_cell"] = batches
    with pytest.raises(PublishRejected) as excinfo:
        store.save_run(**kwargs)
    joined = "\n".join(excinfo.value.errors)
    assert "重复细胞身份" in joined
    assert "批次归属与来源错配" in joined and "cell_a1" in joined
    assert not list(store.versions_dir.glob(".staging-*"))


def test_save_rejects_illegal_gene_and_shape_mismatch(tmp_path):
    store, kwargs = _publish_pieces(tmp_path)
    kwargs["gene_ids"] = ["GeneA", "GeneB", "GeneC", "GhostGene"]
    with pytest.raises(PublishRejected) as excinfo:
        store.save_run(**kwargs)
    joined = "\n".join(excinfo.value.errors)
    assert "GhostGene" in joined and "非法身份" in joined


def test_save_rejects_when_report_kept_disagrees(tmp_path):
    """QC 报告保留集合与实际结果不一致也要整次拒绝。"""
    store, kwargs = _publish_pieces(tmp_path)
    report = dict(kwargs["qc_report"])
    report["kept"] = [c for c in report["kept"] if c != "cell_b2"]
    kwargs["qc_report"] = report
    with pytest.raises(PublishRejected, match="qc_report"):
        store.save_run(**kwargs)


def test_reorder_and_resave_keeps_source_identity(tmp_path):
    """结果顺序调整 + 重复保存不改变来源身份，且仍能逐对象核对。"""
    store, kwargs = _publish_pieces(tmp_path)
    v1 = store.save_run(**kwargs)

    # 重排保留细胞顺序（矩阵行、ID、批次、rule_used 同步重排）
    order = [3, 0, 2, 1]
    reordered = dict(kwargs)
    reordered["normalized"] = kwargs["normalized"][order]
    reordered["cell_ids"] = [kwargs["cell_ids"][i] for i in order]
    reordered["batch_of_cell"] = [kwargs["batch_of_cell"][i] for i in order]
    reordered["rule_used"] = {
        kwargs["cell_ids"][i]: kwargs["rule_used"][kwargs["cell_ids"][i]]
        for i in order
    }
    v2 = store.save_run(**reordered)
    assert v2 == v1  # 顺序不影响版本身份，幂等返回已发布版本

    reopened = store.load_run(v1)
    matrix = make_toy_matrix()
    verified = store.load_run(v1, source_matrix=matrix)
    assert set(verified["cell_ids"]) == {"cell_a1", "cell_a2", "cell_b1", "cell_b2"}
    assert dict(zip(verified["cell_ids"], verified["batch_of_cell"])) == {
        "cell_a1": "batchA", "cell_a2": "batchA",
        "cell_b1": "batchB", "cell_b2": "batchB",
    }
    np.testing.assert_allclose(
        reopened["normalized"][reopened["cell_ids"].index("cell_b1")],
        np.log1p([400.0, 300.0, 200.0, 100.0]),
    )


def test_load_run_rejects_changed_source(tmp_path):
    """保存重开后用改动过的矩阵核对：必须报指纹不一致。"""
    store, kwargs = _publish_pieces(tmp_path)
    v1 = store.save_run(**kwargs)
    changed_counts = make_toy_matrix().counts.copy()
    changed_counts[0, 0] += 1
    tampered = CountMatrix(
        counts=changed_counts,
        cell_ids=make_toy_matrix().cell_ids,
        gene_ids=make_toy_matrix().gene_ids,
        batch_of_cell=make_toy_matrix().batch_of_cell,
    )
    with pytest.raises(PublishRejected, match="输入指纹"):
        store.load_run(v1, source_matrix=tampered)


def test_merge_qc_intersection_only_and_reasons():
    """多方案合并：交集、仅某方案保留、各自排除原因清楚列出。"""
    matrix = make_toy_matrix()
    strict = QC
    relaxed = dict(QC, batchA=QCParams(min_total_counts=10, min_detected_genes=1),
                   batchB=QCParams(min_total_counts=10, min_detected_genes=1))
    merged = merge_qc(
        {"strict": matrix, "relaxed": matrix},
        {"strict": strict, "relaxed": relaxed},
    )
    assert set(merged.intersection_ids()) == {"cell_a1", "cell_a2", "cell_b1", "cell_b2"}
    assert merged.only_kept_ids("relaxed") == ["cell_a3"]
    assert merged.only_kept_ids("strict") == []
    a3 = next(k for k in merged.excluded if k.cell_id == "cell_a3")
    assert any("total_counts<500" in r for r in merged.excluded[a3]["strict"])
    assert "relaxed" not in merged.excluded[a3]
    a4 = next(k for k in merged.excluded if k.cell_id == "cell_a4")
    assert "all_zero_counts" in merged.excluded[a4]["strict"]
    assert "all_zero_counts" in merged.excluded[a4]["relaxed"]
    assert merged.name_collisions == []
    payload = merged.to_dict()
    assert {c["cell_id"] for c in payload["intersection"]} >= {"cell_a1"}
    assert payload["intersection"][0]["source_fingerprint"]


def test_merge_qc_never_merges_same_name_different_source():
    """相同名称但不同来源的细胞不能合并，需明确列出同名碰撞。"""
    m1 = make_toy_matrix()
    # 另一个来源：同样的细胞名，但计数不同 -> 不同指纹
    counts2 = m1.counts.copy()
    counts2[0, 0] += 5
    m2 = CountMatrix(
        counts=counts2, cell_ids=m1.cell_ids, gene_ids=m1.gene_ids,
        batch_of_cell=m1.batch_of_cell,
    )
    relaxed = {b: QCParams(min_total_counts=1, min_detected_genes=1) for b in ("batchA", "batchB")}
    merged = merge_qc(
        {"src1": m1, "src2": m2},
        {"src1": relaxed, "src2": relaxed},
    )
    # 来源不同，交集必须为空
    assert merged.intersection == []
    collided = {c["cell_id"] for c in merged.name_collisions}
    assert "cell_a1" in collided
    collision = next(c for c in merged.name_collisions if c["cell_id"] == "cell_a1")
    schemes = {s["scheme"] for s in collision["sources"]}
    assert schemes == {"src1", "src2"}
    # 仅某方案保留的对象按 (指纹, ID) 区分，互不混淆
    only1 = {(r.source_fingerprint, r.cell_id) for r in merged.only_kept["src1"]}
    only2 = {(r.source_fingerprint, r.cell_id) for r in merged.only_kept["src2"]}
    assert only1 and only2 and only1.isdisjoint(only2)
