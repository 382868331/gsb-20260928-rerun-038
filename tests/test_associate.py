"""两种测量（RNA+蛋白）关联行为核对：重复名人工选择、零与缺测区分、
保存重开、补测新版本、撤回与下游影响追踪。"""

import numpy as np
import pytest

from scplatform import (
    AmbiguousPairingError,
    CountMatrix,
    NormalizeParams,
    PublishRejectedError,
    QCParams,
    ResultStore,
    associate_assays,
    filter_cells,
    normalize,
)


def make_rna():
    """RNA 单批次：cell_dup 与蛋白两侧重名；cell_zero 全零真实计数；
    cell_rna_only 没有蛋白测量。"""
    return {
        "rna1": CountMatrix(
            counts=np.array([[5, 0, 3], [0, 0, 0], [7, 1, 0]], dtype=np.int64),
            cell_ids=("cell_dup", "cell_zero", "cell_rna_only"),
            gene_ids=("G1", "G2", "G3"),
            batch_of_cell=("rna1",) * 3,
        )
    }


def make_protein():
    """蛋白两批次：cell_dup 在 prot1/prot2 各出现一次（一对多候选）；
    cell_zero 真实零测量；cell_prot_only 没有 RNA 测量。"""
    return {
        "prot1": CountMatrix(
            counts=np.array([[10, 2], [0, 0], [4, 0]], dtype=np.int64),
            cell_ids=("cell_dup", "cell_zero", "cell_prot_only"),
            gene_ids=("M1", "M2"),
            batch_of_cell=("prot1",) * 3,
        ),
        "prot2": CountMatrix(
            counts=np.array([[9, 9]], dtype=np.int64),
            cell_ids=("cell_dup",),
            gene_ids=("M1", "M2"),
            batch_of_cell=("prot2",),
        ),
    }


MANUAL = (("rna1::cell_dup", "prot2::cell_dup"),)


def test_duplicate_display_name_never_auto_paired():
    with pytest.raises(AmbiguousPairingError) as excinfo:
        associate_assays(make_rna(), make_protein(), "paired")
    unresolved = excinfo.value.unresolved
    assert set(unresolved) == {"cell_dup"}
    assert unresolved["cell_dup"]["rna"] == ["rna1::cell_dup"]
    assert unresolved["cell_dup"]["protein"] == ["prot1::cell_dup", "prot2::cell_dup"]


def test_manual_choice_resolves_one_to_many():
    joint = associate_assays(make_rna(), make_protein(), "paired",
                             manual_links=MANUAL)
    links = {l.rna_key: l for l in joint.links}
    assert links["rna1::cell_dup"].protein_key == "prot2::cell_dup"
    assert links["rna1::cell_dup"].decided_by == "manual"
    assert links["rna1::cell_zero"].decided_by == "auto_unique"
    # paired 范围只保留配对行
    assert joint.n_rows == 2
    assert set(joint.row_keys) == {
        "rna1::cell_dup|prot2::cell_dup",
        "rna1::cell_zero|prot1::cell_zero",
    }


def test_manual_link_validation():
    with pytest.raises(ValueError, match="不存在"):
        associate_assays(make_rna(), make_protein(), "paired",
                         manual_links=(("rna1::ghost", "prot1::cell_dup"),))
    with pytest.raises(ValueError, match="显示名不一致"):
        associate_assays(make_rna(), make_protein(), "paired",
                         manual_links=(("rna1::cell_dup", "prot1::cell_zero"),))


def test_scope_all_keeps_unpaired_with_modality_missing():
    joint = associate_assays(make_rna(), make_protein(), "all",
                             manual_links=MANUAL)
    # 2 配对 + cell_rna_only + prot1::cell_dup + cell_prot_only
    assert joint.n_rows == 5
    row = joint.row_keys.index("rna1::cell_rna_only")
    protein_cols = [j for j, m in enumerate(joint.modality_of_feature)
                    if m == "protein"]
    # 整种蛋白测量缺失：NaN 且 measured=False，绝不是 0
    assert np.isnan(joint.values[row, protein_cols]).all()
    assert not joint.measured[row, protein_cols].any()
    rna_cols = [j for j, m in enumerate(joint.modality_of_feature) if m == "rna"]
    assert joint.measured[row, rna_cols].all()
    row_p = joint.row_keys.index("prot1::cell_prot_only")
    assert np.isnan(joint.values[row_p, rna_cols]).all()
    assert not joint.measured[row_p, rna_cols].any()


def test_true_zero_distinct_from_missing():
    joint = associate_assays(make_rna(), make_protein(), "paired",
                             manual_links=MANUAL)
    row = joint.row_keys.index("rna1::cell_zero|prot1::cell_zero")
    # 真实零：两种测量都是测量值 0 且 measured=True
    assert (joint.values[row] == 0).all()
    assert joint.measured[row].all()
    assert not np.isnan(joint.values[row]).any()


def test_field_order_and_identity_preserved():
    joint = associate_assays(make_rna(), make_protein(), "paired",
                             manual_links=MANUAL)
    assert joint.feature_ids == (
        "rna::G1", "rna::G2", "rna::G3", "protein::M1", "protein::M2",
    )
    assert joint.modality_of_feature == ("rna",) * 3 + ("protein",) * 2


def test_save_reopen_verify_and_remeasure_new_version(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_rna(), make_protein()
    joint = associate_assays(rna, protein, "all", manual_links=MANUAL)
    params = {"note": "assoc-v1"}
    first = store.save_association(joint=joint, rna=rna, protein=protein,
                                   params=params)
    again = store.save_association(joint=joint, rna=rna, protein=protein,
                                   params=params)
    assert first == again  # 幂等

    # 换实例重开：行列身份、字段顺序、关联决定完整恢复
    store2 = ResultStore(tmp_path)
    reopened = store2.load_association(first)
    assert reopened.row_keys == joint.row_keys
    assert reopened.feature_ids == joint.feature_ids
    assert [l.to_dict() for l in reopened.links] == [l.to_dict() for l in joint.links]
    summary = store2.verify_association(first, rna, protein)
    assert summary["status"] == "ok"
    assert summary["feature_ids"] == list(joint.feature_ids)

    # prot2 补测 M3：来源指纹变化 -> 新版本，旧版本解释不变
    old = protein["prot2"]
    protein2 = dict(protein)
    protein2["prot2"] = CountMatrix(
        counts=np.array([[9, 9, 6]], dtype=np.int64),
        cell_ids=old.cell_ids,
        gene_ids=("M1", "M2", "M3"),
        batch_of_cell=old.batch_of_cell,
    )
    joint2 = associate_assays(rna, protein2, "all", manual_links=MANUAL)
    second = store2.save_association(joint=joint2, rna=rna, protein=protein2,
                                     params=params)
    assert second != first
    assert len(store2.list_assocs()) == 2
    old_summary = store2.verify_association(first, rna, protein)
    assert old_summary["feature_ids"] == list(joint.feature_ids)  # 旧结果不改


def test_publish_rejects_tampered_and_leaves_nothing(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_rna(), make_protein()
    joint = associate_assays(rna, protein, "paired", manual_links=MANUAL)
    other_rna = {
        "rna1": CountMatrix(
            counts=np.array([[1, 1, 1], [0, 0, 0], [7, 1, 0]], dtype=np.int64),
            cell_ids=("cell_dup", "cell_zero", "cell_rna_only"),
            gene_ids=("G1", "G2", "G3"),
            batch_of_cell=("rna1",) * 3,
        )
    }
    with pytest.raises(PublishRejectedError):
        store.save_association(joint=joint, rna=other_rna, protein=protein,
                               params={})
    assert store.list_assocs() == []


def _save_downstream_filter(store, depends_on, note="dep"):
    matrix = CountMatrix(
        counts=np.array([[600, 300, 100], [40, 30, 20]], dtype=np.int64),
        cell_ids=("c1", "c2"),
        gene_ids=("G1", "G2", "G3"),
        batch_of_cell=("b1", "b1"),
    )
    qc = {"b1": QCParams(min_total_counts=10, min_detected_genes=2)}
    norm = {"b1": NormalizeParams(target_sum=1e4, log1p=True)}
    filtered, report = filter_cells(matrix, qc)
    normalized, rule_used = normalize(filtered, norm)
    return store.save_run(
        source_matrix=matrix,
        params={"note": note, "qc": {b: p.to_dict() for b, p in qc.items()}},
        qc_report=report.to_dict(),
        normalized=normalized,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
        depends_on=depends_on,
    )


def test_withdraw_association_lists_affected_filters(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_rna(), make_protein()
    joint = associate_assays(rna, protein, "paired", manual_links=MANUAL)
    assoc_dir = store.save_association(joint=joint, rna=rna, protein=protein,
                                       params={})
    dependent = _save_downstream_filter(store, [assoc_dir.name])
    independent = _save_downstream_filter(store, ["some_other_assoc"], note="indep")
    assert dependent != independent

    impact = store.withdraw_association(assoc_dir)
    assert impact["association"] == assoc_dir.name
    assert impact["filters"] == [dependent.name]  # 只列出受影响的下游筛选
    assert store.association_status(assoc_dir) == "withdrawn"
    with pytest.raises(PublishRejectedError, match="失效"):
        store.verify_association(assoc_dir, rna, protein)


def test_withdrawal_impact_of_batch_traces_assoc_and_filters(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_rna(), make_protein()
    joint = associate_assays(rna, protein, "paired", manual_links=MANUAL)
    assoc_dir = store.save_association(joint=joint, rna=rna, protein=protein,
                                       params={})
    dependent = _save_downstream_filter(store, [assoc_dir.name])

    impact = store.withdrawal_impact("prot2")
    assert impact["associations"] == [assoc_dir.name]
    assert impact["filters"] == [dependent.name]
    # 撤回来源批次后关联结果随之失效
    store.withdraw_batch("prot2")
    assert store.association_status(assoc_dir) == "withdrawn:prot2"
    with pytest.raises(PublishRejectedError):
        store.verify_association(assoc_dir, rna, protein)
