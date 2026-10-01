"""联合分析集行为核对：缺测不等于零、来源区分、模式比较、版本与撤回。""" 

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    PublishRejectedError,
    QCParams,
    ResultStore,
    compare_join_modes,
    join_batches,
    qc_on_joint,
)


def make_batches():
    """两个批次：基因集合重叠但不完全相同，且有同显示名的细胞。"""
    batch1 = CountMatrix(
        counts=np.array(
            [
                [100, 5, 5],   # c1: 总计 110，检出 3
                [0, 1, 0],     # c2: 低计数，G1 是真实测量到的 0
            ],
            dtype=np.int64,
        ),
        cell_ids=("c1", "c2"),
        gene_ids=("G1", "G2", "G3"),
        batch_of_cell=("b1", "b1"),
    )
    batch2 = CountMatrix(
        counts=np.array(
            [
                [40, 30, 20],  # 显示名同样是 c1，但来源不同
                [10, 0, 0],    # c3: 低计数
            ],
            dtype=np.int64,
        ),
        cell_ids=("c1", "c3"),
        gene_ids=("G2", "G3", "G4"),
        batch_of_cell=("b2", "b2"),
    )
    return {"b1": batch1, "b2": batch2}


QC = {
    "b1": QCParams(min_total_counts=50, min_detected_genes=3),
    "b2": QCParams(min_total_counts=50, min_detected_genes=3),
}


def test_union_marks_missing_as_nan_not_zero():
    joint = join_batches(make_batches(), mode="union")
    assert joint.gene_ids == ("G1", "G2", "G3", "G4")
    # b2 的细胞没有测量 G1：NaN + observed=False，绝不是 0
    row = joint.cell_keys.index("b2::c1")
    g1 = joint.gene_ids.index("G1")
    assert np.isnan(joint.values[row, g1])
    assert not joint.observed[row, g1]
    # b1::c2 的 G1 是真实测量到的 0：值为 0.0 且 observed=True
    row2 = joint.cell_keys.index("b1::c2")
    assert joint.values[row2, g1] == 0.0
    assert joint.observed[row2, g1]
    assert joint.missing_by_batch() == {"b1": ["G4"], "b2": ["G1"]}


def test_intersection_keeps_only_common_genes():
    joint = join_batches(make_batches(), mode="intersection")
    assert joint.gene_ids == ("G2", "G3")
    assert joint.observed.all()
    assert joint.missing_by_batch() == {"b1": [], "b2": []}


def test_same_display_name_distinguished_by_source():
    joint = join_batches(make_batches(), mode="union")
    assert "b1::c1" in joint.cell_keys and "b2::c1" in joint.cell_keys
    assert len(set(joint.cell_keys)) == joint.n_cells
    # 显示名可重复，但联合键绝不合并
    assert list(joint.display_names).count("c1") == 2


def test_join_requires_common_genes_for_intersection():
    batches = make_batches()
    batches["b2"] = CountMatrix(
        counts=np.ones((1, 1), dtype=np.int64),
        cell_ids=("x",),
        gene_ids=("G9",),
        batch_of_cell=("b2",),
    )
    with pytest.raises(ValueError, match="共同"):
        join_batches(batches, mode="intersection")


def test_qc_on_joint_ignores_missing_values():
    batches = make_batches()
    joint = join_batches(batches, mode="union")
    report = qc_on_joint(joint, QC)
    # b1::c1 总计 110、检出 3 -> 保留；缺测的 G4 不参与计数
    assert "b1::c1" in report.kept_ids
    # b1::c2 总计 1 -> 剔除，原因不含把缺测当零的条目
    verdict = next(v for v in report.verdicts if v.cell_key == "b1::c2")
    assert not verdict.kept
    assert verdict.n_missing_genes == 1
    assert any("total_counts<50" in r for r in verdict.reasons)


def test_compare_join_modes_shows_retention_difference():
    comparison = compare_join_modes(make_batches(), QC)
    assert comparison.n_genes_intersection == 2
    assert comparison.n_genes_union == 4
    # intersection 只看共同基因 G2,G3：b1::c1 总计 10、检出 2 -> 剔除；
    # union 用各自批次的全部测量 -> 保留
    assert "b1::c1" in comparison.entered_ids
    assert "b1::c1" not in comparison.intersection_report.kept_ids
    assert "b1::c1" in comparison.union_report.kept_ids
    assert comparison.left_ids == []
    assert comparison.missing_by_batch == {"b1": ["G4"], "b2": ["G1"]}
    summary = comparison.to_dict()
    # b2::c1 在共同基因上只检出 2 个，union 下检出 3 个，同样进入保留集
    assert summary["only_union_kept"] == ["b1::c1", "b2::c1"]


def test_save_and_verify_joint_roundtrip(tmp_path):
    store = ResultStore(tmp_path / "results")
    batches = make_batches()
    joint = join_batches(batches, mode="union")
    comparison = compare_join_modes(batches, QC)
    out = store.save_joint(
        sources=batches,
        joint=joint,
        params={"qc": {b: p.to_dict() for b, p in QC.items()}},
        qc_report=qc_on_joint(joint, QC).to_dict(),
        comparison=comparison.to_dict(),
    )
    assert (out / "joint.npz").exists()
    # 重开后逐细胞、逐基因核对通过，缺测说明随版本保存
    info = store.verify_joint(out, batches)
    assert info["status"] == "ok"
    assert info["cell_keys"] == list(joint.cell_keys)
    assert info["missing_by_batch"] == {"b1": ["G4"], "b2": ["G1"]}
    # 幂等：相同来源 + 参数重存复用同一版本
    again = store.save_joint(sources=batches, joint=joint,
                             params={"qc": {b: p.to_dict() for b, p in QC.items()}})
    assert again == out


def test_remeasured_batch_creates_new_version_without_overwrite(tmp_path):
    store = ResultStore(tmp_path / "results")
    batches = make_batches()
    joint_v1 = join_batches(batches, mode="union")
    out_v1 = store.save_joint(sources=batches, joint=joint_v1, params={})

    # b2 后来补测了 G1：内容指纹改变 -> 新联合版本，旧版本保留
    old = batches["b2"]
    batches2 = dict(batches)
    batches2["b2"] = CountMatrix(
        counts=np.array([[40, 30, 20, 7], [10, 0, 0, 0]], dtype=np.int64),
        cell_ids=old.cell_ids,
        gene_ids=("G2", "G3", "G4", "G1"),
        batch_of_cell=old.batch_of_cell,
    )
    joint_v2 = join_batches(batches2, mode="union")
    out_v2 = store.save_joint(sources=batches2, joint=joint_v2, params={})
    assert out_v2 != out_v1
    assert (out_v1 / "joint.npz").exists()
    assert len(store.list_joint_versions()) == 2
    # 旧版本仍可用旧来源核对通过
    assert store.verify_joint(out_v1, batches)["status"] == "ok"
    assert store.verify_joint(out_v2, batches2)["status"] == "ok"


def test_publish_rejects_inconsistent_joint_and_leaves_nothing(tmp_path):
    store = ResultStore(tmp_path / "results")
    batches = make_batches()
    joint = join_batches(batches, mode="union")

    # 篡改缺测说明：把 b2 未测量的 G1 伪装成已测量的 0
    tampered_values = joint.values.copy()
    tampered_observed = joint.observed.copy()
    row = joint.cell_keys.index("b2::c1")
    col = joint.gene_ids.index("G1")
    tampered_values[row, col] = 0.0
    tampered_observed[row, col] = True
    from scplatform import JointMatrix
    tampered = JointMatrix(
        values=tampered_values,
        observed=tampered_observed,
        cell_keys=joint.cell_keys,
        display_names=joint.display_names,
        gene_ids=joint.gene_ids,
        batch_of_cell=joint.batch_of_cell,
        mode=joint.mode,
        source_fingerprints=joint.source_fingerprints,
    )
    with pytest.raises(PublishRejectedError):
        store.save_joint(sources=batches, joint=tampered, params={})
    assert store.list_joint_versions() == []
    assert not list((tmp_path / "results" / "versions").glob(".staging-*"))

    # 基因顺序与来源不一致同样整次拒绝
    reordered = JointMatrix(
        values=joint.values[:, ::-1],
        observed=joint.observed[:, ::-1],
        cell_keys=joint.cell_keys,
        display_names=joint.display_names,
        gene_ids=tuple(reversed(joint.gene_ids)),
        batch_of_cell=joint.batch_of_cell,
        mode=joint.mode,
        source_fingerprints=joint.source_fingerprints,
    )
    with pytest.raises(PublishRejectedError):
        store.save_joint(sources=batches, joint=reordered, params={})
    assert store.list_joint_versions() == []


def test_withdraw_batch_only_affects_dependent_joints(tmp_path):
    store = ResultStore(tmp_path / "results")
    batches = make_batches()
    out_inter = store.save_joint(
        sources=batches, joint=join_batches(batches, mode="intersection"), params={}
    )
    out_union = store.save_joint(
        sources=batches, joint=join_batches(batches, mode="union"), params={}
    )

    affected = store.withdraw_batch("b1")
    assert set(affected) == {out_inter.name, out_union.name}
    info = store.verify_joint(out_union, batches)
    assert info["status"] == "withdrawn_dependency"
    assert info["withdrawn_dependencies"] == ["b1"]

    # 不依赖 b1 的联合版本不受影响
    only_b2 = {"b2": batches["b2"]}
    out_b2 = store.save_joint(
        sources=only_b2, joint=join_batches(only_b2, mode="union"), params={}
    )
    assert store.verify_joint(out_b2, only_b2)["status"] == "ok"

    # 精确撤回某一次测量：补测后的新版本不受影响
    old_fp = batches["b1"].content_fingerprint()
    store2 = ResultStore(tmp_path / "results2")
    out_old = store2.save_joint(
        sources=batches, joint=join_batches(batches, mode="union"), params={}
    )
    batches_new = dict(batches)
    b1 = batches["b1"]
    # 构造一个真正补测过的 b1（多测一个基因）
    batches_new["b1"] = CountMatrix(
        counts=np.column_stack([b1.counts, [3, 0]]),
        cell_ids=b1.cell_ids,
        gene_ids=b1.gene_ids + ("G4",),
        batch_of_cell=b1.batch_of_cell,
    )
    out_new = store2.save_joint(
        sources=batches_new, joint=join_batches(batches_new, mode="union"), params={}
    )
    affected2 = store2.withdraw_batch("b1", fingerprint=old_fp)
    assert affected2 == [out_old.name]
    assert store2.verify_joint(out_new, batches_new)["status"] == "ok"
