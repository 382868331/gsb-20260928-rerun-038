"""联合分析集行为核对：缺测不为零、模式比较、版本不覆盖、撤回隔离。"""

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    PublishRejectedError,
    QCParams,
    ResultStore,
    compare_join_modes,
    join_batches,
)


def make_batches():
    """两个基因集部分重叠的批次；cell_x 在两批次中同名但不同源。"""
    batch_a = CountMatrix(
        counts=np.array([[100, 50, 10], [5, 5, 0]], dtype=np.int64),
        cell_ids=("cell_x", "cell_a2"),
        gene_ids=("G1", "G2", "G3"),
        batch_of_cell=("batchA", "batchA"),
    )
    batch_b = CountMatrix(
        counts=np.array([[80, 40, 4], [60, 0, 6]], dtype=np.int64),
        cell_ids=("cell_x", "cell_b2"),
        gene_ids=("G2", "G3", "G4"),
        batch_of_cell=("batchB", "batchB"),
    )
    return {"batchA": batch_a, "batchB": batch_b}


def test_union_marks_missing_as_nan_not_zero():
    joint = join_batches(make_batches(), "union")
    assert joint.gene_ids == ("G1", "G2", "G3", "G4")
    # batchB 未测 G1：必须是 NaN 且 measured=False，绝不是 0
    row_b1 = joint.cell_keys.index("batchB::cell_x")
    g1 = joint.gene_ids.index("G1")
    assert np.isnan(joint.values[row_b1, g1])
    assert not joint.measured[row_b1, g1]
    # 真实测量位置有值且 measured=True；真实零计数仍是测量值 0
    g4 = joint.gene_ids.index("G4")
    assert joint.values[row_b1, g4] == 4 and joint.measured[row_b1, g4]
    row_b2 = joint.cell_keys.index("batchB::cell_b2")
    g3 = joint.gene_ids.index("G3")
    assert joint.values[row_b2, g3] == 0 and joint.measured[row_b2, g3]
    assert joint.missing_genes_by_batch() == {"batchA": ["G4"], "batchB": ["G1"]}


def test_same_display_name_kept_distinct_by_source():
    joint = join_batches(make_batches(), "union")
    assert joint.display_names.count("cell_x") == 2
    assert "batchA::cell_x" in joint.cell_keys
    assert "batchB::cell_x" in joint.cell_keys
    assert len(set(joint.cell_keys)) == joint.n_cells


def test_common_mode_uses_only_shared_genes():
    joint = join_batches(make_batches(), "common")
    assert joint.gene_ids == ("G2", "G3")
    assert joint.measured.all()  # 共同基因模式下无缺测


def test_no_common_genes_rejected():
    m1 = CountMatrix(np.ones((1, 1), dtype=int), ("c1",), ("G1",), ("b1",))
    m2 = CountMatrix(np.ones((1, 1), dtype=int), ("c2",), ("G2",), ("b2",))
    with pytest.raises(ValueError, match="共同基因"):
        join_batches({"b1": m1, "b2": m2}, "common")


def test_foreign_cells_in_batch_rejected():
    bad = CountMatrix(
        np.ones((1, 1), dtype=int), ("c1",), ("G1",), ("other_batch",)
    )
    with pytest.raises(ValueError, match="混入了其他批次"):
        join_batches({"b1": bad}, "union")


def test_compare_join_modes_shows_cell_retention_impact():
    qc = {
        "batchA": QCParams(min_total_counts=1, min_detected_genes=3),
        "batchB": QCParams(min_total_counts=1, min_detected_genes=3),
    }
    comparison = compare_join_modes(make_batches(), qc)
    # common 只有 G2,G3 两个基因 -> detected_genes<3；union 有 3 个实测基因
    assert comparison.common_kept == []
    assert set(comparison.union_kept) == {"batchA::cell_x", "batchB::cell_x"}
    assert set(comparison.only_union_kept) == set(comparison.union_kept)
    report = comparison.to_dict()
    assert report["common"]["n_genes"] == 2
    assert report["union"]["n_genes"] == 4
    assert report["union"]["missing_genes_by_batch"] == {
        "batchA": ["G4"], "batchB": ["G1"],
    }
    assert report["verdict_diffs"]  # 两种选择的质量判断差异被明确列出


def test_joint_versions_never_overwrite_and_remeasure_makes_new(tmp_path):
    store = ResultStore(tmp_path)
    sources = make_batches()
    joint = join_batches(sources, "union")
    params = {"note": "v1"}
    first = store.save_joint(joint=joint, sources=sources, params=params)
    again = store.save_joint(joint=joint, sources=sources, params=params)
    assert first == again  # 幂等

    # batchB 补测 G1：来源指纹变化 -> 新版本，旧版本不动
    old_b = sources["batchB"]
    new_b = CountMatrix(
        counts=np.array([[80, 40, 4, 7], [60, 0, 6, 3]], dtype=np.int64),
        cell_ids=old_b.cell_ids,
        gene_ids=("G2", "G3", "G4", "G1"),
        batch_of_cell=old_b.batch_of_cell,
    )
    sources2 = {"batchA": sources["batchA"], "batchB": new_b}
    joint2 = join_batches(sources2, "union")
    second = store.save_joint(joint=joint2, sources=sources2, params=params)
    assert second != first
    assert len(store.list_joints()) == 2
    # 旧版本重开仍可追溯
    reopened = store.load_joint(first)
    assert reopened.gene_ids == ("G1", "G2", "G3", "G4")
    assert reopened.missing_genes_by_batch()["batchB"] == ["G1"]


def test_joint_publish_rejects_inconsistent_and_leaves_nothing(tmp_path):
    store = ResultStore(tmp_path)
    sources = make_batches()
    joint = join_batches(sources, "union")
    # 篡改：把缺测伪装成 0
    bad_values = joint.values.copy()
    bad_values[np.isnan(bad_values)] = 0.0
    from scplatform import JointMatrix
    with pytest.raises(ValueError, match="缺测位置必须存储为 NaN"):
        JointMatrix(
            values=bad_values,
            measured=joint.measured,
            cell_keys=joint.cell_keys,
            display_names=joint.display_names,
            batch_of_cell=joint.batch_of_cell,
            gene_ids=joint.gene_ids,
            mode=joint.mode,
            source_fingerprints=joint.source_fingerprints,
        )
    # 来源被替换但沿用旧指纹 -> 发布前核对拒绝
    other = CountMatrix(
        np.ones((2, 3), dtype=int), ("cell_x", "cell_a2"), ("G1", "G2", "G3"),
        ("batchA", "batchA"),
    )
    with pytest.raises(PublishRejectedError):
        store.save_joint(
            joint=joint,
            sources={"batchA": other, "batchB": sources["batchB"]},
            params={},
        )
    assert store.list_joints() == []  # 不留下任何混杂版本


def test_verify_joint_after_reopen(tmp_path):
    store = ResultStore(tmp_path)
    sources = make_batches()
    joint = join_batches(sources, "union")
    out = store.save_joint(joint=joint, sources=sources, params={"p": 1})
    # 换实例重开，逐细胞逐组测量可追溯
    store2 = ResultStore(tmp_path)
    summary = store2.verify_joint(out, sources)
    assert summary["cell_keys"] == list(joint.cell_keys)
    assert summary["missing_genes_by_batch"] == {"batchA": ["G4"], "batchB": ["G1"]}
    with pytest.raises(PublishRejectedError):
        store2.verify_joint(out, {"batchA": sources["batchA"]})  # 来源不全


def test_withdraw_batch_only_affects_dependent_joints(tmp_path):
    store = ResultStore(tmp_path)
    sources = make_batches()
    joint_ab = join_batches(sources, "union")
    only_a = {"batchA": sources["batchA"]}
    joint_a = join_batches(only_a, "union")
    dir_ab = store.save_joint(joint=joint_ab, sources=sources, params={})
    dir_a = store.save_joint(joint=joint_a, sources=only_a, params={})

    affected = store.withdraw_batch("batchB")
    assert affected == [dir_ab.name]  # 只影响依赖 batchB 的联合结果
    assert store.joint_status(dir_ab) == "withdrawn:batchB"
    assert store.joint_status(dir_a) == "ok"
    with pytest.raises(PublishRejectedError, match="撤回"):
        store.verify_joint(dir_ab, sources)
    assert store.verify_joint(dir_a, only_a)["status"] == "ok"
