"""容量约束配对研究：相似性、容量、锁定冲突、批准闭包与版本化核对。"""

import numpy as np
import pytest

from scplatform import (
    AssociationDecision,
    CountMatrix,
    PairingConflictError,
    PublishRejectedError,
    ResultStore,
    approve_pairs,
    pairing_candidates,
    solve_pairing,
)

RNA_SRC = "rna_run"
PROTEIN_SRC = "protein_run"


def make_modalities():
    rna = CountMatrix(
        counts=np.array([[1, 2], [5, 0]], dtype=np.int64),
        cell_ids=("cell_a", "cell_b"),
        gene_ids=("g1", "g2"),
        batch_of_cell=("rna_batch", "rna_batch"),
    )
    protein = CountMatrix(
        counts=np.array([[1, 2], [0, 3]], dtype=np.int64),
        cell_ids=("cell_x", "cell_y"),
        gene_ids=("g1", "g2"),
        batch_of_cell=("protein_batch", "protein_batch"),
    )
    return rna, protein


def solve(rna, protein, **kwargs):
    return solve_pairing(
        rna, protein, rna_source=RNA_SRC, protein_source=PROTEIN_SRC, **kwargs
    )


def pair_set(plan):
    return {(p.rna_cell, p.protein_cell) for p in plan.pairs}


def test_candidates_distinguish_missing_from_true_zero():
    rna, protein = make_modalities()
    protein_measured = np.ones(protein.counts.shape, dtype=bool)
    protein_measured[1, :] = False  # cell_y 全部缺测
    candidates = {
        (c.rna_cell, c.protein_cell): c
        for c in pairing_candidates(
            rna, protein, rna_source=RNA_SRC, protein_source=PROTEIN_SRC,
            protein_measured=protein_measured,
        )
    }
    unobserved = candidates[(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_y")]
    assert unobserved.similarity is None
    assert unobserved.n_shared_features == 0
    # 真实零计数仍是测量值：cell_b 的 g2=0 参与相似性计算
    observed = candidates[(f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_x")]
    assert observed.similarity is not None
    assert observed.n_shared_features == 2


def test_unobserved_pairs_never_auto_paired():
    rna, protein = make_modalities()
    protein_measured = np.ones(protein.counts.shape, dtype=bool)
    protein_measured[1, :] = False
    plan = solve(rna, protein, protein_measured=protein_measured)
    assert plan.n_unobserved_candidates == 2
    assert pair_set(plan) == {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x")}
    assert plan.unpaired_protein == (f"{PROTEIN_SRC}::cell_y",)


def test_one_to_one_capacity_not_reused():
    rna, protein = make_modalities()
    plan = solve(rna, protein)
    assert pair_set(plan) == {
        (f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x"),
        (f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_y"),
    }
    rna_use = [p.rna_cell for p in plan.pairs]
    protein_use = [p.protein_cell for p in plan.pairs]
    assert len(set(rna_use)) == len(rna_use)
    assert len(set(protein_use)) == len(protein_use)


def test_limited_one_to_many_capacity():
    rna, protein = make_modalities()
    plan = solve(rna, protein, capacities={PROTEIN_SRC: 2})
    assert pair_set(plan) == {
        (f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x"),
        (f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_x"),
    }
    assert plan.unpaired_protein == (f"{PROTEIN_SRC}::cell_y",)
    # 容量仍不可超用：第三个 RNA 细胞不能再占用 cell_x
    rna3 = CountMatrix(
        counts=np.array([[1, 2], [5, 0], [2, 2]], dtype=np.int64),
        cell_ids=("cell_a", "cell_b", "cell_c"),
        gene_ids=("g1", "g2"),
        batch_of_cell=("rna_batch",) * 3,
    )
    plan3 = solve(rna3, protein, capacities={PROTEIN_SRC: 2})
    assert sum(1 for p in plan3.pairs if p.protein_cell == f"{PROTEIN_SRC}::cell_x") == 2


def test_conflicting_locks_reported_without_auto_revoke():
    rna, protein = make_modalities()
    locks = {
        (f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x"),
        (f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_y"),
    }
    with pytest.raises(PairingConflictError) as excinfo:
        solve(rna, protein, manual_locks=locks)
    message = str(excinfo.value)
    assert "cell_a" in message and "容量为 1" in message
    assert excinfo.value.conflicts  # 冲突逐条列出，绝不自动撤销


def test_unknown_lock_cells_reported():
    rna, protein = make_modalities()
    with pytest.raises(PairingConflictError, match="未知 RNA 细胞"):
        solve(rna, protein, manual_locks={("rna_run::ghost", f"{PROTEIN_SRC}::cell_x")})


def test_manual_locks_satisfied_and_marked():
    rna, protein = make_modalities()
    lock = (f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_x")
    plan = solve(rna, protein, manual_locks={lock})
    assert lock in pair_set(plan)
    locked = next(p for p in plan.pairs if (p.rna_cell, p.protein_cell) == lock)
    assert locked.basis == "manual"
    assert plan.locked_pairs() == {lock}


def test_compare_with_existing_decisions():
    rna, protein = make_modalities()
    plan = solve(rna, protein)
    decisions = [
        AssociationDecision("cell_a", f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x", "manual"),
        AssociationDecision("cell_b", f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_x", "manual"),
    ]
    comparison = plan.compare_with_decisions(decisions)
    assert comparison["n_existing"] == 2
    assert comparison["n_preserved"] == 1
    assert comparison["preservation_rate"] == 0.5
    assert comparison["overall_consistency"] == plan.overall_consistency()
    assert comparison["dropped"] == [[f"{RNA_SRC}::cell_b", f"{PROTEIN_SRC}::cell_x"]]


def test_approve_pairs_expands_to_dependency_closure():
    rna, protein = make_modalities()
    plan = solve(rna, protein, capacities={PROTEIN_SRC: 2})
    approved = approve_pairs(plan, {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x")})
    # cell_x 被两对共享：批准一对必须连同依赖它的另一对
    assert pair_set(approved) == pair_set(plan)

    one_to_one = solve(rna, protein)
    partial = approve_pairs(one_to_one, {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x")})
    assert pair_set(partial) == {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x")}
    assert f"{RNA_SRC}::cell_b" in partial.unpaired_rna
    assert f"{PROTEIN_SRC}::cell_y" in partial.unpaired_protein


def test_approve_pairs_rejects_unknown_pairs():
    rna, protein = make_modalities()
    plan = solve(rna, protein)
    with pytest.raises(ValueError, match="不在方案中"):
        approve_pairs(plan, {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_y")})


def test_store_roundtrip_and_idempotent(tmp_path):
    rna, protein = make_modalities()
    store = ResultStore(tmp_path / "results")
    plan = solve(rna, protein)
    out = store.save_pairing(plan=plan, rna=rna, protein=protein, params={"round": 1})
    again = store.save_pairing(plan=plan, rna=rna, protein=protein, params={"round": 1})
    assert out == again  # 相同来源+参数幂等复用

    loaded = store.load_pairing(out)
    assert pair_set(loaded) == pair_set(plan)
    summary = store.verify_pairing(out, rna, protein)
    assert summary["status"] == "ok"
    assert summary["n_pairs"] == plan.n_pairs
    assert summary["overall_consistency"] == plan.overall_consistency()


def test_remeasurement_forms_new_version_not_overwrite(tmp_path):
    rna, protein = make_modalities()
    store = ResultStore(tmp_path / "results")
    plan = solve(rna, protein)
    out_v1 = store.save_pairing(plan=plan, rna=rna, protein=protein, params={"round": 1})

    # 补测改变来源内容：指纹变化，自动生成新版本
    rna_v2 = CountMatrix(
        counts=np.array([[1, 2], [5, 1]], dtype=np.int64),
        cell_ids=("cell_a", "cell_b"),
        gene_ids=("g1", "g2"),
        batch_of_cell=("rna_batch", "rna_batch"),
    )
    plan_v2 = solve(rna_v2, protein)
    out_v2 = store.save_pairing(plan=plan_v2, rna=rna_v2, protein=protein, params={"round": 1})
    assert out_v2 != out_v1
    assert store.verify_pairing(out_v1, rna, protein)["status"] == "ok"  # 旧版本解释不变
    with pytest.raises(PublishRejectedError):
        store.verify_pairing(out_v1, rna_v2, protein)


def test_save_pairing_rejects_tampered_plan(tmp_path):
    rna, protein = make_modalities()
    store = ResultStore(tmp_path / "results")
    plan = solve(rna, protein)
    from scplatform import PairedMatch

    tampered = approve_pairs(plan, {(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x")})
    fake = type(tampered)(
        pairs=(PairedMatch(f"{RNA_SRC}::cell_a", f"{PROTEIN_SRC}::cell_x", "similarity", 0.12345),),
        capacities=tampered.capacities,
        manual_locks=tampered.manual_locks,
        min_similarity=tampered.min_similarity,
        rna_source=tampered.rna_source,
        protein_source=tampered.protein_source,
        source_fingerprints=tampered.source_fingerprints,
        batches=tampered.batches,
        unpaired_rna=tampered.unpaired_rna,
        unpaired_protein=tampered.unpaired_protein,
        n_unobserved_candidates=tampered.n_unobserved_candidates,
    )
    with pytest.raises(PublishRejectedError, match="相似性"):
        store.save_pairing(plan=fake, rna=rna, protein=protein, params={})
    assert store.list_pairings() == []  # 整次拒绝，不留部分版本


def test_withdraw_batch_impact_includes_pairings_and_downstream(tmp_path):
    rna, protein = make_modalities()
    store = ResultStore(tmp_path / "results")
    plan = solve(rna, protein)
    out = store.save_pairing(plan=plan, rna=rna, protein=protein, params={})
    store.register_downstream_filter(
        kind="pairing", version=out.name, filter_id="high_consistency"
    )
    impact = store.withdraw_batch_impact("rna_batch")
    assert out.name in impact["pairings"]
    assert impact["downstream_filters"]["pairing"][out.name] == ["high_consistency"]
    with pytest.raises(PublishRejectedError, match="撤回"):
        store.verify_pairing(out, rna, protein)
