"""RNA/蛋白容量约束配对研究与部分批准后的下游关联测试。"""

import numpy as np
import pytest

from scplatform import (
    AssociationDecision,
    CountMatrix,
    PairingCapacities,
    ResultStore,
    associate_measurements,
    plan_capacity_pairing,
)


def make_capacity_data():
    rna = CountMatrix(
        counts=np.array([[1, 2], [4, 8]], dtype=np.int64),
        cell_ids=("r1", "r2"),
        gene_ids=("G1", "G2"),
        batch_of_cell=("rna_batch", "rna_batch"),
    )
    protein = CountMatrix(
        counts=np.array([[1, 2], [2, 1], [4, 8]], dtype=np.int64),
        cell_ids=("p1", "p2", "p3"),
        gene_ids=("G1", "G2"),
        batch_of_cell=("protein_batch", "protein_batch", "protein_batch"),
    )
    candidates = {
        ("rna_run::r1", "protein_run::p1", "family-a"),
        ("rna_run::r1", "protein_run::p2", "family-a"),
        ("rna_run::r2", "protein_run::p3", "family-b"),
    }
    return rna, protein, candidates


def test_limited_one_to_many_reuses_source_without_double_counting():
    rna, protein, candidates = make_capacity_data()
    plan = plan_capacity_pairing(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        capacities=PairingCapacities(
            rna={"rna_run::r1": 2}, default=1
        ),
        candidate_pairs=candidates,
    )
    assert plan.feasible
    assert set(plan.pairs) == {
        ("rna_run::r1", "protein_run::p1"),
        ("rna_run::r1", "protein_run::p2"),
        ("rna_run::r2", "protein_run::p3"),
    }
    assert plan.overall_consistency > 0
    report = plan.to_dict()
    assert report["capacities"]["rna"]["rna_run::r1"] == 2

    association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="paired_only",
        capacities=PairingCapacities(rna={"rna_run::r1": 2}, default=1),
        approved_decisions=plan.approved_decisions(),
        candidate_pairs=candidates,
    )
    assert association.row_keys == (
        "paired:rna_run::r1->protein_run::p1",
        "paired:rna_run::r1->protein_run::p2",
        "paired:rna_run::r2->protein_run::p3",
    )
    assert association.rna_cell_keys.count("rna_run::r1") == 2
    assert association.protein_cell_keys.count("protein_run::p1") == 1


def test_no_shared_observation_is_not_high_similarity_but_existing_can_retain():
    rna, protein, _ = make_capacity_data()
    protein_no_overlap = CountMatrix(
        counts=np.array([[9, 8], [7, 6], [5, 4]], dtype=np.int64),
        cell_ids=protein.cell_ids,
        gene_ids=("CD1", "CD2"),
        batch_of_cell=protein.batch_of_cell,
    )
    candidates = {
        ("rna_run::r1", "protein_run::p1", "family-a"),
        ("rna_run::r2", "protein_run::p2", "family-b"),
    }
    plan = plan_capacity_pairing(
        rna,
        protein_no_overlap,
        rna_source="rna_run",
        protein_source="protein_run",
        candidate_pairs=candidates,
        existing_pairs={("rna_run::r1", "protein_run::p1")},
    )
    assert plan.feasible
    assert plan.candidate_similarity == {}
    assert set(plan.no_observation_pairs) == {
        candidate_edge[:2] for candidate_edge in candidates
    }
    assert plan.pairs == (("rna_run::r1", "protein_run::p1"),)
    assert plan.retained_pairs == (("rna_run::r1", "protein_run::p1"),)
    assert plan.overall_consistency is None


def test_conflicting_locks_are_reported_and_not_automatically_released():
    rna, protein, candidates = make_capacity_data()
    plan = plan_capacity_pairing(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        capacities=PairingCapacities(default=1),
        locked_pairs={
            ("rna_run::r1", "protein_run::p1"),
            ("rna_run::r1", "protein_run::p2"),
        },
        candidate_pairs=candidates,
    )
    assert not plan.feasible
    assert plan.pairs == ()
    assert plan.conflicts[0]["type"] == "rna_capacity_locked"
    assert plan.conflicts[0]["locked_partners"] == [
        "protein_run::p1",
        "protein_run::p2",
    ]
    with pytest.raises(ValueError, match="无可行配对方案"):
        plan.approved_decisions()


def test_partial_approval_must_include_locked_closure_and_version_reverifies(tmp_path):
    rna, protein, candidates = make_capacity_data()
    lock = {("rna_run::r1", "protein_run::p1")}
    plan = plan_capacity_pairing(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        capacities=PairingCapacities(rna={"rna_run::r1": 2}, default=1),
        locked_pairs=lock,
        candidate_pairs=candidates,
    )
    with pytest.raises(ValueError, match="锁定配对"):
        plan.approved_decisions({("rna_run::r1", "protein_run::p2")})

    approved = plan.approved_decisions(
            {
                ("rna_run::r1", "protein_run::p1"),
                ("rna_run::r1", "protein_run::p2"),
            }
        )
    assert [decision.basis for decision in approved] == ["manual", "capacity_approved"]
    association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        capacities=PairingCapacities(rna={"rna_run::r1": 2}, default=1),
        approved_decisions=approved,
        candidate_pairs=candidates,
    )
    store = ResultStore(tmp_path)
    out = store.save_association(
        association=association, rna=rna, protein=protein, params={"round": 1}
    )
    reopened = store.load_association(out)
    assert reopened.capacities.rna_capacity("rna_run::r1") == 2
    summary = store.verify_association(out, rna, protein)
    assert summary["row_keys"] == list(association.row_keys)
    store.register_downstream_filter(
        kind="association", version=out.name, filter_id="joint_qc"
    )
    impact = store.withdraw_batch_impact("protein_batch")
    assert impact["associations"] == [out.name]
    assert impact["downstream_filters"]["association"] == {out.name: ["joint_qc"]}
