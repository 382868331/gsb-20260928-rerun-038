"""RNA/蛋白关联、人工消歧、持久化与下游影响追踪测试。"""

import numpy as np
import pytest

from scplatform import (
    CountMatrix,
    PublishRejectedError,
    ResultStore,
    associate_measurements,
    association_candidates,
)


def make_modalities():
    rna = CountMatrix(
        counts=np.array([[1, 0], [2, 3], [4, 5]], dtype=np.int64),
        cell_ids=("shared", "dup", "rna_only"),
        gene_ids=("geneA", "geneB"),
        batch_of_cell=("rna_batch", "rna_batch", "rna_batch"),
    )
    protein = CountMatrix(
        counts=np.array([[10, 20], [30, 0], [40, 50]], dtype=np.int64),
        cell_ids=("shared", "dup", "protein_only"),
        gene_ids=("CD1", "CD2"),
        batch_of_cell=("protein_batch", "protein_batch", "protein_batch"),
    )
    return rna, protein


def manual_duplicate_pair():
    return {
        ("rna_run::shared", "protein_run::shared"),
        ("rna_run::dup", "protein_run::dup"),
    }


def test_candidates_do_not_pair_duplicate_display_names():
    rna, protein = make_modalities()
    candidates = {c.display_name: c for c in association_candidates(
        rna, protein, rna_source="rna_run", protein_source="protein_run"
    )}
    assert candidates["shared"].status == "ambiguous"
    assert candidates["dup"].status == "ambiguous"
    assert candidates["dup"].rna_cells == ("rna_run::dup",)
    assert candidates["dup"].protein_cells == ("protein_run::dup",)


def test_paired_only_requires_manual_selection_for_ambiguous_names():
    rna, protein = make_modalities()
    with pytest.raises(ValueError, match="不能自动配对"):
        associate_measurements(
            rna,
            protein,
            rna_source="rna_run",
            protein_source="protein_run",
            mode="paired_only",
        )

    paired = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="paired_only",
        manual_pairs=manual_duplicate_pair(),
    )
    assert paired.row_keys == (
        "paired:rna_run::shared->protein_run::shared",
        "paired:rna_run::dup->protein_run::dup",
    )
    assert [d.basis for d in paired.decisions] == ["manual", "manual"]
    assert paired.fields == (("rna", "geneA"), ("rna", "geneB"), ("protein", "CD1"), ("protein", "CD2"))


def test_all_cells_distinguishes_measured_zero_from_entire_missing_modality():
    rna, protein = make_modalities()
    association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    assert len(association.row_keys) == 4
    rna_only_row = association.row_keys.index("unpaired-rna:rna_run::rna_only")
    protein_only_row = association.row_keys.index("unpaired-protein:protein_run::protein_only")
    assert association.missing_measurement_by_row()[association.row_keys[rna_only_row]] == ["protein"]
    assert association.missing_measurement_by_row()[association.row_keys[protein_only_row]] == ["rna"]
    assert np.isnan(association.values[rna_only_row, 2:]).all()
    assert not association.measured[rna_only_row, 2:].any()
    assert np.isnan(association.values[protein_only_row, :2]).all()
    assert not association.measured[protein_only_row, :2].any()
    shared_row = association.row_keys.index("paired:rna_run::shared->protein_run::shared")
    assert association.values[shared_row, 1] == 0
    assert association.measured[shared_row, 1]
    dup_row = association.row_keys.index("paired:rna_run::dup->protein_run::dup")
    assert association.values[dup_row, 3] == 0
    assert association.measured[dup_row, 3]


def test_save_reopen_and_reverify_preserves_identities_order_and_decisions(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_modalities()
    all_cells = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    params = {"selection": "small-data"}
    paired_only = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="paired_only",
        manual_pairs=manual_duplicate_pair(),
    )
    first_out = None
    for association, mode, mode_params in [
        (all_cells, "all_cells", params),
        (paired_only, "paired_only", {**params, "mode": "paired_only"}),
    ]:
        out = store.save_association(
            association=association,
            rna=rna,
            protein=protein,
            params=mode_params,
        )
        if first_out is None:
            first_out = out
        reopened = store.load_association(out)
        assert reopened.mode == mode
        assert reopened.row_keys == association.row_keys
        assert reopened.display_names == association.display_names
        assert reopened.fields == association.fields
        assert reopened.rna_cell_keys == association.rna_cell_keys
        assert reopened.protein_cell_keys == association.protein_cell_keys
        assert reopened.decisions == association.decisions
        assert reopened.to_dict()["association_decisions"] == [
            decision.to_dict() for decision in association.decisions
        ]
        assert reopened.to_dict()["fields"] == [
            list(field) for field in association.fields
        ]
        np.testing.assert_array_equal(reopened.measured, association.measured)
        np.testing.assert_allclose(
            reopened.values[reopened.measured],
            association.values[association.measured],
        )
        summary = store.verify_association(out, rna, protein)
        assert summary["row_keys"] == list(association.row_keys)
        assert summary["missing_measurement_by_row"] == (
            association.missing_measurement_by_row()
        )
    assert store.save_association(
        association=all_cells, rna=rna, protein=protein, params=params
    ) == first_out


def test_remeasurement_creates_new_version_without_changing_old_interpretation(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_modalities()
    first_association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    first = store.save_association(
        association=first_association, rna=rna, protein=protein, params={"v": 1}
    )
    remeasured_rna = CountMatrix(
        counts=np.array([[1, 9], [2, 3], [4, 5]], dtype=np.int64),
        cell_ids=rna.cell_ids,
        gene_ids=rna.gene_ids,
        batch_of_cell=rna.batch_of_cell,
    )
    second_association = associate_measurements(
        remeasured_rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    second = store.save_association(
        association=second_association, rna=remeasured_rna, protein=protein, params={"v": 1}
    )
    assert second != first
    old = store.load_association(first)
    assert old.measured.shape == (4, 4)
    assert old.values[0, 1] == 0
    assert old.decisions == first_association.decisions
    assert old.to_dict()["association_decisions"] == [
        decision.to_dict() for decision in first_association.decisions
    ]
    with pytest.raises(PublishRejectedError, match="测量值"):
        store.verify_association(first, remeasured_rna, protein)


def test_withdraw_association_lists_registered_downstream_filters(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_modalities()
    association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    out = store.save_association(
        association=association, rna=rna, protein=protein, params={}
    )
    other = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="paired_only",
        manual_pairs=manual_duplicate_pair(),
    )
    other_out = store.save_association(
        association=other, rna=rna, protein=protein, params={"mode": "paired_only"}
    )
    store.register_downstream_filter(
        kind="association",
        version=out.name,
        filter_id="downstream_filter_1",
        details={"kept_rows": association.row_keys[:2]},
    )
    store.register_downstream_filter(
        kind="association",
        version=out.name,
        filter_id="downstream_filter_2",
    )
    store.register_downstream_filter(
        kind="association",
        version=other_out.name,
        filter_id="unrelated_filter",
    )
    assert store.withdraw_association(out) == [
        "downstream_filter_1",
        "downstream_filter_2",
    ]
    with pytest.raises(PublishRejectedError, match="关联已撤回"):
        store.verify_association(out, rna, protein)


def test_batch_withdrawal_reports_association_and_downstream_impact(tmp_path):
    store = ResultStore(tmp_path)
    rna, protein = make_modalities()
    association = associate_measurements(
        rna,
        protein,
        rna_source="rna_run",
        protein_source="protein_run",
        mode="all_cells",
        manual_pairs=manual_duplicate_pair(),
    )
    out = store.save_association(
        association=association, rna=rna, protein=protein, params={}
    )
    store.register_downstream_filter(
        kind="association",
        version=out.name,
        filter_id="protein_qc",
    )
    impact = store.withdraw_batch_impact("protein_batch")
    assert impact["associations"] == [out.name]
    assert impact["downstream_filters"]["association"] == {
        out.name: ["protein_qc"]
    }
