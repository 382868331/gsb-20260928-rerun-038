"""多模态细胞关联行为核对：配对依据、缺测区分、范围比较、撤回与版本持久化。"""

import dataclasses

import numpy as np
import pytest

from scplatform import (
    ModalityMatrix,
    ModalityQCParams,
    PublishRejectedError,
    ResultStore,
    build_multimodal,
    compare_scopes,
    suggest_pairing,
)


def make_modalities():
    rna = ModalityMatrix(
        values=np.array([[100.0, 50.0, 0.0], [5.0, 5.0, 0.0], [9.0, 1.0, 2.0]]),
        cell_ids=("bc1", "bc2", "bc3"),
        feature_ids=("G1", "G2", "G3"),
        modality="rna",
        cell_display_names=("cellX", "cellX", "cellZ"),  # 重复显示名
        units=("counts", "counts", "counts"),
        measurement_version="rna-v1",
        batch_of_cell=("b1", "b1", "b1"),
    )
    protein = ModalityMatrix(
        values=np.array([[3.0, 0.0], [9.0, 8.0], [2.0, 4.0]]),
        cell_ids=("pA", "pB", "pC"),
        feature_ids=("CD3", "CD19"),
        modality="protein",
        cell_display_names=("cellX", "cellX-doublet", "cellY"),
        units=("CLR", "CLR"),
        measurement_version="prot-v1",
        batch_of_cell=("p1", "p1", "p1"),
    )
    keys = {
        "rna": {c: c for c in rna.cell_ids},
        "protein": {"pA": "bc1", "pB": "bc1", "pC": "bc2"},
    }
    return {"rna": rna, "protein": protein}, keys


def resolve_first(pairing):
    return pairing.with_resolved(
        pairing.ambiguous[0].group_id,
        {"rna": "bc1", "protein": "pA"},
        decided_by="me",
    )


def test_shared_key_pairs_and_duplicate_display_names_not_merged():
    mods, keys = make_modalities()
    pairing = suggest_pairing(mods, keys)
    assert pairing.duplicate_display_names == {"rna": ["cellX"], "protein": []}
    auto = [p for p in pairing.pairs if p.evidence == "shared_key"]
    assert [p.cell_id for p in auto] == ["protein::pC|rna::bc2"]
    assert [g.shared_key for g in pairing.ambiguous] == ["bc1"]
    assert pairing.ambiguous[0].candidates["protein"] == ["pA", "pB"]


def test_one_to_many_requires_manual_choice_with_snapshot():
    mods, keys = make_modalities()
    pairing = suggest_pairing(mods, keys)
    group = pairing.ambiguous[0].group_id
    with pytest.raises(ValueError, match="不在候选组"):
        pairing.with_resolved(group, {"rna": "bc1", "protein": "pC"}, decided_by="me")
    resolved = resolve_first(pairing)
    record = next(p for p in resolved.pairs if p.evidence == "explicit")
    assert record.modality_cells == {"rna": "bc1", "protein": "pA"}
    assert record.decided_by == "me"
    assert record.candidates["protein"] == ["pA", "pB"]
    leftover = [u for u in resolved.unpaired if u["cell_id"] == "pB"]
    assert leftover and leftover[0]["reason"] == "ambiguous"
    assert len(pairing.pairs) == 1  # 旧决定未被改写


def test_true_zero_distinct_from_missing_modality():
    mods, keys = make_modalities()
    pairing = resolve_first(suggest_pairing(mods, keys))
    multi = build_multimodal(mods, pairing)
    rows = {r.row_id: i for i, r in enumerate(multi.rows)}
    paired = rows["pair:protein::pA|rna::bc1"]
    cd19 = multi.feature_ids["protein"].index("CD19")
    assert multi.values["protein"][paired, cd19] == 0
    assert multi.measured["protein"][paired, cd19]
    unpaired = rows["unpaired:rna::bc3"]
    assert np.isnan(multi.values["protein"][unpaired]).all()
    assert not multi.measured["protein"][unpaired].any()
    assert multi.feature_ids == {
        "rna": ("G1", "G2", "G3"), "protein": ("CD3", "CD19"),
    }
    assert multi.units["protein"] == ("CLR", "CLR")
    assert multi.measurement_versions == {"rna": "rna-v1", "protein": "prot-v1"}


def test_scope_comparison_marks_missing_and_shows_screening_impact():
    mods, keys = make_modalities()
    pairing = resolve_first(suggest_pairing(mods, keys))
    qc = {
        "rna": ModalityQCParams(min_total_signal=10, min_detected_features=2),
        "protein": ModalityQCParams(min_total_signal=1, min_detected_features=1),
    }
    report = compare_scopes(mods, pairing, qc).to_dict()
    assert report["paired_only_row_ids"] == []
    assert set(report["all_only_row_ids"]) == {
        "unpaired:protein::pB", "unpaired:rna::bc3",
    }
    missing = {(x["row_id"], x["modality"])
               for x in report["all"]["missing_modality_cells"]}
    assert ("unpaired:rna::bc3", "protein") in missing
    assert "unpaired:rna::bc3" not in report["paired"]["kept_row_ids"]
    assert "unpaired:rna::bc3" in report["all"]["kept_row_ids"]


def test_save_idempotent_tamper_rejected_and_reopen(tmp_path):
    mods, keys = make_modalities()
    pairing = resolve_first(suggest_pairing(mods, keys))
    multi = build_multimodal(mods, pairing)
    store = ResultStore(tmp_path)
    out = store.save_multimodal(multi=multi, modalities=mods, pairing=pairing, params={"x": 1})
    again = store.save_multimodal(multi=multi, modalities=mods, pairing=pairing, params={"x": 1})
    assert out == again

    forged = dataclasses.replace(
        pairing,
        modality_fingerprints={
            **pairing.modality_fingerprints, "rna": "deadbeef",
        },
    )
    with pytest.raises(PublishRejectedError):
        store.save_multimodal(multi=multi, modalities=mods, pairing=forged, params={})
    assert len(store.list_multimodal()) == 1

    store2 = ResultStore(tmp_path)
    summary = store2.verify_multimodal(out, mods)
    assert summary["status"] == "ok"
    reopened = store2.load_multimodal(out)
    assert reopened.feature_ids["protein"] == ("CD3", "CD19")
    assert reopened.units["protein"] == ("CLR", "CLR")
    evidence = {p.evidence for p in store2.load_pairing(out).pairs}
    assert evidence == {"shared_key", "explicit"}


def test_withdraw_association_only_affects_dependent_version_and_screens(tmp_path):
    mods, keys = make_modalities()
    pairing = resolve_first(suggest_pairing(mods, keys))
    multi = build_multimodal(mods, pairing)
    store = ResultStore(tmp_path)
    v1 = store.save_multimodal(multi=multi, modalities=mods, pairing=pairing, params={"v": 1})
    v2 = store.save_multimodal(multi=multi, modalities=mods, pairing=pairing, params={"v": 2})
    store.register_downstream_screen("screen_on_v2", multimodal_version=v2, scope="paired")
    explicit_id = next(p.cell_id for p in pairing.pairs if p.evidence == "explicit")
    impact = store.withdraw_association(v2, explicit_id, reason="污染")
    assert [s["screen"] for s in impact["affected_downstream_screens"]] == ["screen_on_v2"]
    assert store.multimodal_status(v2).startswith("withdrawn_link")
    assert store.multimodal_status(v1) == "ok"
    with pytest.raises(PublishRejectedError, match="撤回"):
        store.verify_multimodal(v2, mods)
    assert store.verify_multimodal(v1, mods)["status"] == "ok"


def test_remeasure_makes_new_version_old_remains_interpretable(tmp_path):
    mods, keys = make_modalities()
    pairing = resolve_first(suggest_pairing(mods, keys))
    multi = build_multimodal(mods, pairing)
    store = ResultStore(tmp_path)
    old_dir = store.save_multimodal(multi=multi, modalities=mods, pairing=pairing, params={})

    protein2 = ModalityMatrix(
        values=np.array([[3.0, 0.0, 1.0], [9.0, 8.0, 1.0], [2.0, 4.0, 1.0]]),
        cell_ids=("pA", "pB", "pC"),
        feature_ids=("CD3", "CD19", "CD4"),
        modality="protein",
        units=("CLR", "CLR", "CLR"),
        measurement_version="prot-v2",
        batch_of_cell=("p1", "p1", "p1"),
    )
    mods2 = {"rna": mods["rna"], "protein": protein2}
    store.remeasure_modality("protein", pairing.modality_fingerprints["protein"], protein2)
    pairing2 = resolve_first(suggest_pairing(mods2, keys))
    multi2 = build_multimodal(mods2, pairing2)
    new_dir = store.save_multimodal(multi=multi2, modalities=mods2, pairing=pairing2, params={})
    assert new_dir != old_dir
    reopened = store.load_multimodal(old_dir)
    assert reopened.measurement_versions["protein"] == "prot-v1"
    assert reopened.feature_ids["protein"] == ("CD3", "CD19")
    impact = store.withdraw_modality("protein")
    affected = {v["version"] for v in impact["affected_multimodal_versions"]}
    assert {old_dir.name, new_dir.name} <= affected
