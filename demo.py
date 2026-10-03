"""端到端演示：并行比较两套筛选规则，结果分版本保存且互不覆盖。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scplatform import (
    NormalizeParams,
    QCParams,
    ResultStore,
    compare_qc,
    filter_cells,
    normalize,
)
from scplatform.toydata import make_toy_matrix

RESULTS = Path("results")


def run_pipeline(
    store: ResultStore,
    run_name: str,
    qc_params: dict,
    norm_params: dict,
    comparison: dict | None = None,
) -> Path:
    matrix = make_toy_matrix()

    filtered, report = filter_cells(matrix, qc_params)
    normalized, rule_used = normalize(filtered, norm_params)

    params = {
        "run_name": run_name,
        "qc": {b: p.to_dict() for b, p in qc_params.items()},
        "normalize": {b: p.to_dict() for b, p in norm_params.items()},
    }

    out_dir = store.save_run(
        source_matrix=matrix,
        params=params,
        qc_report=report.to_dict(),
        normalized=normalized,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
        comparison=comparison,
    )

    print(f"[{run_name}] 保留细胞: {report.kept_ids}")
    print(f"[{run_name}] 剔除明细: {json.dumps(report.to_dict()['dropped'], ensure_ascii=False)}")
    print(f"[{run_name}] 全零细胞警告: {report.all_zero_cells}, 空批次警告: {report.empty_batches}")
    print(f"[{run_name}] 结果已保存 -> {out_dir}")
    return out_dir


def main() -> None:
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    store = ResultStore(RESULTS)

    qc_strict = {
        "batchA": QCParams(min_total_counts=500, min_detected_genes=3),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
    }
    qc_relaxed = {
        "batchA": QCParams(min_total_counts=10, min_detected_genes=2),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
    }
    norm_params = {
        "batchA": NormalizeParams(target_sum=1e4, log1p=True),
        "batchB": NormalizeParams(target_sum=1e3, log1p=True),
    }

    # 同一原始矩阵上并行比较两套筛选参数：哪些细胞进入/离开及原因
    comparison = compare_qc(
        make_toy_matrix(), qc_strict, qc_relaxed,
        baseline_label="strict", candidate_label="relaxed-batchA",
    )
    print("[compare] 进入(新保留):", comparison.entered_ids)
    print("[compare] 离开(新剔除):", comparison.left_ids)
    for change in comparison.changes:
        if change.direction != "unchanged":
            print(
                f"[compare] {change.cell_id} {change.direction}: "
                f"strict 原因={change.before_reasons} -> relaxed 原因={change.after_reasons}"
            )
    comparison_dict = comparison.to_dict()

    baseline_dir = run_pipeline(store, "baseline", qc_strict, norm_params,
                                comparison=comparison_dict)
    # 参数变更 -> 生成新版本，不覆盖旧结果
    run_pipeline(store, "relaxed-batchA", qc_relaxed, norm_params,
                 comparison=comparison_dict)
    # 相同输入+相同参数重跑 -> 幂等，复用已发布版本，不产生覆盖
    rerun_dir = run_pipeline(store, "baseline", qc_strict, norm_params,
                             comparison=comparison_dict)
    assert rerun_dir == baseline_dir, "相同输入与参数必须复用已发布版本"

    versions = store.list_versions()
    print(f"\n共 {len(versions)} 个版本:")
    for v in versions:
        print(" -", v.name)

    demo_joint()

    demo_multimodal()




def demo_joint() -> None:
    """跨批次联合：基因部分重叠、缺测说明、模式比较、补测出新版本。"""
    import numpy as np
    from scplatform import CountMatrix, compare_join_modes, join_batches

    print("\n=== 联合分析集演示 ===")
    sources = {
        "batchA": CountMatrix(
            counts=np.array([[600, 300, 100], [800, 100, 50]], dtype=np.int64),
            cell_ids=("cell_1", "cell_2"),
            gene_ids=("GeneA", "GeneB", "GeneC"),
            batch_of_cell=("batchA", "batchA"),
        ),
        "batchB": CountMatrix(
            counts=np.array([[40, 30, 20], [30, 20, 5]], dtype=np.int64),
            cell_ids=("cell_1", "cell_3"),  # cell_1 同名但不同源
            gene_ids=("GeneB", "GeneC", "GeneD"),
            batch_of_cell=("batchB", "batchB"),
        ),
    }
    joint = join_batches(sources, "union")
    print("[joint] 细胞身份:", joint.cell_keys)
    print("[joint] 各批次缺测基因:", joint.missing_genes_by_batch())

    qc = {"batchA": QCParams(1, 3), "batchB": QCParams(1, 3)}
    comparison = compare_join_modes(sources, qc)
    print("[joint] common 保留:", comparison.common_kept)
    print("[joint] union 保留:", comparison.union_kept)

    store = ResultStore(RESULTS)
    out = store.save_joint(joint=joint, sources=sources, params={"note": "joint-v1"})
    print("[joint] 已保存 ->", out)

    # batchB 补测 GeneA：来源指纹变化 -> 新联合版本，旧版本不覆盖
    old_b = sources["batchB"]
    sources2 = {
        "batchA": sources["batchA"],
        "batchB": CountMatrix(
            counts=np.array([[40, 30, 20, 9], [30, 20, 5, 7]], dtype=np.int64),
            cell_ids=old_b.cell_ids,
            gene_ids=("GeneB", "GeneC", "GeneD", "GeneA"),
            batch_of_cell=old_b.batch_of_cell,
        ),
    }
    out2 = store.save_joint(
        joint=join_batches(sources2, "union"), sources=sources2, params={"note": "joint-v1"}
    )
    assert out2 != out, "补测后必须生成新联合版本"
    print("[joint] 补测后新版本 ->", out2)

    # 撤回 batchB：只影响依赖它的联合结果
    affected = store.withdraw_batch("batchB")
    print("[joint] 撤回 batchB 影响版本:", affected, "状态:", store.joint_status(out2))


def demo_multimodal() -> None:
    """多模态：共享条码配对、一对多人工选择、缺测区分、范围比较、补测与撤回。"""
    import numpy as np
    from scplatform import (
        ModalityMatrix,
        ModalityQCParams,
        build_multimodal,
        compare_scopes,
        suggest_pairing,
    )

    print("\n=== 多模态细胞关联演示 ===")
    rna = ModalityMatrix(
        values=np.array([[100.0, 50.0, 0.0], [5.0, 5.0, 0.0], [9.0, 1.0, 2.0]]),
        cell_ids=("bc1", "bc2", "bc3"),
        feature_ids=("GeneA", "GeneB", "GeneC"),
        modality="rna",
        cell_display_names=("cellX", "cellX", "cellZ"),  # 显示名重复
        units=("counts", "counts", "counts"),
        measurement_version="rna-2026-09",
        batch_of_cell=("rnaBatch",) * 3,
    )
    protein = ModalityMatrix(
        values=np.array([[3.0, 0.0], [9.0, 8.0], [2.0, 4.0]]),
        cell_ids=("pA", "pB", "pC"),
        feature_ids=("CD3", "CD19"),
        modality="protein",
        cell_display_names=("cellX", "cellX-doublet?", "cellY"),
        units=("CLR", "CLR"),
        measurement_version="prot-2026-09",
        batch_of_cell=("protBatch",) * 3,
    )
    keys = {
        "rna": {c: c for c in rna.cell_ids},
        "protein": {"pA": "bc1", "pB": "bc1", "pC": "bc2"},
    }
    pairing = suggest_pairing({"rna": rna, "protein": protein}, keys)
    print("[multi] 自动配对:", [p.cell_id for p in pairing.pairs])
    print("[multi] 待人工选择:", [(g.group_id, g.candidates) for g in pairing.ambiguous])
    print("[multi] 重复显示名核对:", pairing.duplicate_display_names)
    pairing = pairing.with_resolved(
        pairing.ambiguous[0].group_id,
        {"rna": "bc1", "protein": "pA"},
        decided_by="alice",
        note="孔位影像确认 pA 对应 bc1",
    )
    multi = build_multimodal({"rna": rna, "protein": protein}, pairing)
    scope = compare_scopes(
        {"rna": rna, "protein": protein},
        pairing,
        {"rna": ModalityQCParams(10, 2), "protein": ModalityQCParams(1, 1)},
    ).to_dict()
    print("[multi] paired 范围保留:", scope["paired"]["kept_row_ids"])
    print("[multi] all 范围保留:", scope["all"]["kept_row_ids"])
    print("[multi] 缺测说明:", scope["all"]["missing_modality_cells"])

    store = ResultStore(RESULTS)
    v1 = store.save_multimodal(
        multi=multi,
        modalities={"rna": rna, "protein": protein},
        pairing=pairing,
        params={"scope": "paired"},
    )
    print("[multi] 已保存 ->", v1)
    store.register_downstream_screen(
        "tcell_screen", multimodal_version=v1, scope="paired"
    )
    # 补测 CD4：测量版本变化 -> 新联合版本，旧版本保持可解释
    protein2 = ModalityMatrix(
        values=np.array([[3.0, 0.0, 1.0], [9.0, 8.0, 1.0], [2.0, 4.0, 1.0]]),
        cell_ids=("pA", "pB", "pC"),
        feature_ids=("CD3", "CD19", "CD4"),
        modality="protein",
        units=("CLR", "CLR", "CLR"),
        measurement_version="prot-2026-10",
        batch_of_cell=("protBatch",) * 3,
    )
    mods2 = {"rna": rna, "protein": protein2}
    store.remeasure_modality("protein", pairing.modality_fingerprints["protein"], protein2)
    pairing2 = suggest_pairing(mods2, keys).with_resolved(
        "g0001:bc1", {"rna": "bc1", "protein": "pA"}, decided_by="alice"
    )
    v2 = store.save_multimodal(
        multi=build_multimodal(mods2, pairing2),
        modalities=mods2,
        pairing=pairing2,
        params={"scope": "paired"},
    )
    print("[multi] 补测后新版本 ->", v2)
    # 撤回一条关联：只影响包含它的版本与下游筛选
    pair_id = next(p.cell_id for p in pairing.pairs if p.evidence == "explicit")
    impact = store.withdraw_association(v1, pair_id, reason="复核为双胞体")
    print("[multi] 撤回关联影响:", impact["affected_downstream_screens"])
    print("[multi] v1 状态:", store.multimodal_status(v1), "v2 状态:", store.multimodal_status(v2))


if __name__ == "__main__":
    main()
