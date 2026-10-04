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

    demo_association()


def demo_association() -> None:
    """RNA+蛋白关联：重名人工选择、缺测与零区分、撤回影响追踪。"""
    import numpy as np
    from scplatform import AmbiguousPairingError, CountMatrix, associate_assays

    print("\n=== RNA 与蛋白关联演示 ===")
    rna = {
        "rna1": CountMatrix(
            counts=np.array([[5, 0, 3], [0, 0, 0], [7, 1, 0]], dtype=np.int64),
            cell_ids=("cell_dup", "cell_zero", "cell_rna_only"),
            gene_ids=("G1", "G2", "G3"),
            batch_of_cell=("rna1",) * 3,
        )
    }
    protein = {
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
    # cell_dup 在蛋白两侧各出现一次：一对多候选，绝不自动配对
    try:
        associate_assays(rna, protein, "paired")
    except AmbiguousPairingError as exc:
        print("[assoc] 重名拒绝自动配对:", exc.unresolved)

    joint = associate_assays(
        rna, protein, "all",
        manual_links=(("rna1::cell_dup", "prot2::cell_dup"),),
    )
    print("[assoc] 行身份:", joint.row_keys)
    print("[assoc] 字段顺序:", joint.feature_ids)
    print("[assoc] 关联决定:", [l.to_dict() for l in joint.links])

    store = ResultStore(RESULTS)
    out = store.save_association(joint=joint, rna=rna, protein=protein,
                                 params={"note": "assoc-v1"})
    print("[assoc] 已保存 ->", out)
    summary = store.verify_association(out, rna, protein)
    print("[assoc] 重开核对状态:", summary["status"])

    impact = store.withdraw_association(out)
    print("[assoc] 撤回关联影响下游筛选:", impact["filters"])


if __name__ == "__main__":
    main()
