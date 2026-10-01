"""端到端演示：并行比较两套筛选规则，结果分版本保存且互不覆盖。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scplatform import (
    NormalizeParams,
    QCParams,
    ResultStore,
    compare_join_modes,
    compare_qc,
    filter_cells,
    join_batches,
    normalize,
    qc_on_joint,
)
from scplatform import CountMatrix
from scplatform.toydata import make_toy_matrix
import numpy as np

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

    demo_joint(store)


def demo_joint(store: ResultStore) -> None:
    """联合分析集演示：基因集合不同的批次联合、模式比较、补测与撤回。"""
    print("\n=== 联合分析集 ===")
    batches = {
        "batchA": CountMatrix(
            counts=np.array([[600, 300, 100], [10, 5, 0]], dtype=np.int64),
            cell_ids=("cell_a1", "cell_a2"),
            gene_ids=("GeneA", "GeneB", "GeneC"),
            batch_of_cell=("batchA", "batchA"),
        ),
        "batchB": CountMatrix(
            # 与 batchA 基因集合重叠但不同（无 GeneA，多 GeneD），
            # 且有同显示名细胞 cell_a1 —— 按来源区分，绝不合并
            counts=np.array([[40, 30, 20, 10], [30, 20, 0, 0]], dtype=np.int64),
            cell_ids=("cell_a1", "cell_b2"),
            gene_ids=("GeneB", "GeneC", "GeneD", "GeneE"),
            batch_of_cell=("batchB", "batchB"),
        ),
    }
    qc = {
        "batchA": QCParams(min_total_counts=50, min_detected_genes=3),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=3),
    }

    # 两种联合选择对保留细胞的影响
    comparison = compare_join_modes(batches, qc)
    print("[joint] 共同基因数:", comparison.n_genes_intersection,
          "全部基因数:", comparison.n_genes_union)
    print("[joint] 各批次缺测基因:", comparison.missing_by_batch)
    print("[joint] 仅 union 保留:", comparison.entered_ids)
    print("[joint] 仅 intersection 保留:", comparison.left_ids)

    # 保存 union 联合版本：缺测为 NaN + observed=False，绝不当作零表达
    joint = join_batches(batches, mode="union")
    report = qc_on_joint(joint, qc)
    out = store.save_joint(
        sources=batches,
        joint=joint,
        params={"qc": {b: p.to_dict() for b, p in qc.items()}},
        qc_report=report.to_dict(),
        comparison=comparison.to_dict(),
    )
    print("[joint] 联合版本已保存 ->", out.name)
    info = store.verify_joint(out, batches)
    print("[joint] 重开核对:", info["status"], "缺测说明:", info["missing_by_batch"])

    # batchB 补测 GeneA -> 新联合版本，旧版本保留
    old = batches["batchB"]
    batches_v2 = dict(batches)
    batches_v2["batchB"] = CountMatrix(
        counts=np.column_stack([old.counts, [5, 0]]),
        cell_ids=old.cell_ids,
        gene_ids=old.gene_ids + ("GeneA",),
        batch_of_cell=old.batch_of_cell,
    )
    joint_v2 = join_batches(batches_v2, mode="union")
    out_v2 = store.save_joint(sources=batches_v2, joint=joint_v2,
                              params={"qc": {b: p.to_dict() for b, p in qc.items()}})
    print("[joint] 补测后生成新版本 ->", out_v2.name, "（旧版本保留）")

    # 撤回 batchA：只影响依赖它的联合结果
    affected = store.withdraw_batch("batchA")
    print("[joint] 撤回 batchA，受影响版本:", affected)
    print("[joint] 旧版本状态:", store.verify_joint(out, batches)["status"])


if __name__ == "__main__":
    main()
