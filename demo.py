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


if __name__ == "__main__":
    main()
