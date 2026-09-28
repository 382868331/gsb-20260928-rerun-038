"""端到端演示：两个批次用不同规则跑完整流程，结果分版本保存。"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scplatform import (
    NormalizeParams,
    QCParams,
    ResultStore,
    filter_cells,
    normalize,
)
from scplatform.toydata import make_toy_matrix

RESULTS = Path("results")


def run_pipeline(store: ResultStore, run_name: str, qc_strict_a: bool) -> Path:
    matrix = make_toy_matrix()

    # 两个批次使用不同规则：batchA 严格，batchB 宽松
    qc_params = {
        "batchA": QCParams(min_total_counts=500 if qc_strict_a else 100,
                           min_detected_genes=3),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
    }
    norm_params = {
        "batchA": NormalizeParams(target_sum=1e4, log1p=True),
        "batchB": NormalizeParams(target_sum=1e3, log1p=True),
    }

    filtered, report = filter_cells(matrix, qc_params)
    normalized, rule_used = normalize(filtered, norm_params)

    params = {
        "run_name": run_name,
        "qc": {b: p.to_dict() for b, p in qc_params.items()},
        "normalize": {b: p.to_dict() for b, p in norm_params.items()},
    }
    fingerprint = hashlib.sha256(matrix.counts.tobytes()).hexdigest()[:10]

    out_dir = store.save_run(
        params=params,
        qc_report=report.to_dict(),
        normalized=normalized,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
        input_fingerprint=fingerprint,
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

    run_pipeline(store, "baseline", qc_strict_a=True)
    # 参数变更 -> 生成新版本，不覆盖旧结果
    run_pipeline(store, "relaxed-batchA", qc_strict_a=False)

    versions = store.list_versions()
    print(f"\n共 {len(versions)} 个版本:")
    for v in versions:
        print(" -", v.name)


if __name__ == "__main__":
    main()
