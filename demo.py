"""端到端演示。

流程：造数 -> 从 CSV 载入（带来源文件名）-> 两套筛选参数并行比较 ->
严格/宽松规则各跑一次 -> 结果分版本保存 -> 发布基线版本。

重复运行不会覆盖任何已有版本；相同输入+相同参数会幂等返回同一版本目录。
"""

from __future__ import annotations

import json
from pathlib import Path

from scplatform import (
    NormalizeParams,
    QCParams,
    ResultStore,
    compare_filters,
    filter_cells,
    load_matrix_csv,
    normalize,
    write_matrix_csv,
)
from scplatform.toydata import make_toy_matrix

RESULTS = Path("results")
INPUTS = Path("example_data")


def run_pipeline(store: ResultStore, run_name: str, qc_strict_a: bool) -> Path:
    source = INPUTS / "toy_matrix.csv"
    matrix = load_matrix_csv(source)

    qc_params = {
        "batchA": QCParams(
            min_total_counts=500 if qc_strict_a else 10,
            min_detected_genes=3 if qc_strict_a else 2,
        ),
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

    out_dir = store.save_run(
        raw_matrix=matrix,
        source_name=source.name,
        params=params,
        qc_report=report.to_dict(),
        normalized=normalized,
        cell_ids=list(filtered.cell_ids),
        gene_ids=list(filtered.gene_ids),
        batch_of_cell=list(filtered.batch_of_cell),
        rule_used=rule_used,
    )

    print(f"[{run_name}] 输入指纹: {matrix.fingerprint()[:12]}")
    print(f"[{run_name}] 保留细胞: {report.kept_ids}")
    print(f"[{run_name}] 剔除明细: {json.dumps(report.to_dict()['dropped'], ensure_ascii=False)}")
    print(f"[{run_name}] 全零细胞警告: {report.all_zero_cells}, 空批次警告: {report.empty_batches}")
    print(f"[{run_name}] 结果已保存 -> {out_dir}")
    return out_dir


def main() -> None:
    INPUTS.mkdir(exist_ok=True)
    write_matrix_csv(INPUTS / "toy_matrix.csv", make_toy_matrix())
    store = ResultStore(RESULTS)

    strict = {
        "batchA": QCParams(min_total_counts=500, min_detected_genes=3),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
    }
    relaxed = {
        "batchA": QCParams(min_total_counts=10, min_detected_genes=2),
        "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
    }
    comparison = compare_filters(
        load_matrix_csv(INPUTS / "toy_matrix.csv"),
        strict,
        relaxed,
        label_before="strict",
        label_after="relaxed",
    )
    print(comparison.format_text())
    (RESULTS).mkdir(exist_ok=True)
    (RESULTS / "qc_comparison.json").write_text(
        json.dumps(comparison.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()

    baseline_dir = run_pipeline(store, "baseline", qc_strict_a=True)
    store.publish(baseline_dir)
    print(f"[baseline] 已发布: {baseline_dir}")
    run_pipeline(store, "relaxed-batchA", qc_strict_a=False)

    versions = store.list_versions()
    print(f"\n共 {len(versions)} 个版本（历史结果不删除、不覆盖）:")
    for version in versions:
        tag = " [已发布]" if store.is_published(version) else ""
        print(" -", version.name, tag)


if __name__ == "__main__":
    main()
