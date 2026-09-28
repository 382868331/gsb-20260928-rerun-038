from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from sc_platform import NormalizationParams, QcParams, ResultStore, build_dataset

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 12)

GENES = ["G1", "G2", "G3", "G4"]
CELLS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
# Rows are genes, columns are stable cell ids. A3 is all zero;
# batch_C is a deliberately empty sample.
MATRIX = [
    [1, 2, 0, 0, 0, 0, 0],
    [9, 8, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 5, 1, 0],
    [0, 0, 0, 0, 5, 9, 3],
]


def main() -> None:
    root = Path("results_demo")
    shutil.rmtree(root, ignore_errors=True)
    store = ResultStore(root)
    dataset = build_dataset(
        "mini_demo",
        MATRIX,
        GENES,
        CELLS,
        {"batch_A": ["A1", "A2", "A3", "A4"], "batch_B": ["B1", "B2", "B3"], "batch_C": []},
    )

    qc_rules_v1 = {
        "batch_A": QcParams(min_total_counts=5, min_detected_genes=2, min_cells_detected=2),
        "batch_B": QcParams(min_total_counts=5, min_detected_genes=1, min_cells_detected=2),
        "batch_C": QcParams(),
    }
    norm_rules = {
        "batch_A": NormalizationParams(target_sum=100),
        "batch_B": NormalizationParams(target_sum=1000),
        "batch_C": NormalizationParams(target_sum=100),
    }
    raw_dir, qc_dir, norm_dir, qc, normalized = store.process(
        dataset, qc_rules_v1, norm_rules
    )

    print("raw:", raw_dir)
    print("QC version:", qc.qc_id, qc_dir)
    print("normalized version:", normalized.norm_id, norm_dir)
    print("\nper-cell audit:")
    print(qc.cell_decisions)
    print("\nnormalized counts:")
    print(normalized.normalized_counts.round(6))
    print("\nwarnings:")
    for warning in qc.report.warnings:
        print(" -", warning)

    qc_rules_v2 = dict(qc_rules_v1)
    qc_rules_v2["batch_A"] = QcParams(min_total_counts=1, min_detected_genes=1, min_cells_detected=2)
    _, qc_dir2, norm_dir2, qc2, normalized2 = store.process(
        dataset, qc_rules_v2, norm_rules
    )
    print("\nsecond QC version (A4 now passes):", qc2.qc_id, qc_dir2)
    print("second normalized version:", normalized2.norm_id, norm_dir2)
    print("retained v1:", qc.retained_cells)
    print("retained v2:", qc2.retained_cells)
    print("\nall versions:", store.list_versions(dataset))


if __name__ == "__main__":
    main()
