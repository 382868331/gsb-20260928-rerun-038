from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sc_platform import (
    NormalizationParams,
    QcParams,
    ResultStore,
    build_dataset,
    run_normalization,
    run_qc,
    validate_dataset,
)

GENES = ["G1", "G2", "G3", "G4"]
CELLS = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
MATRIX = [
    [1, 2, 0, 0, 0, 0, 0],
    [9, 8, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 5, 1, 0],
    [0, 0, 0, 0, 5, 9, 3],
]
BATCHES = {"batch_A": CELLS[:4], "batch_B": CELLS[4:], "batch_C": []}


@pytest.fixture()
def dataset():
    return build_dataset("mini", MATRIX, GENES, CELLS, BATCHES)


def test_empty_sample_and_zero_cells_are_reported_not_swallowed(dataset):
    report = validate_dataset(dataset)
    assert report.empty_samples == ("batch_C",)
    assert report.zero_cells == ("A3",)
    assert any("batch_C" in item for item in report.warnings)
    assert any("A3" in item for item in report.warnings)
    assert set(dataset.cell_ids) == set(CELLS)


def test_duplicate_genes_are_hard_errors():
    dataset = build_dataset(
        "dup",
        [[1, 2], [3, 4]],
        ["G1", "G1"],
        ["C1", "C2"],
        {"x": ["C1", "C2"]},
    )
    with pytest.raises(ValueError, match="duplicate gene"):
        run_qc(dataset, QcParams())


def test_qc_counts_and_cell_traceability(dataset):
    rules = {
        "batch_A": QcParams(min_total_counts=5, min_detected_genes=2, min_cells_detected=2),
        "batch_B": QcParams(min_total_counts=5, min_detected_genes=1, min_cells_detected=2),
        "batch_C": QcParams(),
    }
    result = run_qc(dataset, rules)
    assert result.retained_cells == ("A1", "A2", "B1", "B2")
    assert result.dropped_cells == ("A3", "A4", "B3")
    assert result.retained_genes == ("G1", "G2", "G3", "G4")
    decisions = result.cell_decisions
    assert decisions.loc["A3", "reasons"] == "min_total_counts,min_detected_genes"
    assert bool(decisions.loc["A1", "retained"])
    assert not bool(decisions.loc["B3", "retained"])


def test_batch_specific_normalization_values_and_isolation(dataset):
    rules = {
        "batch_A": QcParams(min_total_counts=5, min_detected_genes=2),
        "batch_B": QcParams(min_total_counts=5, min_detected_genes=1),
        "batch_C": QcParams(),
    }
    norms = {
        "batch_A": NormalizationParams(target_sum=100),
        "batch_B": NormalizationParams(target_sum=1000),
        "batch_C": NormalizationParams(target_sum=100),
    }
    qc = run_qc(dataset, rules)
    result = run_normalization(qc, norms)
    values = result.normalized_counts
    np.testing.assert_allclose(values["A1"].sum(), 100)
    np.testing.assert_allclose(values["B1"].sum(), 1000)
    np.testing.assert_allclose(values.loc["G2", "A1"], 90.0)
    np.testing.assert_allclose(values.loc["G3", "B1"], 500.0)
    assert result.cell_factors.loc["A1", "batch"] == "batch_A"
    assert result.cell_factors.loc["B1", "batch"] == "batch_B"
    batch_a_values = values.loc[["G3", "G4"], ["A1", "A2"]]
    assert batch_a_values.eq(0).all().all()


def test_parameter_change_creates_new_version_without_touching_raw(tmp_path, dataset):
    store = ResultStore(tmp_path)
    v1_rules = {
        "batch_A": QcParams(min_total_counts=5, min_detected_genes=2),
        "batch_B": QcParams(min_total_counts=5, min_detected_genes=1),
        "batch_C": QcParams(),
    }
    v2_rules = dict(v1_rules)
    v2_rules["batch_A"] = QcParams(min_total_counts=1, min_detected_genes=1)
    norms = {
        "batch_A": NormalizationParams(100),
        "batch_B": NormalizationParams(1000),
        "batch_C": NormalizationParams(100),
    }
    _, qc_dir1, _, qc1, _ = store.process(dataset, v1_rules, norms)
    _, qc_dir2, _, qc2, _ = store.process(dataset, v2_rules, norms)
    assert qc_dir1 != qc_dir2
    assert qc1.qc_id != qc2.qc_id
    assert "A4" not in qc1.retained_cells
    assert "A4" in qc2.retained_cells
    raw = pd.read_csv(store.raw_dir(dataset) / "raw_counts.csv", index_col=0)
    assert list(raw.columns) == CELLS
    provenance = json.loads((qc_dir1 / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["dropped_cells"] == ["A3", "A4", "B3"]
    versions = store.list_versions(dataset)
    assert len(versions["qc_versions"]) == 2
    assert len(versions["normalization_versions"]) == 2


def test_demo_runs_and_reports_versions():
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "demo.py"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "second QC version" in completed.stdout
    assert "batch_C" in completed.stdout
    assert (repo_root / "results_demo" / "raw").exists()
