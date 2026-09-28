from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import CountsDataset, QcParams, NormalizationParams
from .normalize import NormalizationResult, run_normalization
from .qc import QcResult, run_qc


def _params_json(rule) -> dict[str, Any]:
    if isinstance(rule, (QcParams, NormalizationParams)):
        return json.loads(rule.canonical())
    return {batch: json.loads(item.canonical()) for batch, item in sorted(rule.items())}


class ResultStore:
    """File-backed immutable result registry."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def raw_dir(self, dataset: CountsDataset) -> Path:
        return self.root / "raw" / f"{dataset.dataset_id}__{dataset.fingerprint()}"

    def save_raw(self, dataset: CountsDataset) -> Path:
        out = self.raw_dir(dataset)
        out.mkdir(parents=True, exist_ok=True)
        counts_path = out / "raw_counts.csv"
        if counts_path.exists():
            existing = pd.read_csv(counts_path, index_col=0)
            if not existing.equals(dataset.counts):
                raise RuntimeError(f"fingerprint collision at {out}")
        else:
            dataset.counts.to_csv(counts_path)
        manifest = {
            "dataset_id": dataset.dataset_id,
            "fingerprint": dataset.fingerprint(),
            "cell_batches": dict(dataset.cell_batches),
            "batch_cell_ids": {k: list(v) for k, v in dataset.batch_cell_ids.items()},
        }
        (out / "dataset.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out

    def save_qc(self, dataset: CountsDataset, qc: QcResult) -> Path:
        out = self.root / "qc" / dataset.fingerprint() / qc.qc_id
        if out.exists():
            return out
        out.mkdir(parents=True)
        qc.filtered_counts.to_csv(out / "filtered_counts.csv")
        qc.cell_decisions.to_csv(out / "cell_decisions.csv")
        (out / "retained_cells.txt").write_text("\n".join(qc.retained_cells), encoding="utf-8")
        provenance = {
            "qc_id": qc.qc_id,
            "dataset_id": qc.dataset_id,
            "dataset_fingerprint": qc.dataset_fingerprint,
            "params_by_batch": _params_json(qc.params_by_batch),
            "retained_cells": list(qc.retained_cells),
            "dropped_cells": list(qc.dropped_cells),
            "retained_genes": list(qc.retained_genes),
            "dropped_genes": list(qc.dropped_genes),
            "validation_warnings": list(qc.report.warnings),
        }
        (out / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out

    def save_normalization(self, qc: QcResult, normalized: NormalizationResult) -> Path:
        out = self.root / "norm" / qc.qc_id / normalized.norm_id
        if out.exists():
            return out
        out.mkdir(parents=True)
        normalized.normalized_counts.to_csv(out / "normalized_counts.csv")
        normalized.cell_factors.to_csv(out / "cell_factors.csv")
        provenance = {
            "norm_id": normalized.norm_id,
            "qc_id": normalized.qc_id,
            "dataset_id": normalized.dataset_id,
            "params_by_batch": _params_json(normalized.params_by_batch),
        }
        (out / "provenance.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return out

    def process(self, dataset, qc_params, norm_params):
        """Save raw first, then create a new versioned QC+normalization chain."""
        raw_dir = self.save_raw(dataset)
        qc_result = run_qc(dataset, qc_params)
        qc_dir = self.save_qc(dataset, qc_result)
        norm_result = run_normalization(qc_result, norm_params)
        norm_dir = self.save_normalization(qc_result, norm_result)
        return raw_dir, qc_dir, norm_dir, qc_result, norm_result

    def list_versions(self, dataset: CountsDataset) -> dict[str, list[str]]:
        fingerprint = dataset.fingerprint()
        qc_root = self.root / "qc" / fingerprint
        norm_root = self.root / "norm"
        qc_ids = sorted(p.name for p in qc_root.glob("qc_*")) if qc_root.exists() else []
        norm_ids = []
        for qc_id in qc_ids:
            batch_root = norm_root / qc_id
            if batch_root.exists():
                norm_ids.extend(sorted(p.name for p in batch_root.glob("norm_*")))
        return {"qc_versions": qc_ids, "normalization_versions": sorted(norm_ids)}
