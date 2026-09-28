from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import pandas as pd

from .models import CountsDataset, QcParams


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    zero_cells: tuple[str, ...]
    empty_samples: tuple[str, ...]

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def validate_dataset(dataset: CountsDataset) -> ValidationReport:
    """Validate raw input explicitly; never drop data silently."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = dataset.counts

    if counts.index.duplicated().any():
        dups = tuple(counts.index[counts.index.duplicated(keep=False)].unique())
        errors.append(f"duplicate gene ids are not allowed: {dups}")
    if counts.columns.duplicated().any():
        dups = tuple(counts.columns[counts.columns.duplicated(keep=False)].unique())
        errors.append(f"duplicate cell ids are not allowed: {dups}")

    try:
        numeric = counts.apply(pd.api.types.is_numeric_dtype).all()
    except Exception:  # pragma: no cover - pandas API guard
        numeric = False
    if not numeric:
        errors.append("count matrix must be numeric")
    elif (counts.to_numpy() < 0).any():
        errors.append("count matrix contains negative values")

    missing = [cell for cell in counts.columns if cell not in dataset.cell_batches]
    if missing:
        errors.append(f"cells without batch assignment: {tuple(missing)}")
    unknown_assignments = sorted(set(dataset.cell_batches) - set(counts.columns))
    if unknown_assignments:
        errors.append(f"batch assignments for missing cells: {tuple(unknown_assignments)}")

    declared = dataset.batch_cell_ids
    empty_samples = tuple(sorted(batch for batch, cells in declared.items() if len(cells) == 0))
    for batch in empty_samples:
        warnings.append(f"sample/batch '{batch}' contains zero cells")
    declared_cells = {cell for cells in declared.values() for cell in cells}
    undeclared = sorted(set(counts.columns) - declared_cells) if declared else []
    if declared and undeclared:
        errors.append(f"cells missing from batch manifests: {tuple(undeclared)}")
    for batch, cells in declared.items():
        if len(cells) != len(set(cells)):
            errors.append(f"batch manifest '{batch}' lists a cell more than once")
        mismatched = sorted(cell for cell in cells if dataset.cell_batches.get(cell) != batch)
        if mismatched:
            errors.append(f"batch manifest '{batch}' has mismatched cells: {tuple(mismatched)}")

    totals = counts.sum(axis=0)
    zero_cells = tuple(totals.index[totals == 0])
    for cell in zero_cells:
        warnings.append(f"all-zero cell retained in raw data and flagged: {cell}")

    return ValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        zero_cells=zero_cells,
        empty_samples=empty_samples,
    )


@dataclass(frozen=True)
class QcResult:
    qc_id: str
    dataset_id: str
    dataset_fingerprint: str
    params_by_batch: Mapping[str, QcParams]
    filtered_counts: pd.DataFrame
    retained_cells: tuple[str, ...]
    dropped_cells: tuple[str, ...]
    retained_genes: tuple[str, ...]
    dropped_genes: tuple[str, ...]
    cell_decisions: pd.DataFrame
    report: ValidationReport

    def counts_for(self, batch: str) -> pd.DataFrame:
        cells = [c for c in self.retained_cells if self.cell_decisions.at[c, "batch"] == batch]
        return self.filtered_counts[cells]


def _as_batch_params(dataset: CountsDataset, params):
    if isinstance(params, QcParams):
        batches = sorted(
            set(dataset.batch_cell_ids)
            | {dataset.cell_batches[cell] for cell in dataset.cell_ids}
        )
        return {batch: params for batch in batches}
    return dict(params)


def run_qc(dataset: CountsDataset, params: QcParams | Mapping[str, QcParams]) -> QcResult:
    report = validate_dataset(dataset)
    report.raise_if_invalid()

    params_by_batch = _as_batch_params(dataset, params)
    counts = dataset.counts
    totals = counts.sum(axis=0)
    detected_genes = (counts > 0).sum(axis=0)

    decision_rows: dict[str, dict] = {}
    retained_cells: list[str] = []
    dropped_cells: list[str] = []
    for cell in counts.columns:
        batch = dataset.cell_batches[cell]
        rule = params_by_batch.get(batch)
        if rule is None:
            raise KeyError(f"no QC parameters supplied for batch '{batch}'")
        reasons = []
        if totals[cell] < rule.min_total_counts:
            reasons.append("min_total_counts")
        if detected_genes[cell] < rule.min_detected_genes:
            reasons.append("min_detected_genes")
        keep = not reasons
        decision_rows[cell] = {
            "batch": batch,
            "total_counts": int(totals[cell]),
            "detected_genes": int(detected_genes[cell]),
            "retained": keep,
            "reasons": ",".join(reasons),
        }
        (retained_cells if keep else dropped_cells).append(cell)

    # A gene survives when it passes its own batch's threshold in at least
    # one batch; results from the two batches are never pooled for QC.
    retained_matrix = counts[retained_cells]
    gene_pass = pd.Series(False, index=counts.index)
    for batch, rule in params_by_batch.items():
        batch_cells = [cell for cell in retained_cells if decision_rows[cell]["batch"] == batch]
        if not batch_cells:
            continue
        batch_detected = (retained_matrix[batch_cells] > 0).sum(axis=1)
        gene_pass |= batch_detected >= rule.min_cells_detected
    gene_reasons = {gene: ([] if gene_pass[gene] else ["min_cells_detected"]) for gene in counts.index}
    retained_genes = tuple(gene for gene, reasons in gene_reasons.items() if not reasons)
    dropped_genes = tuple(gene for gene, reasons in gene_reasons.items() if reasons)

    decisions = pd.DataFrame.from_dict(decision_rows, orient="index")
    decisions.index.name = "cell_id"
    filtered = counts.loc[list(retained_genes), retained_cells].copy()

    payload = dataset.fingerprint() + "|" + json.dumps(
        {batch: rule.canonical() for batch, rule in sorted(params_by_batch.items())},
        sort_keys=True,
        ensure_ascii=False,
    )
    qc_id = "qc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    return QcResult(
        qc_id=qc_id,
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=dataset.fingerprint(),
        params_by_batch=dict(params_by_batch),
        filtered_counts=filtered,
        retained_cells=tuple(retained_cells),
        dropped_cells=tuple(dropped_cells),
        retained_genes=retained_genes,
        dropped_genes=dropped_genes,
        cell_decisions=decisions,
        report=report,
    )
