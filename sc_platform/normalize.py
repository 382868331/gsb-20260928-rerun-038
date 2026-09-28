from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

from .models import NormalizationParams
from .qc import QcResult


@dataclass(frozen=True)
class NormalizationResult:
    norm_id: str
    qc_id: str
    dataset_id: str
    params_by_batch: Mapping[str, NormalizationParams]
    normalized_counts: pd.DataFrame
    cell_factors: pd.DataFrame

    def values_for(self, batch: str) -> pd.DataFrame:
        cells = list(self.cell_factors[self.cell_factors["batch"] == batch].index)
        return self.normalized_counts[cells]


def _as_batch_params(qc: QcResult, params):
    if isinstance(params, NormalizationParams):
        return {batch: params for batch in sorted(qc.params_by_batch)}
    missing = sorted(set(qc.params_by_batch) - set(params))
    if missing:
        raise KeyError(f"no normalization parameters supplied for batches {tuple(missing)}")
    return {batch: params[batch] for batch in sorted(qc.params_by_batch)}


def run_normalization(
    qc: QcResult, params: NormalizationParams | Mapping[str, NormalizationParams]
) -> NormalizationResult:
    params_by_batch = _as_batch_params(qc, params)
    counts = qc.filtered_counts.astype(float)
    normalized = pd.DataFrame(np.nan, index=counts.index, columns=counts.columns, dtype=float)
    factor_rows: dict[str, dict] = {}

    for batch, rule in params_by_batch.items():
        cells = [c for c in qc.retained_cells if qc.cell_decisions.at[c, "batch"] == batch]
        if not cells:
            continue
        totals = counts[cells].sum(axis=0)
        if (totals <= 0).any():
            bad = tuple(totals.index[totals <= 0])
            raise ValueError(f"cannot normalize zero-total cells: {bad}")
        scale = rule.target_sum / totals
        batch_norm = counts[cells].multiply(scale, axis=1)
        if rule.log1p:
            batch_norm = np.log1p(batch_norm)
        normalized[cells] = batch_norm
        for cell in cells:
            factor_rows[cell] = {
                "batch": batch,
                "total_counts": float(totals[cell]),
                "scale_factor": float(scale[cell]),
                "target_sum": rule.target_sum,
                "log1p": rule.log1p,
            }

    factors = pd.DataFrame.from_dict(factor_rows, orient="index")
    factors.index.name = "cell_id"
    payload = qc.qc_id + "|" + json.dumps(
        {batch: rule.canonical() for batch, rule in params_by_batch.items()},
        sort_keys=True,
        ensure_ascii=False,
    )
    norm_id = "norm_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]

    return NormalizationResult(
        norm_id=norm_id,
        qc_id=qc.qc_id,
        dataset_id=qc.dataset_id,
        params_by_batch=params_by_batch,
        normalized_counts=normalized,
        cell_factors=factors,
    )
