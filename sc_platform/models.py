from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class QcParams:
    """Cell/gene filtering rules."""

    min_total_counts: int = 1
    min_detected_genes: int = 1
    min_cells_detected: int = 1

    def canonical(self) -> str:
        return json.dumps(
            {
                "min_total_counts": self.min_total_counts,
                "min_detected_genes": self.min_detected_genes,
                "min_cells_detected": self.min_cells_detected,
            },
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class NormalizationParams:
    target_sum: float = 1e4
    log1p: bool = False

    def canonical(self) -> str:
        return json.dumps(
            {"target_sum": self.target_sum, "log1p": self.log1p},
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass(frozen=True)
class CountsDataset:
    """Raw count matrix with stable cell, gene and batch identities.

    Rows are stable gene ids; columns are stable cell ids.
    Raw data is never modified in place.
    """

    dataset_id: str
    counts: pd.DataFrame
    cell_batches: Mapping[str, str]
    batch_cell_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def gene_ids(self) -> tuple[str, ...]:
        return tuple(self.counts.index)

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(self.counts.columns)

    def fingerprint(self) -> str:
        payload = {
            "genes": list(self.gene_ids),
            "cells": list(self.cell_ids),
            "batches": {cell: self.cell_batches[cell] for cell in self.cell_ids},
            "values": self.counts.to_numpy().astype(int).tolist(),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]
