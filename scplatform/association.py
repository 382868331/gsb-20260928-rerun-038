"""RNA 与蛋白两种测量的细胞级关联。

显示名只用于生成候选，从不直接作为内部身份。重复显示名或一对多候选必须
由调用方用 "来源::细胞ID" 明确选择；未配对细胞可在 all 模式中保留，其
缺失的是整种测量，不能写成该测量中的零表达。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint import cell_key
from .model import CountMatrix

RNA = "rna"
PROTEIN = "protein"
MEASUREMENTS = (RNA, PROTEIN)
ASSOCIATION_MODES = ("paired_only", "all_cells")


@dataclass(frozen=True)
class CellCandidate:
    display_name: str
    rna_cells: tuple[str, ...]
    protein_cells: tuple[str, ...]

    @property
    def is_unique(self) -> bool:
        return len(self.rna_cells) == 1 and len(self.protein_cells) == 1

    @property
    def status(self) -> str:
        if self.rna_cells and self.protein_cells:
            return "ambiguous"
        return "unpaired"

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "status": self.status,
            "rna_cells": list(self.rna_cells),
            "protein_cells": list(self.protein_cells),
        }


@dataclass(frozen=True)
class AssociationDecision:
    display_name: str
    rna_cell: str
    protein_cell: str
    basis: str

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "rna_cell": self.rna_cell,
            "protein_cell": self.protein_cell,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class MeasurementAssociation:
    values: np.ndarray
    measured: np.ndarray
    row_keys: tuple[str, ...]
    display_names: tuple[str, ...]
    rna_cell_keys: tuple[str | None, ...]
    protein_cell_keys: tuple[str | None, ...]
    fields: tuple[tuple[str, str], ...]
    mode: str
    decisions: tuple[AssociationDecision, ...]
    rna_source: str
    protein_source: str
    source_fingerprints: dict[str, str]
    batches: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        measured = np.asarray(self.measured, dtype=bool)
        if values.shape != measured.shape:
            raise ValueError("关联 values 与 measured 形状不一致")
        n_rows, n_fields = values.shape
        if self.mode not in ASSOCIATION_MODES:
            raise ValueError(f"未知关联范围: {self.mode!r}，可选 {ASSOCIATION_MODES}")
        if len(self.row_keys) != n_rows:
            raise ValueError("行身份数量与 values 行数不一致")
        if len(self.fields) != n_fields:
            raise ValueError("字段数量与 values 列数不一致")
        for sequence, name in [
            (self.display_names, "display_names"),
            (self.rna_cell_keys, "rna_cell_keys"),
            (self.protein_cell_keys, "protein_cell_keys"),
        ]:
            if len(sequence) != n_rows:
                raise ValueError(f"{name} 数量与 values 行数不一致")
        if len(set(self.row_keys)) != n_rows:
            raise ValueError("关联结果含重复行身份")
        if any(m not in MEASUREMENTS for m, _ in self.fields):
            raise ValueError("字段测量类型只能是 rna 或 protein")
        if len(set(self.fields)) != n_fields:
            raise ValueError("关联结果含重复字段身份")
        rna_fields = tuple(f for f in self.fields if f[0] == RNA)
        protein_fields = tuple(f for f in self.fields if f[0] == PROTEIN)
        if self.fields != rna_fields + protein_fields:
            raise ValueError("字段顺序必须保持为 RNA 字段后接蛋白字段")
        if np.isnan(values[measured]).any():
            raise ValueError("已测量位置不能为 NaN")
        if not np.isnan(values[~measured]).all():
            raise ValueError("整种测量缺测位置必须为 NaN，不能冒充零")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "measured", measured)

    @property
    def n_rows(self) -> int:
        return self.values.shape[0]

    @property
    def n_fields(self) -> int:
        return self.values.shape[1]

    def rna_field_count(self) -> int:
        return sum(1 for measurement, _ in self.fields if measurement == RNA)

    def missing_measurement_by_row(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        rna_cols = self.rna_field_count()
        for i, row in enumerate(self.row_keys):
            missing: list[str] = []
            if not self.measured[i, :rna_cols].any():
                missing.append(RNA)
            if not self.measured[i, rna_cols:].any():
                missing.append(PROTEIN)
            if missing:
                result[row] = missing
        return result

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "rna_source": self.rna_source,
            "protein_source": self.protein_source,
            "n_rows": self.n_rows,
            "fields": [[measurement, feature] for measurement, feature in self.fields],
            "source_fingerprints": dict(self.source_fingerprints),
            "batches": list(self.batches),
            "association_decisions": [d.to_dict() for d in self.decisions],
            "missing_measurement_by_row": self.missing_measurement_by_row(),
            "rows": [
                {
                    "row_key": self.row_keys[i],
                    "display_name": self.display_names[i],
                    "rna_cell": self.rna_cell_keys[i],
                    "protein_cell": self.protein_cell_keys[i],
                }
                for i in range(self.n_rows)
            ],
        }


def association_candidates(
    rna: CountMatrix,
    protein: CountMatrix,
    *,
    rna_source: str,
    protein_source: str,
) -> list[CellCandidate]:
    by_name_rna: dict[str, list[str]] = {}
    by_name_protein: dict[str, list[str]] = {}
    for cell in rna.cell_ids:
        by_name_rna.setdefault(cell, []).append(cell_key(rna_source, cell))
    for cell in protein.cell_ids:
        by_name_protein.setdefault(cell, []).append(cell_key(protein_source, cell))
    names = sorted(set(by_name_rna) | set(by_name_protein))
    return [
        CellCandidate(
            display_name=name,
            rna_cells=tuple(by_name_rna.get(name, [])),
            protein_cells=tuple(by_name_protein.get(name, [])),
        )
        for name in names
    ]


def associate_measurements(
    rna: CountMatrix,
    protein: CountMatrix,
    *,
    rna_source: str,
    protein_source: str,
    mode: str,
    manual_pairs: set[tuple[str, str]] | None = None,
) -> MeasurementAssociation:
    if mode not in ASSOCIATION_MODES:
        raise ValueError(f"未知关联范围: {mode!r}，可选 {ASSOCIATION_MODES}")
    manual_pairs = set(manual_pairs or [])
    valid_rna = {cell_key(rna_source, c) for c in rna.cell_ids}
    valid_protein = {cell_key(protein_source, c) for c in protein.cell_ids}
    unknown_rna = sorted({pair[0] for pair in manual_pairs} - valid_rna)
    unknown_protein = sorted({pair[1] for pair in manual_pairs} - valid_protein)
    if unknown_rna or unknown_protein:
        raise ValueError(f"人工选择含未知来源细胞: rna={unknown_rna}, protein={unknown_protein}")

    decisions: dict[str, AssociationDecision] = {}
    chosen_rna: set[str] = set()
    chosen_protein: set[str] = set()
    candidates = association_candidates(
        rna, protein, rna_source=rna_source, protein_source=protein_source
    )

    for candidate in candidates:
        if not (candidate.rna_cells and candidate.protein_cells):
            continue
        selected = sorted(
            pair
            for pair in manual_pairs
            if pair[0] in candidate.rna_cells and pair[1] in candidate.protein_cells
        )
        if len(selected) > 1:
            raise ValueError(f"显示名 {candidate.display_name!r} 只能选择一个细胞对")
        if selected:
            rna_key, protein_key = selected[0]
            if rna_key in chosen_rna or protein_key in chosen_protein:
                raise ValueError("人工关联必须一对一，不能重复选择同一细胞")
            decisions[candidate.display_name] = AssociationDecision(
                candidate.display_name, rna_key, protein_key, "manual"
            )
            chosen_rna.add(rna_key)
            chosen_protein.add(protein_key)
        elif mode == "paired_only":
            raise ValueError(
                "重复显示名/一对多候选不能自动配对，必须人工选择: "
                + repr(candidate.to_dict())
            )

    rna_index = {cell_key(rna_source, c): i for i, c in enumerate(rna.cell_ids)}
    protein_index = {cell_key(protein_source, c): i for i, c in enumerate(protein.cell_ids)}
    fields = tuple((RNA, gene) for gene in rna.gene_ids) + tuple(
        (PROTEIN, gene) for gene in protein.gene_ids
    )
    rows: list[tuple[str, str, str | None, str | None]] = []

    for cid in rna.cell_ids:
        rna_key = cell_key(rna_source, cid)
        if rna_key in chosen_rna:
            display = next(d.display_name for d in decisions.values() if d.rna_cell == rna_key)
            rows.append((
                f"paired:{rna_key}->{decisions[display].protein_cell}",
                display,
                rna_key,
                decisions[display].protein_cell,
            ))
        elif mode == "all_cells":
            rows.append((f"unpaired-rna:{rna_key}", cid, rna_key, None))

    if mode == "all_cells":
        for cid in protein.cell_ids:
            protein_key = cell_key(protein_source, cid)
            if protein_key not in chosen_protein:
                rows.append((f"unpaired-protein:{protein_key}", cid, None, protein_key))

    if not rows:
        raise ValueError("当前关联范围没有任何细胞行")

    values = np.full((len(rows), len(fields)), np.nan, dtype=np.float64)
    measured = np.zeros(values.shape, dtype=bool)
    n_rna = rna.n_genes
    for row_i, (_, _, rna_key, protein_key) in enumerate(rows):
        if rna_key is not None:
            values[row_i, :n_rna] = rna.counts[rna_index[rna_key]]
            measured[row_i, :n_rna] = True
        if protein_key is not None:
            values[row_i, n_rna:] = protein.counts[protein_index[protein_key]]
            measured[row_i, n_rna:] = True

    batches = tuple(sorted(set(rna.batches) | set(protein.batches)))
    return MeasurementAssociation(
        values=values,
        measured=measured,
        row_keys=tuple(row[0] for row in rows),
        display_names=tuple(row[1] for row in rows),
        rna_cell_keys=tuple(row[2] for row in rows),
        protein_cell_keys=tuple(row[3] for row in rows),
        fields=fields,
        mode=mode,
        decisions=tuple(sorted(decisions.values(), key=lambda d: d.display_name)),
        rna_source=rna_source,
        protein_source=protein_source,
        source_fingerprints={RNA: rna.content_fingerprint(), PROTEIN: protein.content_fingerprint()},
        batches=batches,
    )
