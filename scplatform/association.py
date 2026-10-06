"""RNA 与蛋白两种测量的细胞级关联。

显示名只用于生成候选，从不直接作为内部身份。重复显示名或一对多候选必须
由调用方用 "来源::细胞ID" 明确选择；未配对细胞可在 all 模式中保留，其
缺失的是整种测量，不能写成该测量中的零表达。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .joint import cell_key
from .model import CountMatrix

RNA = "rna"
PROTEIN = "protein"
MEASUREMENTS = (RNA, PROTEIN)
ASSOCIATION_MODES = ("paired_only", "all_cells")


@dataclass(frozen=True)
class PairingCapacities:
    """RNA/蛋白来源细胞可参与配对的数量上限。

    未列出的细胞使用 default。任何细胞最多按自己的容量使用，不能在多个
    候选配对中被重复超额占用。
    """

    rna: dict[str, int] = field(default_factory=dict)
    protein: dict[str, int] = field(default_factory=dict)
    default: int = 1

    def __post_init__(self) -> None:
        if self.default < 1:
            raise ValueError("配对容量必须为正整数")
        for side, capacities in ((RNA, self.rna), (PROTEIN, self.protein)):
            bad = {key: value for key, value in capacities.items() if value < 1}
            if bad:
                raise ValueError(f"{side} 配对容量必须为正整数: {bad}")

    def rna_capacity(self, key: str) -> int:
        return int(self.rna.get(key, self.default))

    def protein_capacity(self, key: str) -> int:
        return int(self.protein.get(key, self.default))

    def effective_for(
        self, rna_keys: tuple[str, ...], protein_keys: tuple[str, ...]
    ) -> "PairingCapacities":
        return PairingCapacities(
            rna={key: self.rna_capacity(key) for key in rna_keys},
            protein={key: self.protein_capacity(key) for key in protein_keys},
            default=self.default,
        )

    def to_dict(self) -> dict:
        return {"default": self.default, RNA: dict(self.rna), PROTEIN: dict(self.protein)}

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PairingCapacities":
        if payload is None:
            return cls()
        return cls(
            rna=dict(payload.get(RNA, {})),
            protein=dict(payload.get(PROTEIN, {})),
            default=int(payload.get("default", 1)),
        )


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
    capacities: PairingCapacities = PairingCapacities()

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
            "capacities": self.capacities.to_dict(),
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
    capacities: PairingCapacities | None = None,
    approved_decisions: tuple[AssociationDecision, ...] | list[AssociationDecision] = (),
    candidate_pairs: set[tuple[str, str, str]] | None = None,
) -> MeasurementAssociation:
    if mode not in ASSOCIATION_MODES:
        raise ValueError(f"未知关联范围: {mode!r}，可选 {ASSOCIATION_MODES}")
    manual_pairs = set(manual_pairs or [])
    capacities = capacities or PairingCapacities()
    valid_rna = {cell_key(rna_source, c) for c in rna.cell_ids}
    valid_protein = {cell_key(protein_source, c) for c in protein.cell_ids}
    unknown_rna = sorted({pair[0] for pair in manual_pairs} - valid_rna)
    unknown_protein = sorted({pair[1] for pair in manual_pairs} - valid_protein)
    if unknown_rna or unknown_protein:
        raise ValueError(f"人工选择含未知来源细胞: rna={unknown_rna}, protein={unknown_protein}")

    decisions: dict[str, list[AssociationDecision]] = {}
    chosen_rna: dict[str, int] = {}
    chosen_protein: dict[str, int] = {}
    valid_rna_keys = set(cell_key(rna_source, c) for c in rna.cell_ids)
    valid_protein_keys = set(cell_key(protein_source, c) for c in protein.cell_ids)
    if candidate_pairs is None:
        candidates = association_candidates(
            rna, protein, rna_source=rna_source, protein_source=protein_source
        )
        candidate_by_pair = {
            (rna_key, protein_key): candidate
            for candidate in candidates
            for rna_key in candidate.rna_cells
            for protein_key in candidate.protein_cells
        }
    else:
        candidate_by_pair = {}
        for rna_key, protein_key, display_name in candidate_pairs:
            if rna_key not in valid_rna_keys or protein_key not in valid_protein_keys:
                raise ValueError(f"未知来源候选: {(rna_key, protein_key, display_name)}")
            candidate_by_pair[(rna_key, protein_key)] = CellCandidate(
                display_name, (rna_key,), (protein_key,)
            )

    def reserve(rna_key: str, protein_key: str) -> None:
        rna_used = chosen_rna.get(rna_key, 0) + 1
        protein_used = chosen_protein.get(protein_key, 0) + 1
        if rna_used > capacities.rna_capacity(rna_key):
            raise ValueError(
                f"RNA 细胞 {rna_key} 超出配对容量 {capacities.rna_capacity(rna_key)}"
            )
        if protein_used > capacities.protein_capacity(protein_key):
            raise ValueError(
                f"蛋白细胞 {protein_key} 超出配对容量 "
                f"{capacities.protein_capacity(protein_key)}"
            )
        chosen_rna[rna_key] = rna_used
        chosen_protein[protein_key] = protein_used

    approved_by_pair = {
        (decision.rna_cell, decision.protein_cell): decision
        for decision in approved_decisions
    }
    if len(approved_by_pair) != len(approved_decisions):
        raise ValueError("批准的容量配对中存在重复细胞对")
    for pair, decision in approved_by_pair.items():
        if pair not in candidate_by_pair:
            raise ValueError(f"批准配对不在可用候选中: {pair}")
        candidate = candidate_by_pair[pair]
        if decision.display_name != candidate.display_name:
            raise ValueError("批准配对的显示名与候选来源不一致")
        reserve(*pair)
        decisions.setdefault(candidate.display_name, []).append(decision)

    for candidate in candidate_by_pair.values():
        if not (candidate.rna_cells and candidate.protein_cells):
            continue
        selected = sorted(
            pair
            for pair in manual_pairs
            if pair[0] in candidate.rna_cells and pair[1] in candidate.protein_cells
        )
        existing_approved = {
            (decision.rna_cell, decision.protein_cell)
            for decision in decisions.get(candidate.display_name, [])
        }
        selected = [pair for pair in selected if pair not in existing_approved]
        is_default_one_to_one = (
            capacities.default == 1 and not capacities.rna and not capacities.protein
        )
        if is_default_one_to_one and len(existing_approved) + len(selected) > 1:
            raise ValueError(f"显示名 {candidate.display_name!r} 只能选择一个细胞对")
        for pair in selected:
            reserve(*pair)
            decisions.setdefault(candidate.display_name, []).append(
                AssociationDecision(candidate.display_name, pair[0], pair[1], "manual")
            )
        if not selected and not existing_approved and mode == "paired_only":
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

    flat_decisions = tuple(
        decision
        for display_name in sorted(decisions)
        for decision in sorted(
            decisions[display_name], key=lambda item: (item.rna_cell, item.protein_cell)
        )
    )
    paired_rna = {decision.rna_cell for decision in flat_decisions}
    paired_protein = {decision.protein_cell for decision in flat_decisions}

    for cid in rna.cell_ids:
        rna_key = cell_key(rna_source, cid)
        rna_decisions = sorted(
            (decision for decision in flat_decisions if decision.rna_cell == rna_key),
            key=lambda decision: decision.protein_cell,
        )
        for decision in rna_decisions:
            rows.append((
                f"paired:{decision.rna_cell}->{decision.protein_cell}",
                decision.display_name,
                decision.rna_cell,
                decision.protein_cell,
            ))
        if not rna_decisions and mode == "all_cells":
            rows.append((f"unpaired-rna:{rna_key}", cid, rna_key, None))

    if mode == "all_cells":
        for cid in protein.cell_ids:
            protein_key = cell_key(protein_source, cid)
            if protein_key not in paired_protein:
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
        decisions=flat_decisions,
        rna_source=rna_source,
        protein_source=protein_source,
        source_fingerprints={RNA: rna.content_fingerprint(), PROTEIN: protein.content_fingerprint()},
        batches=batches,
        capacities=capacities.effective_for(
            tuple(cell_key(rna_source, c) for c in rna.cell_ids),
            tuple(cell_key(protein_source, c) for c in protein.cell_ids),
        ),
    )


@dataclass(frozen=True)
class CapacityPairingPlan:
    """容量约束下的 RNA/蛋白联合配对研究方案。"""

    feasible: bool
    pairs: tuple[tuple[str, str], ...]
    locked_pairs: tuple[tuple[str, str], ...]
    existing_pairs: tuple[tuple[str, str], ...]
    retained_pairs: tuple[tuple[str, str], ...]
    candidate_similarity: dict[tuple[str, str], float]
    no_observation_pairs: tuple[tuple[str, str], ...]
    conflicts: tuple[dict, ...]
    capacities: PairingCapacities
    display_by_pair: dict[tuple[str, str], str]
    overall_consistency: float | None

    @property
    def status(self) -> str:
        return "feasible" if self.feasible else "infeasible_locked_conflict"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "overall_consistency": self.overall_consistency,
            "pairs": [list(pair) for pair in self.pairs],
            "locked_pairs": [list(pair) for pair in self.locked_pairs],
            "existing_pairs": [list(pair) for pair in self.existing_pairs],
            "retained_pairs": [list(pair) for pair in self.retained_pairs],
            "retained_existing": {
                "count": len(self.retained_pairs),
                "total": len(self.existing_pairs),
            },
            "candidate_similarity": [
                {
                    "rna_cell": pair[0],
                    "protein_cell": pair[1],
                    "display_name": self.display_by_pair.get(pair),
                    "similarity": value,
                }
                for pair, value in sorted(self.candidate_similarity.items())
            ],
            "no_observation_pairs": [
                {
                    "rna_cell": pair[0],
                    "protein_cell": pair[1],
                    "display_name": self.display_by_pair.get(pair),
                }
                for pair in self.no_observation_pairs
            ],
            "conflicts": list(self.conflicts),
            "capacities": self.capacities.to_dict(),
        }

    def approved_decisions(
        self, selected_pairs: set[tuple[str, str]] | None = None
    ) -> tuple[AssociationDecision, ...]:
        if not self.feasible:
            raise ValueError("无可行配对方案，不能批准；请先处理冲突锁定关系")
        selected = set(self.pairs if selected_pairs is None else selected_pairs)
        unknown = sorted(selected - set(self.pairs))
        if unknown:
            raise ValueError(f"批准配对不在可行方案中: {unknown}")
        adjacency: dict[str, set[str]] = {}
        for rna_key, protein_key in self.pairs:
            adjacency.setdefault(rna_key, set()).add(protein_key)
            adjacency.setdefault(protein_key, set()).add(rna_key)
        touched: set[str] = set()
        stack = [vertex for pair in selected for vertex in pair]
        while stack:
            vertex = stack.pop()
            if vertex in touched:
                continue
            touched.add(vertex)
            stack.extend(adjacency.get(vertex, set()) - touched)
        required_locks = {
            pair
            for pair in self.locked_pairs
            if pair[0] in touched or pair[1] in touched
        }
        missing_locks = sorted(required_locks - selected)
        if missing_locks:
            raise ValueError(f"批准部分必须包含依赖闭合的锁定配对: {missing_locks}")
        return tuple(
            AssociationDecision(
                self.display_by_pair[pair],
                pair[0],
                pair[1],
                "manual" if pair in required_locks else "capacity_approved",
            )
            for pair in sorted(selected)
        )


def _pair_similarity(
    rna: CountMatrix,
    protein: CountMatrix,
    rna_index: dict[str, int],
    protein_index: dict[str, int],
    rna_key: str,
    protein_key: str,
) -> float | None:
    shared = tuple(
        gene
        for gene in rna.gene_ids
        if gene in set(protein.gene_ids)
    )
    if not shared:
        return None
    rna_cols = np.array([list(rna.gene_ids).index(gene) for gene in shared])
    protein_cols = np.array([list(protein.gene_ids).index(gene) for gene in shared])
    x = rna.counts[rna_index[rna_key], rna_cols].astype(np.float64)
    y = protein.counts[protein_index[protein_key], protein_cols].astype(np.float64)
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denominator == 0:
        return None
    return float(np.dot(xc, yc) / denominator)


def plan_capacity_pairing(
    rna: CountMatrix,
    protein: CountMatrix,
    *,
    rna_source: str,
    protein_source: str,
    capacities: PairingCapacities | None = None,
    locked_pairs: set[tuple[str, str]] | None = None,
    existing_pairs: set[tuple[str, str]] | None = None,
    candidate_pairs: set[tuple[str, str, str]] | None = None,
) -> CapacityPairingPlan:
    """在来源容量、人工锁定和观测相似性约束下寻找联合配对方案。"""
    capacities = capacities or PairingCapacities()
    locked_pairs = set(locked_pairs or [])
    existing_pairs = set(existing_pairs or [])
    rna_keys = tuple(cell_key(rna_source, cid) for cid in rna.cell_ids)
    protein_keys = tuple(cell_key(protein_source, cid) for cid in protein.cell_ids)
    capacities = capacities.effective_for(rna_keys, protein_keys)
    conflicts: list[dict] = []
    if candidate_pairs is None:
        candidates = association_candidates(
            rna, protein, rna_source=rna_source, protein_source=protein_source
        )
        display_by_pair = {
            (rna_key, protein_key): candidate.display_name
            for candidate in candidates
            for rna_key in candidate.rna_cells
            for protein_key in candidate.protein_cells
        }
    else:
        valid = set(rna_keys), set(protein_keys)
        display_by_pair = {}
        for rna_key, protein_key, display_name in sorted(candidate_pairs):
            if rna_key not in valid[0] or protein_key not in valid[1]:
                conflicts.append({
                    "type": "unknown_candidate",
                    "pair": [rna_key, protein_key],
                    "display_name": display_name,
                })
                continue
            display_by_pair[(rna_key, protein_key)] = display_name
    rna_index = {key: i for i, key in enumerate(rna_keys)}
    protein_index = {key: i for i, key in enumerate(protein_keys)}
    for pair in sorted(locked_pairs | existing_pairs):
        if pair[0] not in rna_index or pair[1] not in protein_index:
            conflicts.append({"type": "unknown_pair", "pair": list(pair)})
        elif pair not in display_by_pair:
            conflicts.append({"type": "cross_display_pair", "pair": list(pair)})

    def lock_conflicts(pairs: set[tuple[str, str]]) -> None:
        by_rna: dict[str, set[str]] = {}
        by_protein: dict[str, set[str]] = {}
        for rna_key, protein_key in pairs:
            by_rna.setdefault(rna_key, set()).add(protein_key)
            by_protein.setdefault(protein_key, set()).add(rna_key)
        for key, partners in by_rna.items():
            if len(partners) > capacities.rna_capacity(key):
                conflicts.append({
                    "type": "rna_capacity_locked",
                    "source": key,
                    "capacity": capacities.rna_capacity(key),
                    "locked_partners": sorted(partners),
                })
        for key, partners in by_protein.items():
            if len(partners) > capacities.protein_capacity(key):
                conflicts.append({
                    "type": "protein_capacity_locked",
                    "source": key,
                    "capacity": capacities.protein_capacity(key),
                    "locked_partners": sorted(partners),
                })

    lock_conflicts(locked_pairs)
    if conflicts:
        return CapacityPairingPlan(
            feasible=False,
            pairs=(),
            locked_pairs=tuple(sorted(locked_pairs)),
            existing_pairs=tuple(sorted(existing_pairs)),
            retained_pairs=(),
            candidate_similarity={},
            no_observation_pairs=(),
            conflicts=tuple(conflicts),
            capacities=capacities,
            display_by_pair=display_by_pair,
            overall_consistency=None,
        )

    scores: dict[tuple[str, str], float] = {}
    no_observation: list[tuple[str, str]] = []
    for pair in sorted(display_by_pair):
        score = _pair_similarity(
            rna, protein, rna_index, protein_index, pair[0], pair[1]
        )
        if score is None:
            no_observation.append(pair)
        else:
            scores[pair] = score

    rna_usage = {key: 0 for key in rna_keys}
    protein_usage = {key: 0 for key in protein_keys}
    for pair in locked_pairs:
        rna_usage[pair[0]] += 1
        protein_usage[pair[1]] += 1

    optional = tuple(
        pair
        for pair in sorted(
            (set(scores) | (set(existing_pairs) & set(display_by_pair)))
            - locked_pairs,
            key=lambda item: (
                0 if item in existing_pairs else 1,
                -scores.get(item, 0.0),
                item,
            ),
        )
    )
    best: list[tuple[str, str]] = []
    best_value = (-1, -1, -float("inf"))

    def evaluate(chosen: list[tuple[str, str]]) -> tuple[int, int, float]:
        all_pairs = set(locked_pairs) | set(chosen)
        total = sum(scores.get(pair, 0.0) for pair in chosen)
        total += sum(scores.get(pair, 0.0) for pair in locked_pairs)
        retained = len(all_pairs & existing_pairs)
        return len(all_pairs), retained, total

    def search(position: int, chosen: list[tuple[str, str]]) -> None:
        nonlocal best, best_value
        current = evaluate(chosen)
        if current > best_value:
            best_value = current
            best = list(chosen)
        for index in range(position, len(optional)):
            pair = optional[index]
            rna_key, protein_key = pair
            if rna_usage[rna_key] >= capacities.rna_capacity(rna_key):
                continue
            if protein_usage[protein_key] >= capacities.protein_capacity(protein_key):
                continue
            rna_usage[rna_key] += 1
            protein_usage[protein_key] += 1
            chosen.append(pair)
            search(index + 1, chosen)
            chosen.pop()
            rna_usage[rna_key] -= 1
            protein_usage[protein_key] -= 1

    search(0, [])
    pairs = tuple(sorted(set(locked_pairs) | set(best)))
    retained = tuple(sorted(set(pairs) & existing_pairs))
    observed_scores = [scores[pair] for pair in pairs if pair in scores]
    overall = sum(observed_scores) / len(observed_scores) if observed_scores else None
    return CapacityPairingPlan(
        feasible=True,
        pairs=pairs,
        locked_pairs=tuple(sorted(locked_pairs)),
        existing_pairs=tuple(sorted(existing_pairs)),
        retained_pairs=retained,
        candidate_similarity=scores,
        no_observation_pairs=tuple(no_observation),
        conflicts=(),
        capacities=capacities,
        display_by_pair=display_by_pair,
        overall_consistency=overall,
    )
