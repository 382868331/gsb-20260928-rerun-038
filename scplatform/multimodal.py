"""多模态细胞关联：把同一细胞在不同模态（基因表达、蛋白…）的测量配对。

配对必须有明确来源依据：

- ``shared_key``：该细胞身份在所有相关模态中恰好出现一次（如共享条码），
  平台自动配对，依据可核对；
- ``explicit``：一对多/多对多候选由人工明确选择，记录选择人与候选快照；
- 其余候选保持未配对（``ambiguous`` / ``missing_modality``），
  绝不因为显示名相同就把不同细胞合成一个样本。

分析范围两种：

- ``paired``：只分析已配对细胞；
- ``all``：保留所有细胞，逐模态标明缺少哪种测量。
  整模态缺测以 NaN + measured=False 表示，与该模态内真实零值严格区分。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .modality import ModalityMatrix

SCOPES = ("paired", "all")


def multimodal_cell_id(modality_cell: dict[str, str]) -> str:
    """联合细胞身份：各模态 "模态名::细胞ID" 排序后用 | 连接。

    与显示名无关：两个显示名相同但来源不同的细胞永远得到不同身份，
    不可能被合成一个样本。
    """
    if not modality_cell:
        raise ValueError("联合细胞至少需要一个模态的细胞身份")
    parts = sorted(f"{m}::{cid}" for m, cid in modality_cell.items())
    return "|".join(parts)


@dataclass(frozen=True)
class PairRecord:
    """一条跨模态细胞配对及其来源依据。

    cell_id       : 联合细胞身份（与显示名无关）；
    modality_cells: 模态名 -> 该模态内的细胞 ID；
    evidence      : shared_key / explicit；
    decided_by    : explicit 时记录人工选择者；
    candidates    : explicit 时保存当时的候选快照（保证旧版本仍可解释）；
    note          : 备注。
    """

    cell_id: str
    modality_cells: dict[str, str]
    evidence: str
    decided_by: str = ""
    candidates: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "modality_cells": dict(self.modality_cells),
            "evidence": self.evidence,
            "decided_by": self.decided_by,
            "candidates": {m: list(cs) for m, cs in self.candidates.items()},
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PairRecord":
        return cls(
            cell_id=payload["cell_id"],
            modality_cells=dict(payload["modality_cells"]),
            evidence=payload["evidence"],
            decided_by=payload.get("decided_by", ""),
            candidates={m: list(cs) for m, cs in payload.get("candidates", {}).items()},
            note=payload.get("note", ""),
        )


@dataclass(frozen=True)
class AmbiguousGroup:
    """无法自动确认的候选组：必须人工选择，否则保持未配对。

    同一细胞身份（如条码）在一个或多个模态出现多次，或只有部分模态
    命中且显示名冲突时，候选全部列出，绝不自动挑一个配对。
    """

    group_id: str
    shared_key: str
    candidates: dict[str, list[str]]  # 模态 -> 候选细胞 ID 列表
    display_names: dict[str, dict[str, str]]  # 模态 -> 细胞ID -> 显示名

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "shared_key": self.shared_key,
            "candidates": {m: list(cs) for m, cs in self.candidates.items()},
            "display_names": {m: dict(d) for m, d in self.display_names.items()},
        }


@dataclass(frozen=True)
class Pairing:
    """一组模态表的全部配对决定：已配对记录 + 未决候选 + 未覆盖细胞。

    - pairs   : 已确认配对（evidence 标明来源依据）；
    - ambiguous: 一对多候选组，等待人工选择；
    - unpaired : 未配对细胞，reason 为 ambiguous / missing_modality；
    - duplicate_display_names: 重复显示名核对结果，仅展示用、不参与配对。
    """

    modalities: tuple[str, ...]
    pairs: tuple[PairRecord, ...]
    ambiguous: tuple[AmbiguousGroup, ...]
    unpaired: tuple[dict, ...]
    duplicate_display_names: dict[str, list[str]]
    modality_fingerprints: dict[str, str]
    modality_versions: dict[str, str]

    def pair_for_cell(self, modality: str, cell_id: str) -> PairRecord | None:
        for record in self.pairs:
            if record.modality_cells.get(modality) == cell_id:
                return record
        return None

    def with_resolved(
        self,
        group_id: str,
        choice: dict[str, str],
        *,
        decided_by: str,
        note: str = "",
    ) -> "Pairing":
        """人工从候选组中各模态选一个细胞，产生新配对决定。

        旧 Pairing 不被修改：重新选择后旧联合版本依旧按旧决定可解释。
        候选必须全部来自该组候选，否则拒绝，绝不接受来源不明的细胞。
        """
        groups = {g.group_id: g for g in self.ambiguous}
        if group_id not in groups:
            raise ValueError(f"候选组 {group_id!r} 不存在或已裁决")
        group = groups[group_id]
        if not choice:
            raise ValueError("人工选择至少需要指定一个模态的细胞")
        for modality, picked in choice.items():
            if modality not in group.candidates:
                raise ValueError(f"候选组中没有模态 {modality!r}")
            if picked not in group.candidates[modality]:
                raise ValueError(
                    f"选择的 {modality}::{picked} 不在候选组 {group_id} 中，"
                    "拒绝把来源不明的细胞并入样本"
                )
        record = PairRecord(
            cell_id=multimodal_cell_id(choice),
            modality_cells=dict(choice),
            evidence="explicit",
            decided_by=decided_by,
            candidates={m: list(cs) for m, cs in group.candidates.items()},
            note=note,
        )
        # 新的未决/未配对：去掉该组；该组内未被选中的候选改记为
        # ambiguous 未配对（它们仍未配对，不能被悄悄吞掉）。
        new_ambiguous = tuple(g for g in self.ambiguous if g.group_id != group_id)
        chosen = {(m, c) for m, c in choice.items()}
        leftover: list[dict] = []
        for modality, cells in group.candidates.items():
            for cell in cells:
                if (modality, cell) not in chosen:
                    leftover.append(
                        {
                            "modality": modality,
                            "cell_id": cell,
                            "reason": "ambiguous",
                            "group_id": group_id,
                        }
                    )
        kept_unpaired = tuple(
            u for u in self.unpaired if u.get("group_id") != group_id
        )
        return Pairing(
            modalities=self.modalities,
            pairs=tuple([*self.pairs, record]),
            ambiguous=new_ambiguous,
            unpaired=tuple([*kept_unpaired, *leftover]),
            duplicate_display_names=self.duplicate_display_names,
            modality_fingerprints=dict(self.modality_fingerprints),
            modality_versions=dict(self.modality_versions),
        )

    def to_dict(self) -> dict:
        return {
            "modalities": list(self.modalities),
            "pairs": [p.to_dict() for p in self.pairs],
            "ambiguous": [g.to_dict() for g in self.ambiguous],
            "unpaired": [dict(u) for u in self.unpaired],
            "duplicate_display_names": {
                m: list(v) for m, v in self.duplicate_display_names.items()
            },
            "modality_fingerprints": dict(self.modality_fingerprints),
            "modality_versions": dict(self.modality_versions),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Pairing":
        return cls(
            modalities=tuple(payload["modalities"]),
            pairs=tuple(PairRecord.from_dict(p) for p in payload.get("pairs", [])),
            ambiguous=tuple(
                AmbiguousGroup(
                    group_id=g["group_id"],
                    shared_key=g["shared_key"],
                    candidates={m: list(cs) for m, cs in g["candidates"].items()},
                    display_names={
                        m: dict(d) for m, d in g.get("display_names", {}).items()
                    },
                )
                for g in payload.get("ambiguous", [])
            ),
            unpaired=tuple(dict(u) for u in payload.get("unpaired", [])),
            duplicate_display_names={
                m: list(v) for m, v in payload.get("duplicate_display_names", {}).items()
            },
            modality_fingerprints=dict(payload.get("modality_fingerprints", {})),
            modality_versions=dict(payload.get("modality_versions", {})),
        )


def suggest_pairing(
    modalities: dict[str, ModalityMatrix],
    candidate_keys: dict[str, dict[str, str]] | None = None,
) -> Pairing:
    """根据稳定细胞身份（如共享条码）提出配对，绝不依据显示名。

    规则：

    - 每个共享键（默认就是细胞 ID；可通过 candidate_keys 提供
      “细胞ID -> 共享条码/孔位”映射）在所有模态中恰好命中一个细胞
      -> ``shared_key`` 自动配对；
    - 同一共享键在某模态命中多个不同细胞 ID -> 候选组（ambiguous），
      保持未配对，等待人工选择；
    - 只在部分模态出现 -> 未配对，原因 missing_modality，不自动配；
    - 重复显示名只登记核对，不影响任何配对决定。
    """
    if not modalities:
        raise ValueError("至少需要一个模态才能建立多模态关联")
    names = tuple(modalities.keys())
    if len(set(names)) != len(names):
        raise ValueError("模态名重复")

    candidate_keys = candidate_keys or {}
    key_index: dict[str, dict[str, list[str]]] = {}
    displays: dict[str, dict[str, str]] = {}
    for name, matrix in modalities.items():
        if matrix.modality != name:
            raise ValueError(
                f"模态键 {name!r} 与测量表自报模态 {matrix.modality!r} 不一致，拒绝关联"
            )
        loc: dict[str, list[str]] = {}
        disp: dict[str, str] = {}
        mapping = candidate_keys.get(name, {})
        unknown = sorted(set(mapping) - set(matrix.cell_ids))
        if unknown:
            raise ValueError(f"模态 {name} 的候选键映射含不存在的细胞: {unknown}")
        for i, cid in enumerate(matrix.cell_ids):
            shared = mapping.get(cid, cid)
            loc.setdefault(shared, []).append(cid)
            disp[cid] = matrix.cell_display_names[i]
        key_index[name] = loc
        displays[name] = disp

    all_keys = sorted({key for loc in key_index.values() for key in loc})
    pairs: list[PairRecord] = []
    ambiguous: list[AmbiguousGroup] = []
    unpaired: list[dict] = []

    for seq, shared in enumerate(all_keys, start=1):
        present = {m: loc[shared] for m, loc in key_index.items() if shared in loc}
        if len(present) == len(names) and all(len(v) == 1 for v in present.values()):
            modality_cells = {m: loc[shared][0] for m, loc in key_index.items()}
            pairs.append(
                PairRecord(
                    cell_id=multimodal_cell_id(modality_cells),
                    modality_cells=modality_cells,
                    evidence="shared_key",
                    note=f"共享键 {shared!r} 在所有模态中唯一命中",
                )
            )
            continue
        if any(len(v) > 1 for v in present.values()):
            ambiguous.append(
                AmbiguousGroup(
                    group_id=f"g{seq:04d}:{shared}",
                    shared_key=shared,
                    candidates={m: list(idxs) for m, idxs in present.items()},
                    display_names={
                        m: {cid: displays[m][cid] for cid in present[m]}
                        for m in present
                    },
                )
            )
            for m, cids in present.items():
                for cid in cids:
                    unpaired.append(
                        {
                            "modality": m,
                            "cell_id": cid,
                            "reason": "ambiguous",
                            "group_id": f"g{seq:04d}:{shared}",
                        }
                    )
        else:
            missing = [m for m in names if m not in present]
            for m, cids in present.items():
                for cid in cids:
                    unpaired.append(
                        {
                            "modality": m,
                            "cell_id": cid,
                            "reason": "missing_modality",
                            "missing_modalities": missing,
                        }
                    )

    duplicate_display_names: dict[str, list[str]] = {}
    for name, matrix in modalities.items():
        seen: dict[str, int] = {}
        for disp in matrix.cell_display_names:
            seen[disp] = seen.get(disp, 0) + 1
        duplicate_display_names[name] = sorted(
            d for d, count_ in seen.items() if count_ > 1
        )

    return Pairing(
        modalities=names,
        pairs=tuple(pairs),
        ambiguous=tuple(ambiguous),
        unpaired=tuple(unpaired),
        duplicate_display_names=duplicate_display_names,
        modality_fingerprints={m: x.content_fingerprint() for m, x in modalities.items()},
        modality_versions={m: x.measurement_version for m, x in modalities.items()},
    )


@dataclass(frozen=True)
class MultimodalRow:
    """联合矩阵中的一行：一个物理细胞在各模态的对应位置。"""

    row_id: str
    paired: bool
    modality_cells: dict[str, str]  # 模态 -> 该模态细胞 ID；缺席模态不在其中
    display_names: dict[str, str]
    missing_modalities: tuple[str, ...]
    evidence: str  # shared_key / explicit / unpaired
    pair_cell_id: str  # 配对行的联合身份；未配对行为 ""


@dataclass(frozen=True)
class MultimodalMatrix:
    """多模态联合矩阵：每个模态保留自己的字段顺序、单位与测量版本。

    values[m] / measured[m] 形状均为 n_rows x 该模态字段数：
    measured=False 处必为 NaN，表示该行缺少整种模态测量，
    与该模态内真实测量零值（0 且 measured=True）严格区分。
    """

    rows: tuple[MultimodalRow, ...]
    modalities: tuple[str, ...]
    feature_ids: dict[str, tuple[str, ...]]
    units: dict[str, tuple[str, ...]]
    measurement_versions: dict[str, str]
    values: dict[str, np.ndarray]
    measured: dict[str, np.ndarray]
    pairing_fingerprints: dict[str, str]

    def __post_init__(self) -> None:
        n_rows = len(self.rows)
        values = {m: np.asarray(v, dtype=np.float64) for m, v in self.values.items()}
        measured = {m: np.asarray(v, dtype=bool) for m, v in self.measured.items()}
        if set(values) != set(self.modalities) or set(measured) != set(self.modalities):
            raise ValueError("逐模态测量数组的模态集合与 modalities 不一致")
        for m in self.modalities:
            if values[m].shape != measured[m].shape:
                raise ValueError(f"模态 {m} 的 values 与 measured 形状不一致")
            if values[m].shape[0] != n_rows:
                raise ValueError(f"模态 {m} 的行数与联合行数不一致")
            nf = len(self.feature_ids[m])
            if values[m].shape[1] != nf:
                raise ValueError(f"模态 {m} 的字段数与其字段顺序不一致")
            if len(self.units[m]) != nf:
                raise ValueError(f"模态 {m} 的单位数量与字段数不一致")
            if np.isnan(values[m][measured[m]]).any():
                raise ValueError(f"模态 {m} 的测量位置出现 NaN")
            if measured[m].any() and not np.isnan(values[m][~measured[m]]).all():
                raise ValueError(f"模态 {m} 的整模态缺测位置必须为 NaN")
        for i, row in enumerate(self.rows):
            for m in self.modalities:
                present = m in row.modality_cells
                if present != bool(measured[m][i].any()):
                    raise ValueError(
                        f"第 {i} 行模态 {m} 的缺测标记与配对状态不一致"
                    )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "measured", measured)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def row_ids(self, scope: str) -> list[str]:
        if scope not in SCOPES:
            raise ValueError(f"未知分析范围: {scope!r}，可选 {SCOPES}")
        if scope == "paired":
            return [r.row_id for r in self.rows if r.paired]
        return [r.row_id for r in self.rows]

    def scope_summary(self) -> dict:
        """两种范围各自包含哪些细胞，以及未配对细胞缺少什么测量。"""
        paired, all_rows = [], []
        unpaired_detail = []
        for row in self.rows:
            all_rows.append(row.row_id)
            if row.paired:
                paired.append(row.row_id)
            else:
                unpaired_detail.append(
                    {
                        "row_id": row.row_id,
                        "present_modality": next(iter(row.modality_cells)),
                        "missing_modalities": list(row.missing_modalities),
                    }
                )
        return {
            "paired": {"n_rows": len(paired), "row_ids": paired},
            "all": {
                "n_rows": len(all_rows),
                "row_ids": all_rows,
                "unpaired": unpaired_detail,
            },
        }

    def to_dict(self) -> dict:
        return {
            "modalities": list(self.modalities),
            "feature_ids": {m: list(v) for m, v in self.feature_ids.items()},
            "units": {m: list(v) for m, v in self.units.items()},
            "measurement_versions": dict(self.measurement_versions),
            "rows": [
                {
                    "row_id": r.row_id,
                    "paired": r.paired,
                    "evidence": r.evidence,
                    "pair_cell_id": r.pair_cell_id,
                    "modality_cells": dict(r.modality_cells),
                    "display_names": dict(r.display_names),
                    "missing_modalities": list(r.missing_modalities),
                }
                for r in self.rows
            ],
            "scope_summary": self.scope_summary(),
        }


def build_multimodal(
    modalities: dict[str, ModalityMatrix], pairing: Pairing
) -> MultimodalMatrix:
    """按配对决定构造联合矩阵；发布前可据此重新构造以核对。"""
    names = tuple(modalities.keys())
    if names != tuple(pairing.modalities):
        raise ValueError("配对决定的模态集合/顺序与实际测量表不一致")
    for name, matrix in modalities.items():
        actual = matrix.content_fingerprint()
        if pairing.modality_fingerprints.get(name) != actual:
            raise ValueError(
                f"模态 {name} 的实际内容与配对决定所依据的来源指纹不符，"
                "不能把旧决定套用到不同测量上"
            )

    assigned: dict[str, set[str]] = {m: set() for m in names}
    rows: list[MultimodalRow] = []
    blocks_values = {m: [] for m in names}
    blocks_measured = {m: [] for m in names}

    def append_row(row: MultimodalRow) -> None:
        rows.append(row)
        for m, table in modalities.items():
            if m in row.modality_cells:
                cid = row.modality_cells[m]
                idx = table.cell_ids.index(cid)
                blocks_values[m].append(table.values[idx].copy())
                blocks_measured[m].append(np.ones(table.n_features, dtype=bool))
            else:
                blocks_values[m].append(np.full(table.n_features, np.nan))
                blocks_measured[m].append(np.zeros(table.n_features, dtype=bool))

    for record in sorted(pairing.pairs, key=lambda r: r.cell_id):
        unknown = [m for m in record.modality_cells if m not in modalities]
        if unknown:
            raise ValueError(f"配对 {record.cell_id} 含未知模态: {unknown}")
        for m, cid in record.modality_cells.items():
            if cid not in modalities[m].cell_ids:
                raise ValueError(
                    f"配对 {record.cell_id} 引用了模态 {m} 中不存在的细胞 {cid}"
                )
            if cid in assigned[m]:
                raise ValueError(f"模态 {m} 的细胞 {cid} 被多条配对重复使用")
            assigned[m].add(cid)
        missing = tuple(m for m in names if m not in record.modality_cells)
        append_row(
            MultimodalRow(
                row_id=f"pair:{record.cell_id}",
                paired=True,
                modality_cells=dict(record.modality_cells),
                display_names={
                    m: modalities[m].cell_display_names[
                        modalities[m].cell_ids.index(cid)
                    ]
                    for m, cid in record.modality_cells.items()
                },
                missing_modalities=missing,
                evidence=record.evidence,
                pair_cell_id=record.cell_id,
            )
        )

    for entry in sorted(pairing.unpaired, key=lambda u: (u["modality"], u["cell_id"])):
        m, cid = entry["modality"], entry["cell_id"]
        if cid in assigned[m]:
            continue  # 已通过人工选择进入某配对
        if cid not in modalities[m].cell_ids:
            raise ValueError(f"未配对记录引用了模态 {m} 中不存在的细胞 {cid}")
        assigned[m].add(cid)
        idx = modalities[m].cell_ids.index(cid)
        append_row(
            MultimodalRow(
                row_id=f"unpaired:{m}::{cid}",
                paired=False,
                modality_cells={m: cid},
                display_names={m: modalities[m].cell_display_names[idx]},
                missing_modalities=tuple(x for x in names if x != m),
                evidence="unpaired",
                pair_cell_id="",
            )
        )

    for m in names:
        leftover = set(modalities[m].cell_ids) - assigned[m]
        if leftover:
            raise ValueError(f"模态 {m} 有细胞既未配对也未登记未配对: {sorted(leftover)}")

    values = {m: np.vstack(blocks_values[m]) for m in names}
    measured = {m: np.vstack(blocks_measured[m]) for m in names}
    return MultimodalMatrix(
        rows=tuple(rows),
        modalities=names,
        feature_ids={m: modalities[m].feature_ids for m in names},
        units={m: modalities[m].units for m in names},
        measurement_versions={m: modalities[m].measurement_version for m in names},
        values=values,
        measured=measured,
        pairing_fingerprints=dict(pairing.modality_fingerprints),
    )


@dataclass(frozen=True)
class ModalityQCParams:
    """单模态质量判断规则；真实零值计数，整模态缺测不参与判断。"""

    min_total_signal: float = 1.0
    min_detected_features: int = 1

    def to_dict(self) -> dict:
        return {
            "min_total_signal": self.min_total_signal,
            "min_detected_features": self.min_detected_features,
        }


def _row_verdict(
    multi: MultimodalMatrix,
    row_index: int,
    modality: str,
    params: ModalityQCParams,
) -> dict:
    """逐行逐模态判断；该模态整种缺测时明确记为 missing_modality。"""
    row = multi.rows[row_index]
    if modality not in row.modality_cells:
        return {
            "row_id": row.row_id,
            "modality": modality,
            "status": "missing_modality",
            "reasons": [],
        }
    mask = multi.measured[modality][row_index]
    values = multi.values[modality][row_index][mask]
    total = float(values.sum())
    detected = int((values > 0).sum())
    reasons = []
    if total == 0:
        reasons.append("all_zero_signal")
    if total < params.min_total_signal:
        reasons.append(f"total_signal<{params.min_total_signal:g}")
    if detected < params.min_detected_features:
        reasons.append(f"detected_features<{params.min_detected_features}")
    return {
        "row_id": row.row_id,
        "modality": modality,
        "status": "rejected" if reasons else "kept",
        "reasons": reasons,
    }


@dataclass(frozen=True)
class ScopeComparison:
    """paired 与 all 两种范围对筛选与质量判断的影响。"""

    multi: MultimodalMatrix
    paired_verdicts: dict[str, list[dict]]
    all_verdicts: dict[str, list[dict]]

    def to_dict(self) -> dict:
        def pack(verdicts: dict[str, list[dict]]) -> dict:
            kept = sorted(
                {v["row_id"] for vs in verdicts.values() for v in vs
                 if v["status"] == "kept"}
            )
            rejected = sorted(
                {v["row_id"] for vs in verdicts.values() for v in vs
                 if v["status"] == "rejected"}
            )
            missing = [
                v for vs in verdicts.values() for v in vs
                if v["status"] == "missing_modality"
            ]
            return {
                "kept_row_ids": kept,
                "rejected_row_ids": rejected,
                "missing_modality_cells": [
                    {"row_id": v["row_id"], "modality": v["modality"]} for v in missing
                ],
                "verdicts": verdicts,
            }

        summary = self.multi.scope_summary()
        paired_only = sorted(
            set(summary["paired"]["row_ids"]) - set(summary["all"]["row_ids"])
        )
        all_only = sorted(
            set(summary["all"]["row_ids"]) - set(summary["paired"]["row_ids"])
        )
        return {
            "paired": pack(self.paired_verdicts),
            "all": pack(self.all_verdicts),
            "paired_only_row_ids": paired_only,
            "all_only_row_ids": all_only,
            "note": (
                "paired 范围排除了部分配/未补测细胞；all 范围保留它们并以 "
                "missing_modality 标明缺少哪种测量，缺测不计入该模态质量判断"
            ),
        }


def compare_scopes(
    modalities: dict[str, ModalityMatrix],
    pairing: Pairing,
    params_by_modality: dict[str, ModalityQCParams],
) -> ScopeComparison:
    """比较只分析已配对细胞与保留全部细胞两种范围的筛选影响。"""
    missing = set(modalities) - set(params_by_modality)
    if missing:
        raise ValueError(f"以下模态缺少质量规则: {sorted(missing)}")
    multi = build_multimodal(modalities, pairing)
    paired_rows = [i for i, r in enumerate(multi.rows) if r.paired]
    all_rows = list(range(multi.n_rows))

    def evaluate(rows: list[int]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for modality, table in modalities.items():
            out[modality] = [
                _row_verdict(multi, i, modality, params_by_modality[modality])
                for i in rows
            ]
        return out

    return ScopeComparison(
        multi=multi,
        paired_verdicts=evaluate(paired_rows),
        all_verdicts=evaluate(all_rows),
    )
