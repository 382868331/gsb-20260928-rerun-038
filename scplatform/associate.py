"""RNA 与蛋白两种测量的联合关联：带来源、人工选择、缺测与零严格区分。

- 每种测量、每个细胞都带来源：内部身份为 "批次::细胞ID"，
  显示名仅用于候选匹配，绝不作为自动配对依据。
- 显示名在两侧都唯一时才自动配对；重复显示名产生一对多候选，
  绝不自动配对，必须由调用方通过 manual_links 人工选择。
- 两种细胞范围：paired（仅成功配对的细胞）/ all（保留全部细胞，
  未配对细胞的另一种测量整体缺失）。
- 真实零计数仍是测量值 0；整种测量缺失以 NaN 存储且 measured=False，
  两者绝不混同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .joint import cell_key
from .model import CountMatrix

ASSOC_SCOPES = ("paired", "all")

AUTO_UNIQUE = "auto_unique"
MANUAL = "manual"


class AmbiguousPairingError(ValueError):
    """重复显示名产生一对多候选，自动配对被拒绝，须人工选择。

    unresolved 逐名列出两侧候选的内部身份（"批次::细胞ID"）。
    """

    def __init__(self, unresolved: dict[str, dict[str, list[str]]]):
        self.unresolved = unresolved
        lines = ["重复显示名存在一对多候选，绝不自动配对，须人工选择:"]
        for name, sides in unresolved.items():
            lines.append(
                f"- {name}: rna 候选={sides['rna']} protein 候选={sides['protein']}"
            )
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class AssayLink:
    """一条关联决定：哪个 RNA 细胞与哪个蛋白细胞被配对，以及如何决定。"""

    rna_key: str  # "批次::细胞ID"
    protein_key: str
    display_name: str
    decided_by: str  # AUTO_UNIQUE / MANUAL

    def to_dict(self) -> dict:
        return {
            "rna_key": self.rna_key,
            "protein_key": self.protein_key,
            "display_name": self.display_name,
            "decided_by": self.decided_by,
        }


@dataclass(frozen=True)
class AssayJoint:
    """两种测量的联合结果：行列身份、字段顺序与关联决定全部保留。"""

    values: np.ndarray  # cells x features，整种测量缺失为 NaN，绝不用 0 冒充
    measured: np.ndarray  # cells x features 布尔掩码，True=真实测量
    row_keys: tuple[str, ...]  # 行身份：配对 "rna键|protein键"，未配对为单侧键
    rna_keys: tuple[str | None, ...]  # 每行的 RNA 细胞身份（无则 None）
    protein_keys: tuple[str | None, ...]
    display_names: tuple[str, ...]
    feature_ids: tuple[str, ...]  # "rna::基因" / "protein::标志物"，字段顺序固定
    modality_of_feature: tuple[str, ...]  # 每列所属测量 "rna" / "protein"
    links: tuple[AssayLink, ...]  # 全部关联决定
    scope: str
    source_fingerprints: dict[str, str]  # "rna::批次"/"protein::批次" -> 内容指纹

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        measured = np.asarray(self.measured, dtype=bool)
        if values.shape != measured.shape:
            raise ValueError("values 与 measured 掩码形状不一致")
        n_rows, n_features = values.shape
        if not (len(self.row_keys) == len(self.rna_keys)
                == len(self.protein_keys) == len(self.display_names) == n_rows):
            raise ValueError("行身份数量与矩阵行数不一致")
        if not (len(self.feature_ids) == len(self.modality_of_feature) == n_features):
            raise ValueError("字段身份数量与矩阵列数不一致")
        if len(set(self.row_keys)) != n_rows:
            raise ValueError("联合结果含重复行身份，拒绝构造")
        if len(set(self.feature_ids)) != n_features:
            raise ValueError("联合结果含重复字段身份，拒绝构造")
        if set(self.modality_of_feature) - {"rna", "protein"}:
            raise ValueError("字段所属测量只能是 rna / protein")
        if self.scope not in ASSOC_SCOPES:
            raise ValueError(f"未知细胞范围: {self.scope!r}，可选 {ASSOC_SCOPES}")
        if np.isnan(values[measured]).any():
            raise ValueError("测量位置出现 NaN，缺测掩码与数值不一致")
        if not np.isnan(values[~measured]).all():
            raise ValueError("整种测量缺失必须存储为 NaN，不能用 0 或其他值冒充")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "measured", measured)

    @property
    def n_rows(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    def manual_link_pairs(self) -> tuple[tuple[str, str], ...]:
        """人工选择的关联对（重开/重算时据此复现关联决定）。"""
        return tuple(
            (link.rna_key, link.protein_key)
            for link in self.links
            if link.decided_by == MANUAL
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "feature_ids": list(self.feature_ids),
            "modality_of_feature": list(self.modality_of_feature),
            "source_fingerprints": dict(self.source_fingerprints),
            "links": [link.to_dict() for link in self.links],
            "rows": [
                {
                    "row_key": self.row_keys[i],
                    "rna_key": self.rna_keys[i],
                    "protein_key": self.protein_keys[i],
                    "display_name": self.display_names[i],
                    "n_measured_features": int(self.measured[i].sum()),
                }
                for i in range(self.n_rows)
            ],
        }


def _flatten(matrices: dict[str, CountMatrix], modality: str) -> list[dict]:
    cells: list[dict] = []
    for batch, matrix in matrices.items():
        foreign = sorted(set(matrix.batch_of_cell) - {batch})
        if foreign:
            raise ValueError(
                f"{modality} 批次 {batch} 的矩阵混入了其他批次的细胞: {foreign}，拒绝关联"
            )
        for i, cid in enumerate(matrix.cell_ids):
            cells.append({
                "key": cell_key(batch, cid),
                "name": cid,
                "matrix": matrix,
                "row": i,
            })
    return cells


def associate_assays(
    rna: dict[str, CountMatrix],
    protein: dict[str, CountMatrix],
    scope: str,
    manual_links: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
) -> AssayJoint:
    """把 RNA 与蛋白两种测量按细胞关联成一个联合结果。

    rna / protein: 批次名 -> 该批次的 CountMatrix。显示名在两侧都唯一
    时自动配对；重复显示名的一对多候选绝不自动配对，必须由
    manual_links（("批次::细胞ID", "批次::细胞ID") 对）人工选择，
    否则抛出 AmbiguousPairingError 并列出全部候选。
    scope: "paired" 仅保留配对细胞；"all" 保留全部细胞，未配对者
    另一种测量整体缺失（NaN + measured=False），真实零仍是 0。
    """
    if scope not in ASSOC_SCOPES:
        raise ValueError(f"未知细胞范围: {scope!r}，可选 {ASSOC_SCOPES}")
    if not rna or not protein:
        raise ValueError("RNA 与蛋白两种测量都至少需要一个批次")

    rna_cells = _flatten(rna, "rna")
    protein_cells = _flatten(protein, "protein")
    rna_by_key = {c["key"]: c for c in rna_cells}
    protein_by_key = {c["key"]: c for c in protein_cells}

    # 人工选择的关联决定：身份必须存在、显示名一致、每个细胞至多配对一次
    manual: dict[str, str] = {}
    used_protein: set[str] = set()
    for rna_key, protein_key in manual_links:
        if rna_key not in rna_by_key:
            raise ValueError(f"人工关联的 RNA 细胞不存在: {rna_key}")
        if protein_key not in protein_by_key:
            raise ValueError(f"人工关联的蛋白细胞不存在: {protein_key}")
        if rna_by_key[rna_key]["name"] != protein_by_key[protein_key]["name"]:
            raise ValueError(
                f"人工关联的两侧显示名不一致: {rna_key} 与 {protein_key}，拒绝配对"
            )
        if rna_key in manual:
            raise ValueError(f"RNA 细胞被重复人工配对: {rna_key}")
        if protein_key in used_protein:
            raise ValueError(f"蛋白细胞被重复人工配对: {protein_key}")
        manual[rna_key] = protein_key
        used_protein.add(protein_key)

    # 显示名两侧都唯一 -> 自动配对；否则属于一对多候选，必须人工选择
    rna_by_name: dict[str, list[str]] = {}
    for c in rna_cells:
        rna_by_name.setdefault(c["name"], []).append(c["key"])
    protein_by_name: dict[str, list[str]] = {}
    for c in protein_cells:
        protein_by_name.setdefault(c["name"], []).append(c["key"])

    links: dict[str, tuple[str, str]] = {}  # rna_key -> (protein_key, decided_by)
    for rna_key, protein_key in manual.items():
        links[rna_key] = (protein_key, MANUAL)
    unresolved: dict[str, dict[str, list[str]]] = {}
    for name, rna_keys in sorted(rna_by_name.items()):
        protein_keys = protein_by_name.get(name, [])
        if not protein_keys:
            continue
        unique = len(rna_keys) == 1 and len(protein_keys) == 1
        if unique and rna_keys[0] not in links:
            links[rna_keys[0]] = (protein_keys[0], AUTO_UNIQUE)
            continue
        # 一对多候选：人工选择后仍有未配对的两侧细胞 -> 拒绝
        free_rna = [k for k in rna_keys if k not in links]
        free_protein = [k for k in protein_keys if k not in used_protein]
        if free_rna and free_protein:
            unresolved[name] = {"rna": free_rna, "protein": free_protein}
    if unresolved:
        raise AmbiguousPairingError(unresolved)

    link_list = tuple(
        AssayLink(
            rna_key=rna_key,
            protein_key=protein_key,
            display_name=rna_by_key[rna_key]["name"],
            decided_by=decided_by,
        )
        for rna_key, (protein_key, decided_by) in links.items()
    )

    # 行：paired 仅配对行；all 追加未配对细胞（另一种测量整体缺失）
    paired_rna = set(links)
    paired_protein = {p for p, _ in links.values()}
    rows: list[tuple[str | None, str | None]] = []
    rna_order = {c["key"]: i for i, c in enumerate(rna_cells)}
    for link in sorted(link_list, key=lambda l: rna_order[l.rna_key]):
        rows.append((link.rna_key, link.protein_key))
    if scope == "all":
        for c in rna_cells:
            if c["key"] not in paired_rna:
                rows.append((c["key"], None))
        for c in protein_cells:
            if c["key"] not in paired_protein:
                rows.append((None, c["key"]))

    # 字段：先 RNA 后蛋白，各测量内按批次首次出现顺序，顺序固定可核对
    def feature_union(matrices: dict[str, CountMatrix]) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for matrix in matrices.values():
            for gene in matrix.gene_ids:
                seen.setdefault(gene)
        return tuple(seen)

    rna_features = feature_union(rna)
    protein_features = feature_union(protein)
    feature_ids = tuple(f"rna::{g}" for g in rna_features) + tuple(
        f"protein::{g}" for g in protein_features
    )
    modality_of_feature = ("rna",) * len(rna_features) + ("protein",) * len(protein_features)

    values = np.full((len(rows), len(feature_ids)), np.nan)
    measured = np.zeros(values.shape, dtype=bool)

    def fill(row_idx: int, cell: dict, features: tuple[str, ...], offset: int) -> None:
        matrix = cell["matrix"]
        col_of = {g: j for j, g in enumerate(features)}
        for j, gene in enumerate(matrix.gene_ids):
            if gene not in col_of:
                continue  # 该测量本批次未测此字段：保持 NaN + measured=False
            values[row_idx, offset + col_of[gene]] = matrix.counts[cell["row"], j]
            measured[row_idx, offset + col_of[gene]] = True

    row_keys: list[str] = []
    display_names: list[str] = []
    for i, (rna_key, protein_key) in enumerate(rows):
        if rna_key is not None:
            fill(i, rna_by_key[rna_key], rna_features, 0)
        if protein_key is not None:
            fill(i, protein_by_key[protein_key], protein_features, len(rna_features))
        if rna_key is not None and protein_key is not None:
            row_keys.append(f"{rna_key}|{protein_key}")
            display_names.append(rna_by_key[rna_key]["name"])
        else:
            only = rna_key if rna_key is not None else protein_key
            row_keys.append(str(only))
            cell = rna_by_key.get(rna_key) or protein_by_key[protein_key]
            display_names.append(cell["name"])

    fingerprints: dict[str, str] = {}
    for batch, matrix in rna.items():
        fingerprints[f"rna::{batch}"] = matrix.content_fingerprint()
    for batch, matrix in protein.items():
        fingerprints[f"protein::{batch}"] = matrix.content_fingerprint()

    return AssayJoint(
        values=values,
        measured=measured,
        row_keys=tuple(row_keys),
        rna_keys=tuple(r for r, _ in rows),
        protein_keys=tuple(p for _, p in rows),
        display_names=tuple(display_names),
        feature_ids=feature_ids,
        modality_of_feature=modality_of_feature,
        links=link_list,
        scope=scope,
        source_fingerprints=fingerprints,
    )

