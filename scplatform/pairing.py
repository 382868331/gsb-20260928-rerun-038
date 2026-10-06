"""RNA 与蛋白联合分析的容量约束配对研究。

研究者为每个可配对来源声明容量规则（一对一或有限一对多），系统按
可用观测相似性 + 人工锁定关系求解联合配对：

- 相似性只在两侧都真实测量的同名特征上计算；缺测位置绝不当作零，
  没有任何共同观测的候选对相似性为 None，绝不能被当成高相似。
- 同一来源细胞的容量不能在多个候选中重复使用：容量 1 即一对一，
  容量 k>1 允许有限一对多，超过即不可行。
- 人工锁定必须全部满足；锁定之间冲突（超容量、未知细胞）时抛出
  PairingConflictError 并逐条列出冲突锁定，绝不自动撤销。
- compare_with_decisions 比较总体一致性与保留既有决定的程度。
- approve_pairs 让用户只批准部分配对：与批准对共享来源细胞的配对
  构成依赖闭包，必须整体接受或整体放弃，避免容量被暗中重复占用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import AssociationDecision
from .joint import cell_key
from .model import CountMatrix

PAIR_BASIS = ("manual", "similarity")


class PairingConflictError(ValueError):
    """人工锁定关系互相冲突或超出声明容量，配对无可行解。

    conflicts 逐条列出冲突来源细胞、声明容量与涉及的锁定对；
    系统只展示冲突，绝不自动撤销任何锁定。
    """

    def __init__(self, conflicts: list[str]):
        self.conflicts = list(conflicts)
        super().__init__("人工锁定关系冲突，无可行解:\n" + "\n".join(f"- {c}" for c in conflicts))


@dataclass(frozen=True)
class PairingCandidate:
    """一个候选配对及其可用观测相似性。

    similarity 为 None 表示两侧没有任何共同测量特征（无观测），
    这类候选只能由人工锁定显式选择，绝不参与自动配对。
    """

    rna_cell: str
    protein_cell: str
    similarity: float | None
    n_shared_features: int

    def to_dict(self) -> dict:
        return {
            "rna_cell": self.rna_cell,
            "protein_cell": self.protein_cell,
            "similarity": self.similarity,
            "n_shared_features": self.n_shared_features,
        }


@dataclass(frozen=True)
class PairedMatch:
    rna_cell: str
    protein_cell: str
    basis: str
    similarity: float | None

    def __post_init__(self) -> None:
        if self.basis not in PAIR_BASIS:
            raise ValueError(f"未知配对依据: {self.basis!r}，可选 {PAIR_BASIS}")

    def to_dict(self) -> dict:
        return {
            "rna_cell": self.rna_cell,
            "protein_cell": self.protein_cell,
            "basis": self.basis,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class PairingPlan:
    """一次容量约束配对研究的结果与完整求解配置（可用于重算核对）。"""

    pairs: tuple[PairedMatch, ...]
    capacities: dict[str, int]
    manual_locks: tuple[tuple[str, str], ...]
    min_similarity: float
    rna_source: str
    protein_source: str
    source_fingerprints: dict[str, str]
    batches: tuple[str, ...]
    unpaired_rna: tuple[str, ...]
    unpaired_protein: tuple[str, ...]
    n_unobserved_candidates: int

    def __post_init__(self) -> None:
        for source, capacity in self.capacities.items():
            if source not in (self.rna_source, self.protein_source):
                raise ValueError(f"容量规则声明了未知来源: {source!r}")
            if capacity < 1:
                raise ValueError("容量必须为正整数")
        pair_keys = [(p.rna_cell, p.protein_cell) for p in self.pairs]
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("配对方案含重复配对")
        for source, keys in [
            (self.rna_source, [p.rna_cell for p in self.pairs]),
            (self.protein_source, [p.protein_cell for p in self.pairs]),
        ]:
            capacity = self.capacities[source]
            over = sorted({k for k in keys if keys.count(k) > capacity})
            if over:
                raise ValueError(f"来源 {source} 的容量 {capacity} 被细胞 {over} 超用")

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def overall_consistency(self) -> float | None:
        """自动配对的平均相似性；无自动配对时返回 None。"""
        similarities = [
            p.similarity
            for p in self.pairs
            if p.basis == "similarity" and p.similarity is not None
        ]
        if not similarities:
            return None
        return float(sum(similarities) / len(similarities))

    def locked_pairs(self) -> set[tuple[str, str]]:
        return set(self.manual_locks)

    def compare_with_decisions(
        self, decisions: tuple[AssociationDecision, ...] | list[AssociationDecision]
    ) -> dict:
        """与既有关联决定比较：保留程度、新增与放弃的配对。"""
        existing = {(d.rna_cell, d.protein_cell) for d in decisions}
        current = {(p.rna_cell, p.protein_cell) for p in self.pairs}
        preserved = sorted(existing & current)
        dropped = sorted(existing - current)
        added = sorted(current - existing)
        return {
            "n_existing": len(existing),
            "n_current": len(current),
            "n_preserved": len(preserved),
            "preservation_rate": (len(preserved) / len(existing)) if existing else None,
            "overall_consistency": self.overall_consistency(),
            "preserved": [list(pair) for pair in preserved],
            "dropped": [list(pair) for pair in dropped],
            "added": [list(pair) for pair in added],
        }

    def to_dict(self) -> dict:
        return {
            "rna_source": self.rna_source,
            "protein_source": self.protein_source,
            "capacities": dict(self.capacities),
            "min_similarity": self.min_similarity,
            "manual_locks": [list(lock) for lock in self.manual_locks],
            "n_pairs": self.n_pairs,
            "overall_consistency": self.overall_consistency(),
            "n_unobserved_candidates": self.n_unobserved_candidates,
            "unpaired_rna": list(self.unpaired_rna),
            "unpaired_protein": list(self.unpaired_protein),
            "source_fingerprints": dict(self.source_fingerprints),
            "batches": list(self.batches),
            "pairs": [p.to_dict() for p in self.pairs],
        }


def _check_measured(mask: np.ndarray | None, matrix: CountMatrix, name: str) -> np.ndarray:
    if mask is None:
        return np.ones(matrix.counts.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != matrix.counts.shape:
        raise ValueError(f"{name} 掩码形状与矩阵不一致")
    return mask


def pairing_candidates(
    rna: CountMatrix,
    protein: CountMatrix,
    *,
    rna_source: str,
    protein_source: str,
    rna_measured: np.ndarray | None = None,
    protein_measured: np.ndarray | None = None,
) -> list[PairingCandidate]:
    """列出全部候选配对及可用观测相似性。

    相似性只在两侧都测量的同名特征上计算（log1p 后的余弦相似性）；
    缺测特征被排除而不是当作零，真实零计数仍是有效测量。没有任何
    共同观测特征的候选 similarity 为 None，绝不当作高相似。
    """
    rna_mask = _check_measured(rna_measured, rna, "rna_measured")
    protein_mask = _check_measured(protein_measured, protein, "protein_measured")
    protein_genes = set(protein.gene_ids)
    shared = [g for g in rna.gene_ids if g in protein_genes]
    rna_cols = [rna.gene_ids.index(g) for g in shared]
    protein_cols = [protein.gene_ids.index(g) for g in shared]

    candidates: list[PairingCandidate] = []
    for i, rna_cell in enumerate(rna.cell_ids):
        for j, protein_cell in enumerate(protein.cell_ids):
            similarity: float | None = None
            n_shared = 0
            if shared:
                both = rna_mask[i, rna_cols] & protein_mask[j, protein_cols]
                n_shared = int(both.sum())
                if n_shared > 0:
                    a = np.log1p(rna.counts[i, rna_cols].astype(np.float64))[both]
                    b = np.log1p(protein.counts[j, protein_cols].astype(np.float64))[both]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    similarity = float(a @ b / denom) if denom > 0 else 0.0
            candidates.append(
                PairingCandidate(
                    rna_cell=cell_key(rna_source, rna_cell),
                    protein_cell=cell_key(protein_source, protein_cell),
                    similarity=similarity,
                    n_shared_features=n_shared,
                )
            )
    return candidates


def solve_pairing(
    rna: CountMatrix,
    protein: CountMatrix,
    *,
    rna_source: str,
    protein_source: str,
    capacities: dict[str, int] | None = None,
    manual_locks: set[tuple[str, str]] | None = None,
    min_similarity: float = 0.0,
    rna_measured: np.ndarray | None = None,
    protein_measured: np.ndarray | None = None,
) -> PairingPlan:
    """按容量约束 + 人工锁定 + 观测相似性求解联合配对。

    capacities: 来源名 -> 每个细胞最多参与的配对数（缺省两侧均为 1，
    即一对一）。人工锁定必须全部满足；锁定超容量或引用未知细胞时
    抛出 PairingConflictError 并列出全部冲突锁定，绝不自动撤销。
    无共同观测的候选（similarity=None）不会被自动配对。
    """
    capacities = dict(capacities or {})
    capacities.setdefault(rna_source, 1)
    capacities.setdefault(protein_source, 1)
    for source in capacities:
        if source not in (rna_source, protein_source):
            raise ValueError(f"容量规则声明了未知来源: {source!r}")
        if capacities[source] < 1:
            raise ValueError("容量必须为正整数")
    manual_locks = set(manual_locks or [])

    valid_rna = {cell_key(rna_source, c) for c in rna.cell_ids}
    valid_protein = {cell_key(protein_source, c) for c in protein.cell_ids}
    conflicts: list[str] = []
    for rna_key, protein_key in sorted(manual_locks):
        if rna_key not in valid_rna:
            conflicts.append(f"锁定引用未知 RNA 细胞: {rna_key}")
        if protein_key not in valid_protein:
            conflicts.append(f"锁定引用未知蛋白细胞: {protein_key}")
    for source, keys, capacity in [
        (rna_source, [pair[0] for pair in manual_locks], capacities[rna_source]),
        (protein_source, [pair[1] for pair in manual_locks], capacities[protein_source]),
    ]:
        over = sorted({k for k in keys if keys.count(k) > capacity})
        for cell in over:
            involved = sorted(pair for pair in manual_locks if cell in pair)
            conflicts.append(
                f"来源 {source} 细胞 {cell} 的容量为 {capacity}，"
                f"却被 {len(involved)} 条锁定占用: {[list(p) for p in involved]}"
            )
    if conflicts:
        raise PairingConflictError(conflicts)

    candidates = pairing_candidates(
        rna,
        protein,
        rna_source=rna_source,
        protein_source=protein_source,
        rna_measured=rna_measured,
        protein_measured=protein_measured,
    )
    by_pair = {(c.rna_cell, c.protein_cell): c for c in candidates}
    n_unobserved = sum(1 for c in candidates if c.similarity is None)

    used_rna: dict[str, int] = {}
    used_protein: dict[str, int] = {}
    pairs: list[PairedMatch] = []

    def has_capacity(rna_key: str, protein_key: str) -> bool:
        return (
            used_rna.get(rna_key, 0) < capacities[rna_source]
            and used_protein.get(protein_key, 0) < capacities[protein_source]
        )

    def take(rna_key: str, protein_key: str, basis: str, similarity: float | None) -> None:
        pairs.append(PairedMatch(rna_key, protein_key, basis, similarity))
        used_rna[rna_key] = used_rna.get(rna_key, 0) + 1
        used_protein[protein_key] = used_protein.get(protein_key, 0) + 1

    for rna_key, protein_key in sorted(manual_locks):
        locked = by_pair.get((rna_key, protein_key))
        take(rna_key, protein_key, "manual", locked.similarity if locked else None)

    automatic = sorted(
        (
            c
            for c in candidates
            if c.similarity is not None
            and c.similarity >= min_similarity
            and (c.rna_cell, c.protein_cell) not in manual_locks
        ),
        key=lambda c: (-c.similarity, c.rna_cell, c.protein_cell),
    )
    for candidate in automatic:
        if has_capacity(candidate.rna_cell, candidate.protein_cell):
            take(candidate.rna_cell, candidate.protein_cell, "similarity", candidate.similarity)

    pairs.sort(key=lambda p: (p.rna_cell, p.protein_cell))
    paired_rna = {p.rna_cell for p in pairs}
    paired_protein = {p.protein_cell for p in pairs}
    return PairingPlan(
        pairs=tuple(pairs),
        capacities=capacities,
        manual_locks=tuple(sorted(manual_locks)),
        min_similarity=float(min_similarity),
        rna_source=rna_source,
        protein_source=protein_source,
        source_fingerprints={
            "rna": rna.content_fingerprint(),
            "protein": protein.content_fingerprint(),
        },
        batches=tuple(sorted(set(rna.batches) | set(protein.batches))),
        unpaired_rna=tuple(sorted(valid_rna - paired_rna)),
        unpaired_protein=tuple(sorted(valid_protein - paired_protein)),
        n_unobserved_candidates=n_unobserved,
    )


def approve_pairs(
    plan: PairingPlan, approved: set[tuple[str, str]]
) -> PairingPlan:
    """批准方案中的部分配对，自动扩展到依赖闭包。

    共享同一来源细胞的配对互相依赖（容量不能重复占用），批准其中
    一条必须连同闭包内全部配对一起接受；闭包外的配对被放弃。
    approved 中出现方案之外的配对直接报错，绝不静默忽略。
    """
    approved = set(approved)
    plan_pairs = {(p.rna_cell, p.protein_cell) for p in plan.pairs}
    unknown = sorted(approved - plan_pairs)
    if unknown:
        raise ValueError(f"批准的配对不在方案中: {[list(p) for p in unknown]}")

    by_cell: dict[str, set[tuple[str, str]]] = {}
    for pair in plan_pairs:
        by_cell.setdefault(pair[0], set()).add(pair)
        by_cell.setdefault(pair[1], set()).add(pair)

    closure: set[tuple[str, str]] = set()
    stack = list(approved)
    while stack:
        pair = stack.pop()
        if pair in closure:
            continue
        closure.add(pair)
        for cell in pair:
            stack.extend(by_cell[cell] - closure)

    kept = tuple(p for p in plan.pairs if (p.rna_cell, p.protein_cell) in closure)
    kept_locks = tuple(lock for lock in plan.manual_locks if tuple(lock) in closure)
    paired_rna = {p.rna_cell for p in kept}
    paired_protein = {p.protein_cell for p in kept}
    all_rna = {p.rna_cell for p in plan.pairs} | set(plan.unpaired_rna)
    all_protein = {p.protein_cell for p in plan.pairs} | set(plan.unpaired_protein)
    return PairingPlan(
        pairs=kept,
        capacities=dict(plan.capacities),
        manual_locks=kept_locks,
        min_similarity=plan.min_similarity,
        rna_source=plan.rna_source,
        protein_source=plan.protein_source,
        source_fingerprints=dict(plan.source_fingerprints),
        batches=plan.batches,
        unpaired_rna=tuple(sorted(all_rna - paired_rna)),
        unpaired_protein=tuple(sorted(all_protein - paired_protein)),
        n_unobserved_candidates=plan.n_unobserved_candidates,
    )
