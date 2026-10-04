"""版本化结果保存：版本键绑定原始矩阵内容、细胞/基因身份与全部参数。

输入指纹由平台根据原始 CountMatrix 自行计算，调用方无法通过沿用同一
标记把不同内容混成同一版本。已发布结果永不被后来的运行覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .association import (
    AssociationDecision,
    MeasurementAssociation,
    associate_measurements,
)
from .joint import JointMatrix, join_batches
from .model import CountMatrix


class PublishRejectedError(ValueError):
    """发布前来源核对失败：整次发布被拒绝，不留下任何部分版本。

    reasons 逐条列出非法身份、重复身份、批次错配等拒绝原因。
    """

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("结果发布被拒绝:\n" + "\n".join(f"- {r}" for r in self.reasons))


def validate_result_identities(
    source_matrix: CountMatrix,
    cell_ids: list[str],
    gene_ids: list[str],
    batch_of_cell: list[str],
    normalized: np.ndarray,
) -> None:
    """发布前核对：结果中的细胞/基因/批次关系必须与来源矩阵一致。

    顺序调整不算不一致（按身份集合与逐细胞批次映射核对）；任何非法
    身份、重复身份、批次错配都会抛出 PublishRejectedError 并附带全部
    拒绝原因，调用方不得留下看似有效的部分版本。
    """
    reasons: list[str] = []

    dupes = sorted({c for c in cell_ids if cell_ids.count(c) > 1})
    if dupes:
        reasons.append(f"结果含重复细胞身份: {dupes}")

    source_cells = set(source_matrix.cell_ids)
    unknown = [c for c in cell_ids if c not in source_cells]
    if unknown:
        reasons.append(f"结果含来源矩阵中不存在的细胞身份: {unknown}")

    gene_dupes = sorted({g for g in gene_ids if gene_ids.count(g) > 1})
    if gene_dupes:
        reasons.append(f"结果含重复基因身份: {gene_dupes}")
    if set(gene_ids) != set(source_matrix.gene_ids):
        missing = sorted(set(source_matrix.gene_ids) - set(gene_ids))
        extra = sorted(set(gene_ids) - set(source_matrix.gene_ids))
        reasons.append(f"结果基因身份与来源不一致: 缺少={missing} 多出={extra}")

    if len(batch_of_cell) != len(cell_ids):
        reasons.append(
            f"批次归属数量({len(batch_of_cell)})与结果细胞数({len(cell_ids)})不一致"
        )
    else:
        source_batch = dict(zip(source_matrix.cell_ids, source_matrix.batch_of_cell))
        mismatched = [
            f"{c}: 结果={b} 来源={source_batch.get(c)}"
            for c, b in zip(cell_ids, batch_of_cell)
            if c in source_batch and source_batch[c] != b
        ]
        if mismatched:
            reasons.append(f"细胞批次归属与来源错配: {mismatched}")

    rows, cols = normalized.shape
    if rows != len(cell_ids) or cols != len(gene_ids):
        reasons.append(
            f"归一化矩阵形状 ({rows}, {cols}) 与细胞/基因身份数量 "
            f"({len(cell_ids)}, {len(gene_ids)}) 不一致"
        )

    if reasons:
        raise PublishRejectedError(reasons)


class ResultStore:
    """结果目录: versions/vNNNN_<input_hash>_<param_hash>/...

    - input_hash 绑定原始矩阵计数 + 细胞/基因身份 + 批次归属；
    - param_hash 绑定本次运行的全部参数（含逐批次 QC/归一化规则）；
    - 同一输入+同一参数重复运行是幂等的，直接返回已发布版本；
    - 输入内容变化（哪怕文件名、run_name 完全相同）必然产生新版本；
    - 已存在的版本目录只读，任何后续试验都无法改写其中文件。
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    @staticmethod
    def _version_no(path: Path) -> int:
        try:
            return int(path.name.split("_", 1)[0][1:])
        except (ValueError, IndexError):
            return 0

    def _next_version_no(self) -> int:
        existing = [self._version_no(p) for p in self.list_versions()]
        return (max(existing) if existing else 0) + 1

    def save_run(
        self,
        *,
        source_matrix: CountMatrix,
        params: dict,
        qc_report: dict,
        normalized: np.ndarray,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        rule_used: dict[str, str],
        comparison: dict | None = None,
    ) -> Path:
        # 发布前核对：结果身份必须与来源矩阵一致，否则整次拒绝，
        # 此时尚未创建任何目录，不会留下部分版本。
        validate_result_identities(
            source_matrix, cell_ids, gene_ids, batch_of_cell, normalized
        )
        identity = source_matrix.identity_digests()
        input_hash = source_matrix.content_fingerprint()[:12]
        param_hash = self._hash(params)[:12]
        version_key = f"{input_hash}_{param_hash}"

        for existing in self.list_versions():
            if existing.name.split("_", 1)[1] == version_key:
                # 输入内容与参数完全一致：结果必然相同，幂等返回已发布版本。
                return existing

        next_id = self._next_version_no()
        out_dir = self.versions_dir / f"v{next_id:04d}_{version_key}"
        if out_dir.exists():
            raise RuntimeError(f"结果目录已存在，拒绝覆盖: {out_dir}")
        staging = self.versions_dir / f".staging-v{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            self._write_staging(
                staging, source_matrix, params, qc_report, normalized,
                cell_ids, gene_ids, batch_of_cell, rule_used, comparison,
            )
        except Exception:
            # 写入中途失败：清理暂存目录，不留下看似有效的部分版本。
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        # 锁定为只读：已发布结果不能被后来的试验改写。
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)
        return out_dir

    def _write_staging(
        self,
        staging: Path,
        source_matrix: CountMatrix,
        params: dict,
        qc_report: dict,
        normalized: np.ndarray,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        rule_used: dict[str, str],
        comparison: dict | None,
    ) -> None:
        identity = source_matrix.identity_digests()
        input_hash = source_matrix.content_fingerprint()[:12]
        param_hash = self._hash(params)[:12]

        (staging / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "qc_report.json").write_text(
            json.dumps(qc_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if comparison is not None:
            (staging / "qc_comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        np.savez_compressed(
            staging / "normalized.npz",
            normalized=normalized,
            cell_ids=np.array(cell_ids),
            gene_ids=np.array(gene_ids),
            batch_of_cell=np.array(batch_of_cell),
            rule_used=json.dumps(rule_used, ensure_ascii=False),
        )
        # 输入身份证明：各组成部分的摘要分开记录，可逐项核对来源。
        (staging / "input_identity.json").write_text(
            json.dumps(
                {
                    "input_fingerprint": source_matrix.content_fingerprint(),
                    "digests": identity,
                    "n_cells": source_matrix.n_cells,
                    "n_genes": source_matrix.n_genes,
                    "cell_ids": list(source_matrix.cell_ids),
                    "gene_ids": list(source_matrix.gene_ids),
                    "batches": list(source_matrix.batches),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_fingerprint": source_matrix.content_fingerprint(),
                    "input_hash": input_hash,
                    "param_hash": param_hash,
                    "n_cells": len(cell_ids),
                    "n_genes": len(gene_ids),
                    "published": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    # ---------------- 联合分析集：保存、核对、批次撤回 ----------------

    @property
    def joints_dir(self) -> Path:
        path = self.root / "joints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _batch_registry_path(self) -> Path:
        return self.root / "batches.json"

    def _load_registry(self) -> dict:
        if self._batch_registry_path.exists():
            return json.loads(self._batch_registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_registry(self, registry: dict) -> None:
        self._batch_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _joint_source_hash(joint: JointMatrix) -> str:
        payload = {
            "mode": joint.mode,
            "sources": joint.source_fingerprints,
        }
        return ResultStore._hash(payload)

    def save_joint(
        self,
        *,
        joint: JointMatrix,
        sources: dict[str, CountMatrix],
        params: dict,
    ) -> Path:
        """保存联合分析集为不可覆盖的新版本。

        发布前核对：joint 中采用的细胞、基因顺序、缺测说明必须与
        sources 中的实际矩阵重新联合的结果一致；任何不一致或写入
        中途失败都整次拒绝，不留下混杂的部分版本。
        批次后来补测基因时来源指纹变化，自动生成新版本而非覆盖。
        """
        self._validate_joint(joint, sources)

        source_hash = self._joint_source_hash(joint)[:12]
        param_hash = self._hash(params)[:12]
        version_key = f"{source_hash}_{param_hash}"
        for existing in self.list_joints():
            if existing.name.split("_", 1)[1] == version_key:
                return existing  # 相同来源+参数：幂等复用

        next_id = self._next_joint_no()
        out_dir = self.joints_dir / f"j{next_id:04d}_{version_key}"
        if out_dir.exists():
            raise RuntimeError(f"联合结果目录已存在，拒绝覆盖: {out_dir}")
        staging = self.joints_dir / f".staging-j{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            self._write_joint_staging(staging, joint, params)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)

        # 登记来源批次（补测后指纹更新，撤回状态不受影响）
        registry = self._load_registry()
        for batch, fingerprint in joint.source_fingerprints.items():
            entry = registry.get(batch, {})
            registry[batch] = {
                "fingerprint": fingerprint,
                "status": entry.get("status", "active"),
            }
        self._save_registry(registry)
        return out_dir

    @staticmethod
    def _validate_joint(joint: JointMatrix, sources: dict[str, CountMatrix]) -> None:
        """发布前核对：采用细胞、基因顺序、缺测说明与实际矩阵一致。"""
        reasons: list[str] = []
        if set(joint.source_fingerprints) != set(sources):
            reasons.append(
                f"联合来源批次 {sorted(joint.source_fingerprints)} 与提供的"
                f"实际矩阵 {sorted(sources)} 不一致"
            )
        if reasons:
            raise PublishRejectedError(reasons)
        for batch, matrix in sources.items():
            actual = matrix.content_fingerprint()
            if joint.source_fingerprints[batch] != actual:
                reasons.append(f"批次 {batch} 的来源指纹与实际矩阵不符")
        if not reasons:
            # 用实际矩阵重新联合，逐项核对采用细胞、基因顺序与缺测掩码
            expected = join_batches(sources, joint.mode)
            if list(joint.gene_ids) != list(expected.gene_ids):
                reasons.append("联合结果的基因顺序与实际矩阵重新联合的结果不一致")
            expected_rows = {k: i for i, k in enumerate(expected.cell_keys)}
            unknown = [k for k in joint.cell_keys if k not in expected_rows]
            if unknown:
                reasons.append(f"联合结果含来源中不存在的细胞: {unknown}")
            for i, key in enumerate(joint.cell_keys):
                if key not in expected_rows:
                    continue
                exp = expected_rows[key]
                if joint.batch_of_cell[i] != expected.batch_of_cell[exp]:
                    reasons.append(f"细胞 {key} 的批次归属与实际矩阵错配")
                if not np.array_equal(joint.measured[i], expected.measured[exp]):
                    reasons.append(f"细胞 {key} 的缺测说明与实际矩阵不一致")
                elif not np.allclose(
                    joint.values[i][joint.measured[i]],
                    expected.values[exp][expected.measured[exp]],
                ):
                    reasons.append(f"细胞 {key} 的测量值与实际矩阵不一致")
        if reasons:
            raise PublishRejectedError(reasons)

    def _write_joint_staging(
        self, staging: Path, joint: JointMatrix, params: dict
    ) -> None:
        (staging / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "joint_report.json").write_text(
            json.dumps(joint.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            staging / "joint.npz",
            values=joint.values,
            measured=joint.measured,
            cell_keys=np.array(joint.cell_keys),
            display_names=np.array(joint.display_names),
            batch_of_cell=np.array(joint.batch_of_cell),
            gene_ids=np.array(joint.gene_ids),
        )
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "kind": "joint",
                    "mode": joint.mode,
                    "source_fingerprints": dict(joint.source_fingerprints),
                    "source_hash": self._joint_source_hash(joint)[:12],
                    "param_hash": self._hash(params)[:12],
                    "n_cells": joint.n_cells,
                    "n_genes": joint.n_genes,
                    "published": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _next_joint_no(self) -> int:
        existing = []
        for p in self.list_joints():
            try:
                existing.append(int(p.name.split("_", 1)[0][1:]))
            except (ValueError, IndexError):
                continue
        return (max(existing) if existing else 0) + 1

    def list_joints(self) -> list[Path]:
        joints_dir = self.root / "joints"
        if not joints_dir.exists():
            return []
        return sorted(p for p in joints_dir.glob("j*") if p.is_dir())

    def withdraw_batch(self, batch: str) -> list[str]:
        """撤回一个批次；只有依赖它的联合结果被标记为失效，其余不受影响。

        返回受影响的联合版本目录名列表。
        """
        registry = self._load_registry()
        entry = registry.get(batch, {"fingerprint": None})
        entry["status"] = "withdrawn"
        registry[batch] = entry
        self._save_registry(registry)
        return [v.name for v in self.list_joints() if batch in self._joint_dependencies(v)]

    @staticmethod
    def _joint_dependencies(version_dir: Path) -> dict[str, str]:
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        return meta.get("source_fingerprints", {})

    def joint_status(self, version_dir: str | Path) -> str:
        """联合版本状态：ok / withdrawn:<批次>（依赖批次被撤回）。"""
        registry = self._load_registry()
        for batch in self._joint_dependencies(Path(version_dir)):
            if registry.get(batch, {}).get("status") == "withdrawn":
                return f"withdrawn:{batch}"
        return "ok"

    def load_joint(self, version_dir: str | Path) -> JointMatrix:
        """重开已保存的联合版本，恢复每个细胞与每组测量的完整记录。"""
        version_dir = Path(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        with np.load(version_dir / "joint.npz", allow_pickle=False) as data:
            return JointMatrix(
                values=data["values"],
                measured=data["measured"],
                cell_keys=tuple(str(c) for c in data["cell_keys"]),
                display_names=tuple(str(c) for c in data["display_names"]),
                batch_of_cell=tuple(str(b) for b in data["batch_of_cell"]),
                gene_ids=tuple(str(g) for g in data["gene_ids"]),
                mode=meta["mode"],
                source_fingerprints=meta["source_fingerprints"],
            )

    def verify_joint(
        self, version_dir: str | Path, sources: dict[str, CountMatrix]
    ) -> dict:
        """重开联合版本并逐项核对：身份、缺测说明与实际矩阵一致，
        且依赖批次均未被撤回；任何问题都抛出 PublishRejectedError。"""
        version_dir = Path(version_dir)
        problems: list[str] = []
        status = self.joint_status(version_dir)
        if status != "ok":
            problems.append(f"依赖批次已被撤回，联合结果失效: {status}")
        joint = self.load_joint(version_dir)
        try:
            self._validate_joint(joint, sources)
        except PublishRejectedError as exc:
            problems.extend(exc.reasons)
        if problems:
            raise PublishRejectedError(problems)
        return {
            "version": version_dir.name,
            "mode": joint.mode,
            "status": status,
            "cell_keys": list(joint.cell_keys),
            "gene_ids": list(joint.gene_ids),
            "missing_genes_by_batch": joint.missing_genes_by_batch(),
        }

    def find_version(self, matrix: CountMatrix, params: dict) -> Path | None:
        """按输入内容+参数精确定位已发布版本；内容不同不会命中。"""
        version_key = f"{matrix.content_fingerprint()[:12]}_{self._hash(params)[:12]}"
        for existing in self.list_versions():
            if existing.name.split("_", 1)[1] == version_key:
                return existing
        return None

    # ------------- RNA/蛋白关联：保存、重开、撤回与下游影响 -------------

    @property
    def associations_dir(self) -> Path:
        path = self.root / "associations"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _downstream_path(self) -> Path:
        return self.root / "downstream_filters.json"

    @staticmethod
    def _association_hash(association: MeasurementAssociation) -> str:
        payload = {
            "mode": association.mode,
            "rna_source": association.rna_source,
            "protein_source": association.protein_source,
            "sources": association.source_fingerprints,
            "decisions": [decision.to_dict() for decision in association.decisions],
        }
        return ResultStore._hash(payload)

    def save_association(
        self,
        *,
        association: MeasurementAssociation,
        rna: CountMatrix,
        protein: CountMatrix,
        params: dict,
    ) -> Path:
        self._validate_association(association, rna, protein)
        source_hash = self._association_hash(association)[:12]
        param_hash = self._hash(params)[:12]
        version_key = f"{source_hash}_{param_hash}"
        for existing in self.list_associations():
            if existing.name.split("_", 1)[1] == version_key:
                return existing

        next_id = self._next_association_no()
        out_dir = self.associations_dir / f"a{next_id:04d}_{version_key}"
        staging = self.associations_dir / f".staging-a{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            self._write_association_staging(staging, association, params)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)

        registry = self._load_registry()
        for source_name, fingerprint in association.source_fingerprints.items():
            registry[source_name] = {
                "fingerprint": fingerprint,
                "status": registry.get(source_name, {}).get("status", "active"),
            }
        self._save_registry(registry)
        associations = self._load_association_registry()
        associations[out_dir.name] = {"status": "active"}
        self._save_association_registry(associations)
        return out_dir

    def _validate_association(
        self,
        association: MeasurementAssociation,
        rna: CountMatrix,
        protein: CountMatrix,
    ) -> None:
        expected = associate_measurements(
            rna,
            protein,
            rna_source=association.rna_source,
            protein_source=association.protein_source,
            mode=association.mode,
            manual_pairs={
                (decision.rna_cell, decision.protein_cell)
                for decision in association.decisions
                if decision.basis == "manual"
            },
        )
        reasons: list[str] = []
        if association.source_fingerprints != expected.source_fingerprints:
            reasons.append("RNA/蛋白来源指纹与实际矩阵不一致")
        if association.mode != expected.mode:
            reasons.append("关联范围与实际重算结果不一致")
        if association.decisions != expected.decisions:
            reasons.append("人工/自动关联决定与实际重算结果不一致")
        if association.row_keys != expected.row_keys:
            reasons.append("关联结果行身份或行顺序与实际重算结果不一致")
        if association.fields != expected.fields:
            reasons.append("关联结果字段身份或字段顺序与实际重算结果不一致")
        if not np.array_equal(association.measured, expected.measured):
            reasons.append("关联结果整种测量缺测掩码与实际重算结果不一致")
        elif not np.allclose(
            association.values[association.measured], expected.values[expected.measured]
        ):
            reasons.append("关联结果测量值与实际重算结果不一致")
        if reasons:
            raise PublishRejectedError(reasons)

    def _write_association_staging(
        self, staging: Path, association: MeasurementAssociation, params: dict
    ) -> None:
        report = association.to_dict()
        (staging / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "association_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            staging / "association.npz",
            values=association.values,
            measured=association.measured,
            row_keys=np.array(association.row_keys),
            display_names=np.array(association.display_names),
            rna_cell_keys=np.array([v or "" for v in association.rna_cell_keys]),
            rna_cell_present=np.array([v is not None for v in association.rna_cell_keys]),
            protein_cell_keys=np.array([v or "" for v in association.protein_cell_keys]),
            protein_cell_present=np.array(
                [v is not None for v in association.protein_cell_keys]
            ),
            field_measurement=np.array([m for m, _ in association.fields]),
            field_feature=np.array([f for _, f in association.fields]),
        )
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "kind": "rna_protein_association",
                    "mode": association.mode,
                    "rna_source": association.rna_source,
                    "protein_source": association.protein_source,
                    "batches": list(association.batches),
                    "source_fingerprints": dict(association.source_fingerprints),
                    "source_hash": self._association_hash(association)[:12],
                    "param_hash": self._hash(params)[:12],
                    "association_decisions": report["association_decisions"],
                    "n_rows": association.n_rows,
                    "n_fields": association.n_fields,
                    "published": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _next_association_no(self) -> int:
        numbers = []
        for path in self.list_associations():
            try:
                numbers.append(int(path.name.split("_", 1)[0][1:]))
            except (ValueError, IndexError):
                continue
        return (max(numbers) if numbers else 0) + 1

    def list_associations(self) -> list[Path]:
        if not self.associations_dir.exists():
            return []
        return sorted(p for p in self.associations_dir.glob("a*") if p.is_dir())

    def load_association(self, version_dir: str | Path) -> MeasurementAssociation:
        version_dir = Path(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        with np.load(version_dir / "association.npz", allow_pickle=False) as data:
            return MeasurementAssociation(
                values=data["values"],
                measured=data["measured"],
                row_keys=tuple(str(v) for v in data["row_keys"]),
                display_names=tuple(str(v) for v in data["display_names"]),
                rna_cell_keys=tuple(
                    None if not present else str(v)
                    for v, present in zip(data["rna_cell_keys"], data["rna_cell_present"])
                ),
                protein_cell_keys=tuple(
                    None if not present else str(v)
                    for v, present in zip(
                        data["protein_cell_keys"], data["protein_cell_present"]
                    )
                ),
                fields=tuple(zip(
                    (str(v) for v in data["field_measurement"]),
                    (str(v) for v in data["field_feature"]),
                )),
                mode=meta["mode"],
                decisions=tuple(),
                rna_source=meta["rna_source"],
                protein_source=meta["protein_source"],
                source_fingerprints=meta["source_fingerprints"],
                batches=tuple(meta["batches"]),
            )

    def verify_association(
        self,
        version_dir: str | Path,
        rna: CountMatrix,
        protein: CountMatrix,
    ) -> dict:
        version_dir = Path(version_dir)
        loaded = self.load_association(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        decisions = tuple(AssociationDecision(**item) for item in meta["association_decisions"])
        association = MeasurementAssociation(
            values=loaded.values,
            measured=loaded.measured,
            row_keys=loaded.row_keys,
            display_names=loaded.display_names,
            rna_cell_keys=loaded.rna_cell_keys,
            protein_cell_keys=loaded.protein_cell_keys,
            fields=loaded.fields,
            mode=loaded.mode,
            decisions=decisions,
            rna_source=loaded.rna_source,
            protein_source=loaded.protein_source,
            source_fingerprints=loaded.source_fingerprints,
            batches=loaded.batches,
        )
        problems = []
        if self.association_status(version_dir) != "ok":
            problems.append(f"关联已撤回: {self.association_status(version_dir)}")
        try:
            self._validate_association(association, rna, protein)
        except PublishRejectedError as exc:
            problems.extend(exc.reasons)
        if problems:
            raise PublishRejectedError(problems)
        return {
            "version": version_dir.name,
            "status": "ok",
            "row_keys": list(association.row_keys),
            "fields": [list(field) for field in association.fields],
            "missing_measurement_by_row": association.missing_measurement_by_row(),
        }

    def _load_association_registry(self) -> dict:
        path = self.root / "associations.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _save_association_registry(self, registry: dict) -> None:
        (self.root / "associations.json").write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def association_status(self, version_dir: str | Path) -> str:
        name = Path(version_dir).name
        status = self._load_association_registry().get(name, {}).get("status", "ok")
        return "ok" if status == "active" else status

    def withdraw_association(self, version_dir: str | Path) -> list[str]:
        name = Path(version_dir).name
        registry = self._load_association_registry()
        registry[name] = {"status": "withdrawn"}
        self._save_association_registry(registry)
        return self.affected_downstream_filters("association", name)

    def register_downstream_filter(
        self, *, kind: str, version: str, filter_id: str, details: dict | None = None
    ) -> None:
        if kind not in {"joint", "association"}:
            raise ValueError("kind 必须是 joint 或 association")
        registry = self._load_downstream_filters()
        registry.setdefault(kind, {}).setdefault(version, {})[filter_id] = details or {}
        self._save_downstream_filters(registry)

    def _load_downstream_filters(self) -> dict:
        if self._downstream_path.exists():
            return json.loads(self._downstream_path.read_text(encoding="utf-8"))
        return {}

    def _save_downstream_filters(self, registry: dict) -> None:
        self._downstream_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def affected_downstream_filters(self, kind: str, version: str) -> list[str]:
        return sorted(self._load_downstream_filters().get(kind, {}).get(version, {}))

    def withdraw_batch_impact(self, batch: str) -> dict:
        self.withdraw_batch(batch)
        affected_joints = [
            path.name for path in self.list_joints() if batch in self._joint_dependencies(path)
        ]
        affected_associations = []
        for path in self.list_associations():
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            if batch in meta.get("batches", []):
                affected_associations.append(path.name)
        filters = {
            "joint": {v: self.affected_downstream_filters("joint", v) for v in affected_joints},
            "association": {
                v: self.affected_downstream_filters("association", v)
                for v in affected_associations
            },
        }
        return {
            "batch": batch,
            "joints": affected_joints,
            "associations": affected_associations,
            "downstream_filters": filters,
        }

    def list_versions(self) -> list[Path]:
        return sorted(p for p in self.versions_dir.glob("v*") if p.is_dir())

    def verify_version(self, version_dir: str | Path, source_matrix: CountMatrix) -> dict:
        """重开已保存版本，逐项核对每个保留对象与来源矩阵一致。

        核对输入指纹、四个组成部分摘要，以及 normalized.npz 中每个
        保留细胞的身份与批次归属；任何不一致都抛出 PublishRejectedError
        并列出全部问题，核对通过则返回身份摘要。
        """
        version_dir = Path(version_dir)
        problems: list[str] = []

        identity = json.loads(
            (version_dir / "input_identity.json").read_text(encoding="utf-8")
        )
        expected = source_matrix.identity_digests()
        if identity["input_fingerprint"] != source_matrix.content_fingerprint():
            problems.append("整体输入指纹与来源矩阵不一致")
        for part, digest in expected.items():
            if identity["digests"].get(part) != digest:
                problems.append(f"输入组成部分 {part} 的摘要与来源不一致")

        with np.load(version_dir / "normalized.npz", allow_pickle=False) as data:
            cell_ids = [str(c) for c in data["cell_ids"]]
            gene_ids = [str(g) for g in data["gene_ids"]]
            batch_of_cell = [str(b) for b in data["batch_of_cell"]]
            normalized = data["normalized"]
        try:
            validate_result_identities(
                source_matrix, cell_ids, gene_ids, batch_of_cell, normalized
            )
        except PublishRejectedError as exc:
            problems.extend(exc.reasons)

        if problems:
            raise PublishRejectedError(problems)
        return {
            "version": version_dir.name,
            "input_fingerprint": identity["input_fingerprint"],
            "kept_cell_ids": cell_ids,
            "gene_ids": gene_ids,
            "batch_of_cell": batch_of_cell,
        }
