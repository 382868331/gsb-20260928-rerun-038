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

from .associate import AssayJoint, AssayLink, associate_assays
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
        depends_on: list[str] | None = None,
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
                depends_on,
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
        depends_on: list[str] | None,
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
                    "depends_on": list(depends_on or []),
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

    # ---------------- 两种测量关联：保存、核对、撤回与影响追踪 ----------------

    @property
    def assocs_dir(self) -> Path:
        path = self.root / "assocs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _assoc_registry_path(self) -> Path:
        return self.root / "associations.json"

    def _load_assoc_registry(self) -> dict:
        if self._assoc_registry_path.exists():
            return json.loads(self._assoc_registry_path.read_text(encoding="utf-8"))
        return {}

    def _save_assoc_registry(self, registry: dict) -> None:
        self._assoc_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @staticmethod
    def _assoc_source_hash(joint: AssayJoint) -> str:
        payload = {
            "scope": joint.scope,
            "sources": joint.source_fingerprints,
            "links": [link.to_dict() for link in joint.links],
        }
        return ResultStore._hash(payload)

    def list_assocs(self) -> list[Path]:
        assocs_dir = self.root / "assocs"
        if not assocs_dir.exists():
            return []
        return sorted(p for p in assocs_dir.glob("a*") if p.is_dir())

    def _next_assoc_no(self) -> int:
        existing = []
        for p in self.list_assocs():
            try:
                existing.append(int(p.name.split("_", 1)[0][1:]))
            except (ValueError, IndexError):
                continue
        return (max(existing) if existing else 0) + 1

    def save_association(
        self,
        *,
        joint: AssayJoint,
        rna: dict[str, CountMatrix],
        protein: dict[str, CountMatrix],
        params: dict,
    ) -> Path:
        """保存两种测量的联合结果为不可覆盖的新版本。

        发布前用实际矩阵按同一关联决定重新关联，逐项核对行列身份、
        字段顺序、缺测掩码与测量值；任何不一致或写入中途失败都整次
        拒绝，不留下混杂版本。批次补测后来源指纹变化，自动生成新版本，
        旧版本的解释保持不变。
        """
        self._validate_association(joint, rna, protein)

        source_hash = self._assoc_source_hash(joint)[:12]
        param_hash = self._hash(params)[:12]
        version_key = f"{source_hash}_{param_hash}"
        for existing in self.list_assocs():
            if existing.name.split("_", 1)[1] == version_key:
                return existing  # 相同来源+关联决定+参数：幂等复用

        next_id = self._next_assoc_no()
        out_dir = self.assocs_dir / f"a{next_id:04d}_{version_key}"
        if out_dir.exists():
            raise RuntimeError(f"关联结果目录已存在，拒绝覆盖: {out_dir}")
        staging = self.assocs_dir / f".staging-a{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            self._write_assoc_staging(staging, joint, params)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)

        registry = self._load_assoc_registry()
        registry.setdefault(out_dir.name, {"status": "active"})
        self._save_assoc_registry(registry)
        return out_dir

    @staticmethod
    def _validate_association(
        joint: AssayJoint, rna: dict[str, CountMatrix], protein: dict[str, CountMatrix]
    ) -> None:
        """发布前核对：行列身份、字段顺序、关联决定与实际矩阵一致。"""
        reasons: list[str] = []
        expected_fp: dict[str, str] = {}
        for batch, matrix in rna.items():
            expected_fp[f"rna::{batch}"] = matrix.content_fingerprint()
        for batch, matrix in protein.items():
            expected_fp[f"protein::{batch}"] = matrix.content_fingerprint()
        if dict(joint.source_fingerprints) != expected_fp:
            reasons.append("联合结果的来源指纹与实际 RNA/蛋白矩阵不符")
        if reasons:
            raise PublishRejectedError(reasons)
        try:
            expected = associate_assays(
                rna, protein, joint.scope, manual_links=joint.manual_link_pairs()
            )
        except ValueError as exc:
            raise PublishRejectedError([f"按保存的关联决定重新关联失败: {exc}"])
        if list(joint.feature_ids) != list(expected.feature_ids):
            reasons.append("联合结果的字段顺序与实际矩阵重新关联的结果不一致")
        if list(joint.modality_of_feature) != list(expected.modality_of_feature):
            reasons.append("联合结果的字段测量归属与实际矩阵不一致")
        if list(joint.row_keys) != list(expected.row_keys):
            reasons.append("联合结果的行身份及顺序与实际矩阵重新关联的结果不一致")
        if [l.to_dict() for l in joint.links] != [l.to_dict() for l in expected.links]:
            reasons.append("联合结果的关联决定与重新关联的结果不一致")
        if not reasons:
            if not np.array_equal(joint.measured, expected.measured):
                reasons.append("联合结果的缺测说明与实际矩阵不一致")
            elif not np.allclose(
                joint.values[joint.measured], expected.values[expected.measured]
            ):
                reasons.append("联合结果的测量值与实际矩阵不一致")
        if reasons:
            raise PublishRejectedError(reasons)

    def _write_assoc_staging(
        self, staging: Path, joint: AssayJoint, params: dict
    ) -> None:
        (staging / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "assoc_report.json").write_text(
            json.dumps(joint.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            staging / "assoc.npz",
            values=joint.values,
            measured=joint.measured,
            row_keys=np.array(joint.row_keys),
            rna_keys=np.array([k if k is not None else "" for k in joint.rna_keys]),
            protein_keys=np.array(
                [k if k is not None else "" for k in joint.protein_keys]
            ),
            display_names=np.array(joint.display_names),
            feature_ids=np.array(joint.feature_ids),
            modality_of_feature=np.array(joint.modality_of_feature),
        )
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "kind": "association",
                    "scope": joint.scope,
                    "source_fingerprints": dict(joint.source_fingerprints),
                    "links": [link.to_dict() for link in joint.links],
                    "source_hash": self._assoc_source_hash(joint)[:12],
                    "param_hash": self._hash(params)[:12],
                    "n_rows": joint.n_rows,
                    "n_features": joint.n_features,
                    "published": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load_association(self, version_dir: str | Path) -> AssayJoint:
        """重开已保存的关联版本，恢复行列身份、字段顺序与关联决定。"""
        version_dir = Path(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        with np.load(version_dir / "assoc.npz", allow_pickle=False) as data:
            rna_keys = tuple(
                str(k) if str(k) else None for k in data["rna_keys"]
            )
            protein_keys = tuple(
                str(k) if str(k) else None for k in data["protein_keys"]
            )
            return AssayJoint(
                values=data["values"],
                measured=data["measured"],
                row_keys=tuple(str(k) for k in data["row_keys"]),
                rna_keys=rna_keys,
                protein_keys=protein_keys,
                display_names=tuple(str(n) for n in data["display_names"]),
                feature_ids=tuple(str(f) for f in data["feature_ids"]),
                modality_of_feature=tuple(str(m) for m in data["modality_of_feature"]),
                links=tuple(
                    AssayLink(
                        rna_key=l["rna_key"],
                        protein_key=l["protein_key"],
                        display_name=l["display_name"],
                        decided_by=l["decided_by"],
                    )
                    for l in meta["links"]
                ),
                scope=meta["scope"],
                source_fingerprints=meta["source_fingerprints"],
            )

    def association_status(self, version_dir: str | Path) -> str:
        """关联版本状态：ok / withdrawn（被撤回或来源批次被撤回）。"""
        version_dir = Path(version_dir)
        registry = self._load_assoc_registry()
        if registry.get(version_dir.name, {}).get("status") == "withdrawn":
            return "withdrawn"
        batch_registry = self._load_registry()
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        for source in meta.get("source_fingerprints", {}):
            batch = source.split("::", 1)[1]
            if batch_registry.get(batch, {}).get("status") == "withdrawn":
                return f"withdrawn:{batch}"
        return "ok"

    def verify_association(
        self,
        version_dir: str | Path,
        rna: dict[str, CountMatrix],
        protein: dict[str, CountMatrix],
    ) -> dict:
        """重开关联版本并逐项核对：行列身份、字段顺序、关联决定、缺测
        说明与实际矩阵一致，且未被撤回；任何问题都抛出 PublishRejectedError。"""
        version_dir = Path(version_dir)
        problems: list[str] = []
        status = self.association_status(version_dir)
        if status != "ok":
            problems.append(f"关联结果已失效: {status}")
        joint = self.load_association(version_dir)
        try:
            self._validate_association(joint, rna, protein)
        except PublishRejectedError as exc:
            problems.extend(exc.reasons)
        if problems:
            raise PublishRejectedError(problems)
        return {
            "version": version_dir.name,
            "scope": joint.scope,
            "status": status,
            "row_keys": list(joint.row_keys),
            "feature_ids": list(joint.feature_ids),
            "links": [link.to_dict() for link in joint.links],
        }

    def withdraw_association(self, version_dir: str | Path) -> dict:
        """撤回一个关联版本；返回受影响的下游筛选版本列表。"""
        version_dir = Path(version_dir)
        registry = self._load_assoc_registry()
        entry = registry.get(version_dir.name, {})
        entry["status"] = "withdrawn"
        registry[version_dir.name] = entry
        self._save_assoc_registry(registry)
        return {
            "association": version_dir.name,
            "filters": self._filters_depending_on({version_dir.name}),
        }

    def _filters_depending_on(self, targets: set[str]) -> list[str]:
        """下游筛选版本中 depends_on 命中受影响对象的版本名。"""
        affected: list[str] = []
        for version in self.list_versions():
            meta_path = version / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if set(meta.get("depends_on", [])) & targets:
                affected.append(version.name)
        return affected

    def withdrawal_impact(self, batch: str) -> dict:
        """撤回批次的影响面：依赖它的联合分析集、两种测量关联，
        以及依赖这些结果的下游筛选版本。"""
        joints = [
            v.name for v in self.list_joints() if batch in self._joint_dependencies(v)
        ]
        assocs: list[str] = []
        for assoc in self.list_assocs():
            meta = json.loads((assoc / "meta.json").read_text(encoding="utf-8"))
            sources = meta.get("source_fingerprints", {})
            if any(source.split("::", 1)[1] == batch for source in sources):
                assocs.append(assoc.name)
        filters = self._filters_depending_on(set(joints) | set(assocs))
        return {
            "batch": batch,
            "joints": joints,
            "associations": assocs,
            "filters": filters,
        }
