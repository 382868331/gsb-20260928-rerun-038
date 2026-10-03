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

from .joint import JointMatrix, join_batches
from .model import CountMatrix
from .multimodal import (
    MultimodalMatrix,
    MultimodalRow,
    Pairing,
    build_multimodal,
    multimodal_cell_id,
)
from .modality import ModalityMatrix


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

    # ---------------- 多模态关联：保存、核对、撤回、影响分析 ----------------

    @property
    def multimodal_dir(self) -> Path:
        path = self.root / "multimodal"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def _modality_registry_path(self) -> Path:
        return self.root / "modalities.json"

    @property
    def _downstream_registry_path(self) -> Path:
        return self.root / "downstream.json"

    def _load_json(self, path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    @staticmethod
    def _multimodal_hash(pairing: Pairing, params: dict) -> str:
        payload = {
            "modality_fingerprints": pairing.modality_fingerprints,
            "pairs": [p.to_dict() for p in sorted(pairing.pairs, key=lambda r: r.cell_id)],
            "unpaired": [dict(u) for u in sorted(pairing.unpaired, key=json.dumps)],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:10] + "_" + ResultStore._hash(params)[:10]

    def list_multimodal(self) -> list[Path]:
        root = self.root / "multimodal"
        if not root.exists():
            return []
        return sorted(p for p in root.glob("m*") if p.is_dir())

    def _next_multimodal_no(self) -> int:
        nums = []
        for p in self.list_multimodal():
            try:
                nums.append(int(p.name.split("_", 1)[0][1:]))
            except (ValueError, IndexError):
                continue
        return (max(nums) if nums else 0) + 1

    def _validate_multimodal(
        self,
        multi: MultimodalMatrix,
        modalities: dict[str, ModalityMatrix],
        pairing: Pairing,
    ) -> None:
        """发布前用实际测量表重新构造联合矩阵并逐项核对。"""
        reasons: list[str] = []
        if set(pairing.modalities) != set(modalities):
            reasons.append(
                f"配对模态 {list(pairing.modalities)} 与实际测量表 "
                f"{sorted(modalities)} 不一致"
            )
        for name, table in modalities.items():
            if pairing.modality_fingerprints.get(name) != table.content_fingerprint():
                reasons.append(
                    f"模态 {name} 的来源指纹与实际测量表不符（补测/改动需先重新关联）"
                )
        if reasons:
            raise PublishRejectedError(reasons)
        expected = build_multimodal(modalities, pairing)
        if [r.row_id for r in multi.rows] != [r.row_id for r in expected.rows]:
            reasons.append("联合矩阵行集合/顺序与按配对决定重新构造的结果不一致")
        for name in modalities:
            if multi.feature_ids[name] != expected.feature_ids[name]:
                reasons.append(f"模态 {name} 的字段顺序与实际测量表不一致")
            if multi.units[name] != expected.units[name]:
                reasons.append(f"模态 {name} 的单位与实际测量表不一致")
            if multi.measurement_versions[name] != expected.measurement_versions[name]:
                reasons.append(f"模态 {name} 的测量版本与实际测量表不一致")
            if not np.array_equal(multi.measured[name], expected.measured[name]):
                reasons.append(f"模态 {name} 的缺测说明与实际配对不一致")
            else:
                mask = expected.measured[name]
                if not np.allclose(
                    multi.values[name][mask], expected.values[name][mask]
                ):
                    reasons.append(f"模态 {name} 的测量值与实际测量表不一致")
        if reasons:
            raise PublishRejectedError(reasons)

    def save_multimodal(
        self,
        *,
        multi: MultimodalMatrix,
        modalities: dict[str, ModalityMatrix],
        pairing: Pairing,
        params: dict,
    ) -> Path:
        """保存多模态联合版本：内容、字段顺序、单位、版本、配对决定全绑定。

        发布前用实际测量表重新构造核对，任一不符整次拒绝且不留部分版本。
        补测或重新人工选择都会产生新版本，旧版本目录保持可解释、不被覆盖。
        """
        self._validate_multimodal(multi, modalities, pairing)

        version_key = self._multimodal_hash(pairing, params)
        for existing in self.list_multimodal():
            if existing.name.split("_", 1)[1] == version_key:
                return existing

        next_id = self._next_multimodal_no()
        out_dir = self.multimodal_dir / f"m{next_id:04d}_{version_key}"
        if out_dir.exists():
            raise RuntimeError(f"多模态版本目录已存在，拒绝覆盖: {out_dir}")
        staging = self.multimodal_dir / f".staging-m{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            (staging / "params.json").write_text(
                json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (staging / "pairing.json").write_text(
                json.dumps(pairing.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (staging / "multimodal_report.json").write_text(
                json.dumps(multi.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            npz = {
                "modalities": np.array(list(multi.modalities)),
                "row_ids": np.array([r.row_id for r in multi.rows]),
                "row_paired": np.array([r.paired for r in multi.rows]),
            }
            for name in multi.modalities:
                npz[f"values__{name}"] = multi.values[name]
                npz[f"measured__{name}"] = multi.measured[name]
                npz[f"features__{name}"] = np.array(multi.feature_ids[name])
                npz[f"units__{name}"] = np.array(multi.units[name])
            np.savez_compressed(staging / "multimodal.npz", **npz)
            (staging / "meta.json").write_text(
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "kind": "multimodal",
                        "modalities": list(multi.modalities),
                        "feature_ids": {
                            m: list(v) for m, v in multi.feature_ids.items()
                        },
                        "units": {m: list(v) for m, v in multi.units.items()},
                        "measurement_versions": dict(multi.measurement_versions),
                        "modality_fingerprints": dict(pairing.modality_fingerprints),
                        "version_hash": version_key,
                        "published": True,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)

        registry = self._load_json(self._modality_registry_path)
        for name, fingerprint in pairing.modality_fingerprints.items():
            entry = registry.get(name, {})
            version = entry.get("versions", {})
            version.setdefault(
                fingerprint,
                {"status": "active", "measurement_version": pairing.modality_versions[name]},
            )
            entry["versions"] = version
            entry.setdefault("status", "active")
            registry[name] = entry
        self._modality_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out_dir

    def load_pairing(self, version_dir: str | Path) -> Pairing:
        version_dir = Path(version_dir)
        return Pairing.from_dict(
            json.loads((version_dir / "pairing.json").read_text(encoding="utf-8"))
        )

    def load_multimodal(self, version_dir: str | Path) -> MultimodalMatrix:
        """重开多模态版本：恢复行、各模态字段顺序/单位/版本与缺测说明。"""
        version_dir = Path(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        pairing = self.load_pairing(version_dir)
        with np.load(version_dir / "multimodal.npz", allow_pickle=False) as data:
            names = tuple(str(m) for m in data["modalities"])
            row_ids = tuple(str(r) for r in data["row_ids"])
            rows = []
            report = json.loads(
                (version_dir / "multimodal_report.json").read_text(encoding="utf-8")
            )
            report_rows = {r["row_id"]: r for r in report["rows"]}
            values = {}
            measured = {}
            for name in names:
                values[name] = data[f"values__{name}"]
                measured[name] = data[f"measured__{name}"]
            for rid in row_ids:
                info = report_rows[rid]
                rows.append(
                    MultimodalRow(
                        row_id=rid,
                        paired=bool(info["paired"]),
                        modality_cells={k: str(v) for k, v in info["modality_cells"].items()},
                        display_names={k: str(v) for k, v in info["display_names"].items()},
                        missing_modalities=tuple(info["missing_modalities"]),
                        evidence=info["evidence"],
                        pair_cell_id=info.get("pair_cell_id", ""),
                    )
                )
        return MultimodalMatrix(
            rows=tuple(rows),
            modalities=names,
            feature_ids={m: tuple(meta["feature_ids"][m]) for m in names},
            units={m: tuple(meta["units"][m]) for m in names},
            measurement_versions={m: str(v) for m, v in meta["measurement_versions"].items()},
            values=values,
            measured=measured,
            pairing_fingerprints=dict(meta["modality_fingerprints"]),
        )

    def verify_multimodal(
        self,
        version_dir: str | Path,
        modalities: dict[str, ModalityMatrix],
    ) -> dict:
        """重开后逐行逐模态核对，并检查关联是否已被撤回。"""
        version_dir = Path(version_dir)
        pairing = self.load_pairing(version_dir)
        multi = self.load_multimodal(version_dir)
        problems: list[str] = []
        status = self.multimodal_status(version_dir)
        if status != "ok":
            problems.append(f"关联依赖已撤回，联合结果失效: {status}")
        try:
            self._validate_multimodal(multi, modalities, pairing)
        except PublishRejectedError as exc:
            problems.extend(exc.reasons)
        if problems:
            raise PublishRejectedError(problems)
        return {
            "version": version_dir.name,
            "status": status,
            "row_ids": [r.row_id for r in multi.rows],
            "paired_row_ids": [r.row_id for r in multi.rows if r.paired],
            "scope_summary": multi.scope_summary(),
        }

    def multimodal_status(self, version_dir: str | Path) -> str:
        """版本状态：ok / withdrawn_link:<pair> / withdrawn_modality:<name>。"""
        version_dir = Path(version_dir)
        meta = json.loads((version_dir / "meta.json").read_text(encoding="utf-8"))
        registry = self._load_json(self._modality_registry_path)
        for name, fingerprint in meta["modality_fingerprints"].items():
            entry = registry.get(name, {})
            if entry.get("status") == "withdrawn":
                return f"withdrawn_modality:{name}"
            version_entry = entry.get("versions", {}).get(fingerprint, {})
            if version_entry.get("status") == "withdrawn":
                return f"withdrawn_modality:{name}"
        links = self._load_json_list(self.root / "withdrawn_links.json")
        pairing = self.load_pairing(version_dir)
        pair_ids = {p.cell_id for p in pairing.pairs}
        for link in links:
            if (link.get("version") == version_dir.name
                    and link.get("pair_cell_id") in pair_ids):
                return f"withdrawn_link:{link['pair_cell_id']}"
        return "ok"

    def withdraw_association(
        self, version_dir: str | Path, pair_cell_id: str, *, reason: str = ""
    ) -> dict:
        """撤回某联合版本中的一条关联（不撤回任何底层测量）。

        只影响真正包含该关联的多模态版本及其下游筛选；未依赖该关联的
        测量、其他联合结果不受牵连。旧目录不被改写，状态在登记簿中累积。
        """
        target = Path(version_dir).name
        pairing = self.load_pairing(version_dir)
        if not any(p.cell_id == pair_cell_id for p in pairing.pairs):
            raise ValueError(f"版本 {target} 中不存在关联 {pair_cell_id!r}")
        links_path = self.root / "withdrawn_links.json"
        links = self._load_json_list(links_path)
        entry = {"version": target, "pair_cell_id": pair_cell_id, "reason": reason}
        if entry not in links:
            links.append(entry)
        links_path.write_text(
            json.dumps(links, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.withdrawal_impact(
            kind="link", key=f"{target}:{pair_cell_id}"
        )

    def _load_json_list(self, path: Path) -> list:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data)
        return []

    def withdraw_modality(self, name: str, fingerprint: str | None = None) -> dict:
        """撤回一个模态采集批次（或该模态的某个补测版本）。

        fingerprint=None：撤回该模态整批采集；否则只撤回指定测量版本。
        返回受影响的联合版本与下游筛选清单。
        """
        registry = self._load_json(self._modality_registry_path)
        if name not in registry:
            raise ValueError(f"未登记的模态采集批次: {name!r}")
        if fingerprint is None:
            registry[name]["status"] = "withdrawn"
        else:
            versions = registry[name].setdefault("versions", {})
            if fingerprint not in versions:
                raise ValueError(f"模态 {name} 下不存在指纹 {fingerprint}")
            versions[fingerprint]["status"] = "withdrawn"
        self._modality_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.withdrawal_impact(
            kind="modality", key=name if fingerprint is None else f"{name}:{fingerprint}"
        )

    def withdrawal_impact(self, *, kind: str, key: str) -> dict:
        """列出受撤回影响的多模态版本与下游筛选。"""
        affected_versions: list[str] = []
        for version in self.list_multimodal():
            status = self.multimodal_status(version)
            if status != "ok":
                affected_versions.append({"version": version.name, "status": status})
        affected_names = {v["version"] for v in affected_versions}
        downstream = self._load_json(self._downstream_registry_path)
        affected_screens = []
        for screen, info in downstream.items():
            if info.get("multimodal_version") in affected_names:
                affected_screens.append(
                    {
                        "screen": screen,
                        "multimodal_version": info["multimodal_version"],
                        "status": "stale",
                    }
                )
        return {
            "withdrawal": {"kind": kind, "key": key},
            "affected_multimodal_versions": affected_versions,
            "affected_downstream_screens": affected_screens,
            "unaffected_note": "未依赖该资料的测量与版本不在列表中，保持有效",
        }

    def register_downstream_screen(
        self,
        screen: str,
        *,
        multimodal_version: str | Path,
        scope: str,
        description: str = "",
    ) -> None:
        """登记一个基于多模态版本的下游筛选，供撤回时做影响分析。"""
        registry = self._load_json(self._downstream_registry_path)
        registry[screen] = {
            "multimodal_version": Path(multimodal_version).name,
            "scope": scope,
            "description": description,
        }
        self._downstream_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def remeasure_modality(
        self, name: str, old_fingerprint: str, new_table: ModalityMatrix
    ) -> dict:
        """登记某模态补测：新版本入库；旧版本是否撤回独立决定，默认保留。"""
        if new_table.modality != name:
            raise ValueError("补测表的模态名与被补测模态不一致")
        registry = self._load_json(self._modality_registry_path)
        if name not in registry:
            raise ValueError(f"未登记的模态: {name!r}")
        new_fingerprint = new_table.content_fingerprint()
        versions = registry[name].setdefault("versions", {})
        versions.setdefault(
            new_fingerprint,
            {"status": "active", "measurement_version": new_table.measurement_version},
        )
        if old_fingerprint not in versions:
            raise ValueError("旧指纹未登记，无法建立补测关系")
        versions[old_fingerprint].setdefault("superseded_by", new_fingerprint)
        registry[name]["status"] = "active"
        self._modality_registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        impact = self.withdrawal_impact(kind="remeasure", key=name)
        return {
            "modality": name,
            "old_fingerprint": old_fingerprint,
            "new_fingerprint": new_fingerprint,
            "old_versions_still_interpretable": True,
            **impact,
        }
