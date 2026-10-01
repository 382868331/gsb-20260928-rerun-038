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

from .model import CountMatrix
from .joint import JointMatrix, join_batches


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

    # ---- 联合分析集：多批次（基因集合可不同）联合结果的版本化保存 ----

    def save_joint(
        self,
        *,
        sources: dict[str, CountMatrix],
        joint: JointMatrix,
        params: dict,
        qc_report: dict | None = None,
        comparison: dict | None = None,
    ) -> Path:
        """保存一个联合分析集版本。

        发布前强制核对：joint 的细胞顺序、基因顺序、缺测掩码与每个值
        必须和由 sources 重新联合出来的结果完全一致，否则整次拒绝，
        不留下任何部分版本。版本键绑定全部来源批次的内容指纹、联合
        模式与参数：任一批次补测基因（内容指纹改变）都会生成新版本，
        旧版本只读保留，绝不被覆盖。
        """
        validate_joint_against_sources(sources, joint)
        sources_hash = self._hash(
            {"mode": joint.mode, "sources": [list(p) for p in joint.source_fingerprints]}
        )
        param_hash = self._hash(params)[:12]
        version_key = f"{sources_hash[:12]}_{param_hash}"

        for existing in self.list_joint_versions():
            if existing.name.split("_", 1)[1] == version_key:
                # 相同来源 + 相同模式 + 相同参数：幂等复用已发布版本。
                return existing

        next_id = self._next_joint_version_no()
        out_dir = self.versions_dir / f"j{next_id:04d}_{version_key}"
        if out_dir.exists():
            raise RuntimeError(f"结果目录已存在，拒绝覆盖: {out_dir}")
        staging = self.versions_dir / f".staging-j{next_id:04d}-{os.getpid()}"
        staging.mkdir()
        try:
            self._write_joint_staging(
                staging, joint, params, qc_report, comparison
            )
        except Exception:
            # 任一部分写入失败：清理暂存目录，不留下混杂版本。
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(out_dir)
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)
        return out_dir

    def _next_joint_version_no(self) -> int:
        existing = [
            int(p.name.split("_", 1)[0][1:])
            for p in self.list_joint_versions()
        ]
        return (max(existing) if existing else 0) + 1

    def list_joint_versions(self) -> list[Path]:
        return sorted(p for p in self.versions_dir.glob("j*") if p.is_dir())

    def _write_joint_staging(
        self,
        staging: Path,
        joint: JointMatrix,
        params: dict,
        qc_report: dict | None,
        comparison: dict | None,
    ) -> None:
        np.savez_compressed(
            staging / "joint.npz",
            values=joint.values,
            observed=joint.observed,
            cell_keys=np.array(joint.cell_keys),
            display_names=np.array(joint.display_names),
            gene_ids=np.array(joint.gene_ids),
            batch_of_cell=np.array(joint.batch_of_cell),
        )
        (staging / "joint_identity.json").write_text(
            json.dumps(
                {
                    "mode": joint.mode,
                    "joint_fingerprint": joint.content_fingerprint(),
                    "source_fingerprints": {
                        batch: fp for batch, fp in joint.source_fingerprints
                    },
                    "n_cells": joint.n_cells,
                    "n_genes": joint.n_genes,
                    "cell_keys": list(joint.cell_keys),
                    "display_names": list(joint.display_names),
                    "gene_ids": list(joint.gene_ids),
                    "batch_of_cell": list(joint.batch_of_cell),
                    "missing_by_batch": joint.missing_by_batch(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if qc_report is not None:
            (staging / "joint_qc_report.json").write_text(
                json.dumps(qc_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if comparison is not None:
            (staging / "join_mode_comparison.json").write_text(
                json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        (staging / "meta.json").write_text(
            json.dumps(
                {
                    "kind": "joint",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "mode": joint.mode,
                    "joint_fingerprint": joint.content_fingerprint(),
                    "depends_on": {
                        batch: fp for batch, fp in joint.source_fingerprints
                    },
                    "n_cells": joint.n_cells,
                    "n_genes": joint.n_genes,
                    "published": True,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def verify_joint(
        self, version_dir: str | Path, sources: dict[str, CountMatrix]
    ) -> dict:
        """重开已保存的联合版本，逐细胞、逐基因核对与来源矩阵一致。

        核对每个来源批次的内容指纹、联合细胞/基因顺序、每个值与缺测
        掩码；任何不一致都抛出 PublishRejectedError。核对通过后返回
        身份摘要，并标注 withdrawn_dependencies：若某个依赖批次已被
        撤回，status 为 "withdrawn_dependency"，否则为 "ok"。
        """
        version_dir = Path(version_dir)
        problems: list[str] = []

        identity = json.loads(
            (version_dir / "joint_identity.json").read_text(encoding="utf-8")
        )
        recorded = identity["source_fingerprints"]
        if set(recorded) != set(sources):
            problems.append(
                f"来源批次集合不一致: 记录={sorted(recorded)} 实际={sorted(sources)}"
            )
        for batch, fp in recorded.items():
            if batch in sources and sources[batch].content_fingerprint() != fp:
                problems.append(f"批次 {batch} 的内容指纹与联合版本记录不一致")

        with np.load(version_dir / "joint.npz", allow_pickle=False) as data:
            values = data["values"]
            observed = data["observed"]
            cell_keys = [str(c) for c in data["cell_keys"]]
            display_names = [str(c) for c in data["display_names"]]
            gene_ids = [str(g) for g in data["gene_ids"]]
            batch_of_cell = [str(b) for b in data["batch_of_cell"]]

        if not problems:
            try:
                stored = JointMatrix(
                    values=values,
                    observed=observed,
                    cell_keys=tuple(cell_keys),
                    display_names=tuple(display_names),
                    gene_ids=tuple(gene_ids),
                    batch_of_cell=tuple(batch_of_cell),
                    mode=identity["mode"],
                    source_fingerprints=tuple(sorted(recorded.items())),
                )
                validate_joint_against_sources(sources, stored)
            except PublishRejectedError as exc:
                problems.extend(exc.reasons)
            except ValueError as exc:
                problems.append(str(exc))

        if identity["cell_keys"] != cell_keys:
            problems.append("联合细胞顺序与身份记录不一致")
        if identity["gene_ids"] != gene_ids:
            problems.append("联合基因顺序与身份记录不一致")

        if problems:
            raise PublishRejectedError(problems)

        withdrawn = self._withdrawn_fingerprints()
        withdrawn_deps = sorted(
            batch for batch, fp in recorded.items() if fp in withdrawn
        )
        return {
            "version": version_dir.name,
            "mode": identity["mode"],
            "status": "withdrawn_dependency" if withdrawn_deps else "ok",
            "withdrawn_dependencies": withdrawn_deps,
            "cell_keys": cell_keys,
            "gene_ids": gene_ids,
            "missing_by_batch": identity["missing_by_batch"],
        }

    # ---- 批次撤回：只影响依赖该批次测量的联合结果 ----

    @property
    def _withdrawn_path(self) -> Path:
        return self.root / "withdrawn.json"

    def _withdrawn_records(self) -> list[dict]:
        if not self._withdrawn_path.exists():
            return []
        return json.loads(self._withdrawn_path.read_text(encoding="utf-8"))

    def _withdrawn_fingerprints(self) -> set[str]:
        return {r["fingerprint"] for r in self._withdrawn_records()}

    def withdraw_batch(self, batch: str, fingerprint: str | None = None) -> list[dict]:
        """撤回一个批次的测量；只影响依赖它的联合版本，其他版本不受影响。

        默认撤回该批次名在已发布联合版本中出现过的全部测量指纹；
        传入 fingerprint 可精确撤回某一次测量（例如只撤回补测前的旧
        测量，保留补测后的新联合版本）。返回受影响的联合版本目录名。
        """
        known: set[str] = set()
        for version in self.list_joint_versions():
            identity = json.loads(
                (version / "joint_identity.json").read_text(encoding="utf-8")
            )
            for name, fp in identity["source_fingerprints"].items():
                if name == batch:
                    known.add(fp)
        targets = {fingerprint} if fingerprint else known
        if not targets:
            raise ValueError(f"没有找到批次 {batch!r} 对应的已发布联合测量")

        records = self._withdrawn_records()
        already = {r["fingerprint"] for r in records}
        for fp in sorted(targets):
            if fp not in already:
                records.append(
                    {
                        "batch": batch,
                        "fingerprint": fp,
                        "withdrawn_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        self._withdrawn_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        affected = []
        for version in self.list_joint_versions():
            identity = json.loads(
                (version / "joint_identity.json").read_text(encoding="utf-8")
            )
            if any(fp in targets for fp in identity["source_fingerprints"].values()):
                affected.append(version.name)
        return affected


def validate_joint_against_sources(
    sources: dict[str, CountMatrix], joint: JointMatrix
) -> None:
    """发布前核对：联合结果的细胞、基因顺序与缺测说明必须与实际矩阵一致。

    用 sources 按 joint.mode 重新联合，逐细胞、逐基因、逐值、逐缺测
    标记比对；任何不一致都以 PublishRejectedError 整次拒绝，调用方
    不得留下看似有效的部分版本。
    """
    reasons: list[str] = []
    try:
        expected = join_batches(sources, mode=joint.mode)
    except ValueError as exc:
        raise PublishRejectedError([f"来源批次无法联合: {exc}"]) from exc

    if joint.cell_keys != expected.cell_keys:
        reasons.append("联合细胞身份或顺序与来源矩阵不一致")
    if joint.display_names != expected.display_names:
        reasons.append("细胞显示名与来源矩阵不一致")
    if joint.gene_ids != expected.gene_ids:
        reasons.append("联合基因身份或顺序与来源矩阵不一致")
    if joint.batch_of_cell != expected.batch_of_cell:
        reasons.append("细胞批次归属与来源矩阵不一致")
    if tuple(sorted(joint.source_fingerprints)) != tuple(sorted(expected.source_fingerprints)):
        reasons.append("来源批次指纹记录与实际矩阵不一致")
    if not reasons:
        if joint.observed.shape != expected.observed.shape or not (
            joint.observed == expected.observed
        ).all():
            reasons.append("缺测掩码与实际矩阵不一致（缺测不得解释为已测量）")
        elif not np.allclose(
            joint.values, expected.values, rtol=0, atol=0, equal_nan=True
        ):
            reasons.append("联合矩阵的值与来源矩阵不一致")
    if reasons:
        raise PublishRejectedError(reasons)
