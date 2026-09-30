"""版本化结果保存：版本键绑定原始矩阵内容、细胞/基因身份与全部参数。

输入指纹由平台根据原始 CountMatrix 自行计算，调用方无法通过沿用同一
标记把不同内容混成同一版本。已发布结果永不被后来的运行覆盖。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model import CountMatrix


class PublishRejected(ValueError):
    """发布被拒绝：结果身份与来源不一致，整次发布不留下任何部分版本。

    ``errors`` 按类别收集全部不匹配说明（非法身份 / 重复身份 / 批次错配 /
    形状错配 / 规则与报告错配），便于调用方逐条排查。
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("结果与来源不一致，拒绝整次发布:\n- " + "\n- ".join(self.errors))


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
        identity = source_matrix.identity_digests()
        input_hash = source_matrix.content_fingerprint()[:12]
        param_hash = self._hash(params)[:12]
        version_key = f"{input_hash}_{param_hash}"

        # 发布前先核对结果中的细胞、基因、批次关系与原始矩阵是否一致。
        # 任何不匹配都在写任何文件之前抛出，绝不留下看似有效的部分版本。
        self._validate_against_source(
            source_matrix=source_matrix,
            qc_report=qc_report,
            normalized=normalized,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            batch_of_cell=batch_of_cell,
            rule_used=rule_used,
        )

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
                    "batch_of_cell": list(source_matrix.batch_of_cell),
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
        staging.rename(out_dir)
        # 锁定为只读：已发布结果不能被后来的试验改写。
        for path in [*out_dir.rglob("*"), out_dir]:
            if path.is_file():
                path.chmod(0o444)
        return out_dir

    @staticmethod
    def _validate_against_source(
        *,
        source_matrix: CountMatrix,
        qc_report: dict,
        normalized: np.ndarray,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        rule_used: dict[str, str],
    ) -> None:
        """逐对象核对结果身份与来源；聚合全部问题后一次性拒绝。"""
        errors: list[str] = []

        normalized = np.asarray(normalized)
        if normalized.ndim != 2:
            errors.append(f"归一化矩阵必须是二维数组，实际为 {normalized.ndim} 维")

        source_cells = list(source_matrix.cell_ids)
        source_genes = list(source_matrix.gene_ids)
        source_batch = dict(zip(source_cells, source_matrix.batch_of_cell))
        cell_set = set(source_cells)
        gene_set = set(source_genes)

        # ---- 细胞身份：数量、重复、非法（来源不存在）----
        if len(cell_ids) != len(batch_of_cell):
            errors.append(
                f"cell_ids 数量({len(cell_ids)})与 batch_of_cell 数量"
                f"({len(batch_of_cell)})不一致"
            )
        cell_dupes = sorted({c for c in cell_ids if cell_ids.count(c) > 1})
        if cell_dupes:
            errors.append(f"结果中存在重复细胞身份: {cell_dupes}")
        illegal_cells = sorted(c for c in cell_ids if c not in cell_set)
        if illegal_cells:
            errors.append(
                f"细胞身份不存在于原始矩阵（非法身份，禁止挂在原输入指纹下发布）: "
                f"{illegal_cells}"
            )

        # ---- 基因身份：重复、非法、与列数对齐 ----
        gene_dupes = sorted({g for g in gene_ids if gene_ids.count(g) > 1})
        if gene_dupes:
            errors.append(f"结果中存在重复基因身份: {gene_dupes}")
        illegal_genes = sorted(g for g in gene_ids if g not in gene_set)
        if illegal_genes:
            errors.append(f"基因身份不存在于原始矩阵（非法身份）: {illegal_genes}")
        missing_genes = sorted(g for g in source_genes if g not in gene_ids)
        if missing_genes:
            errors.append(f"结果丢失原始矩阵中的基因列: {missing_genes}")

        # ---- 形状：行列必须与随附身份严格对齐 ----
        if normalized.ndim == 2:
            if normalized.shape[0] != len(cell_ids):
                errors.append(
                    f"归一化矩阵行数({normalized.shape[0]})与细胞数"
                    f"({len(cell_ids)})不一致"
                )
            if normalized.shape[1] != len(gene_ids):
                errors.append(
                    f"归一化矩阵列数({normalized.shape[1]})与基因数"
                    f"({len(gene_ids)})不一致（归一化不得增删基因）"
                )

        # ---- 批次错配：随附批次必须等于原始矩阵中该细胞的归属 ----
        batch_mismatch: list[str] = []
        for cell_id, batch in zip(cell_ids, batch_of_cell):
            if cell_id in source_batch and batch != source_batch[cell_id]:
                batch_mismatch.append(
                    f"{cell_id}: 结果批次={batch!r}，来源批次={source_batch[cell_id]!r}"
                )
        if batch_mismatch:
            errors.append("细胞批次归属与来源错配（批次错配）: " + "; ".join(batch_mismatch))

        # ---- rule_used：键必须恰好是结果细胞，值必须是该细胞的来源批次 ----
        if not isinstance(rule_used, dict):
            errors.append("rule_used 必须是 cell_id -> 批次规则名 的字典")
        else:
            rule_keys = set(rule_used)
            result_cell_set = set(cell_ids)
            missing_rules = sorted(result_cell_set - rule_keys)
            extra_rules = sorted(rule_keys - result_cell_set)
            if missing_rules:
                errors.append(f"以下结果细胞缺少 rule_used 规则记录: {missing_rules}")
            if extra_rules:
                errors.append(f"rule_used 含结果中不存在的细胞: {extra_rules}")
            rule_mismatch = sorted(
                f"{cid}: 规则批次={rule_used[cid]!r}，来源批次={source_batch[cid]!r}"
                for cid in rule_keys & result_cell_set
                if cid in source_batch and rule_used[cid] != source_batch[cid]
            )
            if rule_mismatch:
                errors.append("rule_used 与来源批次错配: " + "; ".join(rule_mismatch))

        # ---- QC 报告：结果细胞必须恰好是报告中被保留的细胞 ----
        kept = qc_report.get("kept") if isinstance(qc_report, dict) else None
        if not isinstance(kept, list):
            errors.append("qc_report 必须包含 kept 细胞身份列表")
        else:
            kept_dupes = sorted({c for c in kept if kept.count(c) > 1})
            if kept_dupes:
                errors.append(f"qc_report.kept 存在重复细胞身份: {kept_dupes}")
            kept_set = set(kept)
            illegal_kept = sorted(c for c in kept_set if c not in cell_set)
            if illegal_kept:
                errors.append(
                    f"qc_report.kept 含原始矩阵中不存在的细胞（非法身份）: {illegal_kept}"
                )
            result_set = set(cell_ids)
            not_in_result = sorted(kept_set - result_set)
            not_in_report = sorted(result_set - kept_set)
            if not_in_result:
                errors.append(f"qc_report 保留但结果中缺失的细胞: {not_in_result}")
            if not_in_report:
                errors.append(f"结果包含但 qc_report 未保留的细胞: {not_in_report}")

        if errors:
            raise PublishRejected(errors)

    def load_run(
        self, version_dir: str | Path, *, source_matrix: CountMatrix | None = None
    ) -> dict:
        """重开已发布版本，核对每个保留对象仍能对回来源。

        始终执行自洽核对（身份无重复、矩阵行列对齐、批次/规则一致）；
        提供 ``source_matrix`` 时额外核对指纹与逐对象来源身份。
        核对失败抛出 :class:`PublishRejected`（含全部不匹配说明）。
        """
        version_dir = Path(version_dir)
        errors: list[str] = []

        meta_path = version_dir / "meta.json"
        identity_path = version_dir / "input_identity.json"
        if not meta_path.exists() or not identity_path.exists():
            raise PublishRejected([f"版本目录不完整，缺少 meta/identity 文件: {version_dir}"])

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        params = json.loads((version_dir / "params.json").read_text(encoding="utf-8"))
        qc_report = json.loads((version_dir / "qc_report.json").read_text(encoding="utf-8"))
        comparison_path = version_dir / "qc_comparison.json"
        comparison = (
            json.loads(comparison_path.read_text(encoding="utf-8"))
            if comparison_path.exists()
            else None
        )
        bundle = np.load(version_dir / "normalized.npz", allow_pickle=False)
        normalized = bundle["normalized"]
        cell_ids = [str(c) for c in bundle["cell_ids"]]
        gene_ids = [str(g) for g in bundle["gene_ids"]]
        batch_of_cell = [str(b) for b in bundle["batch_of_cell"]]
        rule_used = json.loads(str(bundle["rule_used"]))

        if source_matrix is not None:
            current_fp = source_matrix.content_fingerprint()
            if current_fp != identity["input_fingerprint"]:
                errors.append(
                    "输入指纹与发布时不一致：来源矩阵内容已变化，"
                    f"发布指纹={identity['input_fingerprint'][:12]}，"
                    f"当前指纹={current_fp[:12]}"
                )
            if f"{current_fp[:12]}_{meta['param_hash']}" != version_dir.name.split("_", 1)[1]:
                errors.append("版本目录的输入指纹前缀与当前来源矩阵不匹配")

        try:
            self._validate_against_source(
                source_matrix=source_matrix
                if source_matrix is not None
                else CountMatrix(
                    counts=np.zeros((len(identity["cell_ids"]), len(identity["gene_ids"])),
                                   dtype=np.int64),
                    cell_ids=tuple(identity["cell_ids"]),
                    gene_ids=tuple(identity["gene_ids"]),
                    batch_of_cell=tuple(
                        self._batch_from_identity(identity, cid)
                        for cid in identity["cell_ids"]
                    ),
                ),
                qc_report=qc_report,
                normalized=normalized,
                cell_ids=cell_ids,
                gene_ids=gene_ids,
                batch_of_cell=batch_of_cell,
                rule_used=rule_used,
            )
        except PublishRejected as exc:
            errors.extend(exc.errors)

        if errors:
            raise PublishRejected(errors)

        return {
            "version_dir": version_dir,
            "meta": meta,
            "identity": identity,
            "params": params,
            "qc_report": qc_report,
            "comparison": comparison,
            "normalized": normalized,
            "cell_ids": cell_ids,
            "gene_ids": gene_ids,
            "batch_of_cell": batch_of_cell,
            "rule_used": rule_used,
        }

    @staticmethod
    def _batch_from_identity(identity: dict, cell_id: str) -> str:
        """从 identity 中读取发布时记录的逐细胞批次归属。"""
        per_cell = identity.get("batch_of_cell")
        if not isinstance(per_cell, list):
            raise PublishRejected(["input_identity.json 缺少逐细胞批次归属记录"])
        idx = identity["cell_ids"].index(cell_id)
        return str(per_cell[idx])

    def find_version(self, matrix: CountMatrix, params: dict) -> Path | None:
        """按输入内容+参数精确定位已发布版本；内容不同不会命中。"""
        version_key = f"{matrix.content_fingerprint()[:12]}_{self._hash(params)[:12]}"
        for existing in self.list_versions():
            if existing.name.split("_", 1)[1] == version_key:
                return existing
        return None

    def list_versions(self) -> list[Path]:
        return sorted(p for p in self.versions_dir.glob("v*") if p.is_dir())
