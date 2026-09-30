"""版本化结果保存。

版本身份由平台计算，绑定三要素：

1. 原始计数矩阵的字节内容（shape/dtype 纳入）；
2. 细胞 ID、基因 ID 与细胞-批次归属；
3. 完整运行参数。

三者任一改变都会得到不同版本；调用方无法再随手传入指纹。
结果目录一经写入即不可变，发布后更不允许后来的试验覆盖。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .model import CountMatrix

PUBLISHED_MARKER = "published.json"
_ARTIFACTS = (
    "params.json",
    "qc_report.json",
    "normalized.npz",
    "meta.json",
    "identity.json",
    "raw.npz",
)


class ImmutableResultError(RuntimeError):
    """试图覆盖或改写已经写入/发布的版本。"""


class ResultStore:
    """结果目录: <root>/versions/vNNNN_<identity_hash>/"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def version_hash(input_payload: dict, params: dict) -> str:
        blob = json.dumps(
            {"input": input_payload, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def list_versions(self) -> list[Path]:
        return sorted(
            (p for p in self.versions_dir.glob("v*_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_", 1)[0][1:]),
        )

    def _next_version_id(self) -> int:
        ids = [int(p.name.split("_", 1)[0][1:]) for p in self.versions_dir.glob("v*_*")]
        return (max(ids) + 1) if ids else 1

    @staticmethod
    def _verify_filtered_identities(
        raw: CountMatrix,
        *,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        normalized: np.ndarray,
    ) -> None:
        if tuple(gene_ids) != raw.gene_ids:
            raise ValueError("保存的基因身份与原始矩阵不一致，拒绝保存")
        raw_index = {
            cell_id: (i, raw.batch_of_cell[i]) for i, cell_id in enumerate(raw.cell_ids)
        }
        for cell_id, batch in zip(cell_ids, batch_of_cell):
            if cell_id not in raw_index:
                raise ValueError(f"细胞 {cell_id} 不属于原始矩阵，拒绝保存")
            if raw_index[cell_id][1] != batch:
                raise ValueError(f"细胞 {cell_id} 的批次归属与原始矩阵不一致")
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("保存的细胞身份存在重复")
        normalized = np.asarray(normalized)
        if normalized.shape != (len(cell_ids), len(gene_ids)):
            raise ValueError("归一化矩阵形状与细胞/基因身份不一致")

    def _atomic_write(self, version_dir: Path, name: str, data: str | bytes) -> None:
        target = version_dir / name
        if target.exists():
            raise ImmutableResultError(f"结果文件已存在，拒绝覆盖: {target}")
        if (version_dir / PUBLISHED_MARKER).exists():
            raise ImmutableResultError(f"版本已发布，拒绝写入: {version_dir}")
        tmp = target.with_suffix(target.suffix + ".tmp")
        if isinstance(data, str):
            tmp.write_text(data, encoding="utf-8")
        else:
            tmp.write_bytes(data)
        os.replace(tmp, target)

    def save_run(
        self,
        *,
        raw_matrix: CountMatrix,
        params: dict,
        qc_report: dict,
        normalized: np.ndarray,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        rule_used: dict[str, str],
        source_name: str | None = None,
    ) -> Path:
        normalized = np.asarray(normalized)
        self._verify_filtered_identities(
            raw_matrix,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            batch_of_cell=batch_of_cell,
            normalized=normalized,
        )

        identity_payload = raw_matrix.identity_payload()
        fingerprint = raw_matrix.fingerprint()
        identity_hash = self.version_hash(identity_payload, params)

        for existing in self.versions_dir.glob(f"v*_{identity_hash}"):
            return self._reuse_identical(existing)

        version_id = self._next_version_id()
        out_dir = self.versions_dir / f"v{version_id:04d}_{identity_hash}"
        out_dir.mkdir()
        try:
            self._atomic_write(
                out_dir,
                "identity.json",
                json.dumps(
                    {
                        "version_hash": identity_hash,
                        "input_fingerprint": fingerprint,
                        "source_name": source_name,
                        "input": identity_payload,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            self._atomic_write(
                out_dir,
                "params.json",
                json.dumps(params, indent=2, ensure_ascii=False),
            )
            self._atomic_write(
                out_dir,
                "qc_report.json",
                json.dumps(qc_report, indent=2, ensure_ascii=False),
            )
            raw_buffer = io.BytesIO()
            np.savez_compressed(raw_buffer, raw_counts=raw_matrix.counts)
            self._atomic_write(out_dir, "raw.npz", raw_buffer.getvalue())
            norm_buffer = io.BytesIO()
            np.savez_compressed(
                norm_buffer,
                normalized=normalized,
                cell_ids=np.array(cell_ids),
                gene_ids=np.array(gene_ids),
                batch_of_cell=np.array(batch_of_cell),
                rule_used=json.dumps(rule_used, ensure_ascii=False),
            )
            self._atomic_write(out_dir, "normalized.npz", norm_buffer.getvalue())
            self._atomic_write(
                out_dir,
                "meta.json",
                json.dumps(
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "version_hash": identity_hash,
                        "input_fingerprint": fingerprint,
                        "source_name": source_name,
                        "n_cells": len(cell_ids),
                        "n_genes": len(gene_ids),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
        except BaseException:
            for artifact in reversed(_ARTIFACTS):
                (out_dir / artifact).unlink(missing_ok=True)
            out_dir.rmdir()
            raise
        return out_dir

    def _reuse_identical(self, version_dir: Path) -> Path:
        missing = [name for name in _ARTIFACTS if not (version_dir / name).exists()]
        if missing:
            raise ImmutableResultError(
                f"版本内容与 {version_dir.name} 相同但产物不完整 {missing}，不可变存储拒绝补写"
            )
        return version_dir

    def publish(self, version_dir: str | Path) -> Path:
        """标记版本为已发布；重复发布幂等，发布后任何写入都会被拒绝。"""
        version_dir = Path(version_dir)
        if not version_dir.is_relative_to(self.versions_dir):
            raise ValueError(f"不在结果仓库内: {version_dir}")
        missing = [name for name in _ARTIFACTS if not (version_dir / name).exists()]
        if missing:
            raise ImmutableResultError(f"版本产物不完整，拒绝发布: {missing}")
        marker = version_dir / PUBLISHED_MARKER
        if marker.exists():
            return version_dir
        payload = json.dumps(
            {"published_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
            ensure_ascii=False,
        )
        self._atomic_write(version_dir, PUBLISHED_MARKER, payload)
        return version_dir

    def is_published(self, version_dir: str | Path) -> bool:
        return (Path(version_dir) / PUBLISHED_MARKER).exists()
