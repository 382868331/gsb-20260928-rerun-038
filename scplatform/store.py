"""版本化结果保存：参数变更生成新版本，永不覆盖原始数据与历史结果。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class ResultStore:
    """结果目录结构: <root>/versions/vNNNN_<param_hash>/{params,qc_report,meta}.json + normalized.npz"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    def save_run(
        self,
        *,
        params: dict,
        qc_report: dict,
        normalized: np.ndarray,
        cell_ids: list[str],
        gene_ids: list[str],
        batch_of_cell: list[str],
        rule_used: dict[str, str],
        input_fingerprint: str,
    ) -> Path:
        param_hash = self._hash(params)
        next_id = len(self.list_versions()) + 1
        out_dir = self.versions_dir / f"v{next_id:04d}_{param_hash}"
        if out_dir.exists():
            raise RuntimeError(f"结果目录已存在，拒绝覆盖: {out_dir}")
        out_dir.mkdir()

        (out_dir / "params.json").write_text(
            json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "qc_report.json").write_text(
            json.dumps(qc_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        np.savez_compressed(
            out_dir / "normalized.npz",
            normalized=normalized,
            cell_ids=np.array(cell_ids),
            gene_ids=np.array(gene_ids),
            batch_of_cell=np.array(batch_of_cell),
            rule_used=json.dumps(rule_used, ensure_ascii=False),
        )
        (out_dir / "meta.json").write_text(
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_fingerprint": input_fingerprint,
                    "n_cells": len(cell_ids),
                    "n_genes": len(gene_ids),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return out_dir

    def list_versions(self) -> list[Path]:
        return sorted(p for p in self.versions_dir.glob("v*") if p.is_dir())
