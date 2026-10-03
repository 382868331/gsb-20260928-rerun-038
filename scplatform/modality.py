"""单个模态的测量表（如基因表达、表面蛋白）。

与 CountMatrix 不同，ModalityMatrix 显式携带模态名、字段单位与
测量版本；字段顺序是该模态自己的测量顺序，联合时不重排。
缺测（该模态完全未测/未覆盖）与真实零值严格区分：

- 表里出现的每一行都是该模态真实测得的细胞，零是测量值 0；
- 根本没有出现在该模态中的细胞属于“缺少整种测量”，
  由多模态关联层标记，绝不用零值冒充。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .model import CountMatrix


@dataclass(frozen=True)
class ModalityMatrix:
    """cells x features 的单模态测量表。

    modality          : 模态身份，如 "rna" / "protein"，全局唯一；
    values            : 测量值（可为浮点），零是真实测得的零；
    cell_ids          : 该模态实际测得的细胞稳定身份；
    cell_display_names: 展示名，可重复，绝不作为配对依据；
    feature_ids       : 该模态自己的字段（基因/蛋白）顺序；
    units             : 逐字段单位（如 counts / CLR / dsb）；
    measurement_version: 该模态采集/处理版本（补测后通常变化）；
    batch_of_cell     : 采集批次归属。
    """

    values: np.ndarray
    cell_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    modality: str = "rna"
    cell_display_names: tuple[str, ...] | None = None
    units: tuple[str, ...] | None = None
    measurement_version: str = "v1"
    batch_of_cell: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("测量表必须是二维矩阵")
        n_cells, n_features = values.shape
        if len(self.cell_ids) != n_cells:
            raise ValueError("cell_ids 数量与矩阵行数不一致")
        if len(self.feature_ids) != n_features:
            raise ValueError("feature_ids 数量与矩阵列数不一致")
        if len(set(self.cell_ids)) != n_cells:
            raise ValueError(f"模态 {self.modality} 存在重复细胞 ID，拒绝构造")
        if len(set(self.feature_ids)) != n_features:
            dupes = sorted(
                {f for f in self.feature_ids if self.feature_ids.count(f) > 1}
            )
            raise ValueError(f"模态 {self.modality} 存在重复字段 ID: {dupes}")
        if np.isnan(values).any():
            raise ValueError(
                f"模态 {self.modality} 的测量表内不允许 NaN："
                "未覆盖的细胞应不出现在该模态，而不是存成 NaN"
            )
        display = self.cell_display_names
        if display is None:
            display = tuple(self.cell_ids)
        if len(display) != n_cells:
            raise ValueError("cell_display_names 数量与细胞数不一致")
        units = self.units
        if units is None:
            units = tuple("" for _ in self.feature_ids)
        if len(units) != n_features:
            raise ValueError("units 数量与字段数不一致")
        batches = self.batch_of_cell
        if batches is None:
            batches = tuple("default" for _ in self.cell_ids)
        if len(batches) != n_cells:
            raise ValueError("batch_of_cell 数量与细胞数不一致")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cell_display_names", tuple(display))
        object.__setattr__(self, "units", tuple(units))
        object.__setattr__(self, "batch_of_cell", tuple(batches))

    @property
    def n_cells(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]

    @property
    def batches(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.batch_of_cell)))

    def subset_cells(self, keep_ids: list[str]) -> "ModalityMatrix":
        idx = [self.cell_ids.index(c) for c in keep_ids]
        return ModalityMatrix(
            values=self.values[idx],
            cell_ids=tuple(self.cell_ids[i] for i in idx),
            feature_ids=self.feature_ids,
            modality=self.modality,
            cell_display_names=tuple(self.cell_display_names[i] for i in idx),
            units=self.units,
            measurement_version=self.measurement_version,
            batch_of_cell=tuple(self.batch_of_cell[i] for i in idx),
        )

    @staticmethod
    def _digest(payload: object) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def identity_digests(self) -> dict[str, str]:
        """分别摘要测量值、细胞身份、字段身份、单位、测量版本与批次归属。"""
        values_payload = {
            "dtype": str(self.values.dtype),
            "shape": list(self.values.shape),
            "data": self.values.tobytes().hex(),
        }
        return {
            "values": self._digest(values_payload),
            "cell_ids": self._digest(list(self.cell_ids)),
            "cell_display_names": self._digest(list(self.cell_display_names)),
            "feature_ids": self._digest(list(self.feature_ids)),
            "units": self._digest(list(self.units)),
            "measurement_version": self._digest(self.measurement_version),
            "batch_of_cell": self._digest(list(self.batch_of_cell)),
        }

    def content_fingerprint(self) -> str:
        return self._digest({"modality": self.modality, **self.identity_digests()})

    @classmethod
    def from_count_matrix(
        cls,
        matrix: CountMatrix,
        *,
        modality: str = "rna",
        measurement_version: str = "v1",
        units: tuple[str, ...] | None = None,
    ) -> "ModalityMatrix":
        """把已有 CountMatrix 适配为一个模态；计数 0 仍是真实测得的零。"""
        if units is None:
            units = tuple("counts" for _ in matrix.gene_ids)
        return cls(
            values=matrix.counts.astype(np.float64),
            cell_ids=tuple(matrix.cell_ids),
            feature_ids=tuple(matrix.gene_ids),
            modality=modality,
            cell_display_names=tuple(matrix.cell_ids),
            units=units,
            measurement_version=measurement_version,
            batch_of_cell=tuple(matrix.batch_of_cell),
        )
