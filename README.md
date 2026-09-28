# 单细胞计数数据研究平台（最小可运行版）

面向小型计数矩阵，先打通三条基础链路：

- 细胞、基因、样本批次使用稳定字符串 ID，后续聚类可直接沿用筛选后矩阵的行列名。
- 质量筛选逐细胞记录保留/剔除原因；空批次和全零细胞进入校验报告；重复基因直接报错。
- 原始矩阵不可变；QC 和归一化结果按原始数据指纹加参数生成独立版本目录。

## 快速核对

```bash
.venv/bin/python demo.py
.venv/bin/python -m pytest -q
```

内置 4 基因 x 7 细胞矩阵包含全零细胞 A3 和空批次 batch_C。演示中 A1/A2、B1/B2 通过各自批次规则；A3/A4/B3 被剔除并在 cell_decisions.csv 中给出原因。

批次 A 的归一化目标为 100，所以 A1 的 G2 为 9/10*100=90；批次 B 的目标为 1000，所以 B1 的 G3 为 5/10*1000=500。第二次运行放宽批次 A 规则后 A4 被保留，并生成新的 QC/归一化版本，原始矩阵和第一版结果不变。

## 版本目录

- `results_demo/raw/<dataset_id>__<fingerprint>/raw_counts.csv`：不可变原始矩阵。
- `results_demo/qc/<fingerprint>/<qc_id>/`：筛选矩阵、逐细胞决策和 provenance。
- `results_demo/norm/<qc_id>/<norm_id>/`：归一化矩阵、逐细胞缩放因子和 provenance。

`qc_id` 由原始数据指纹和各批次 QC 参数哈希得到；`norm_id` 再绑定 `qc_id` 和各批次归一化参数。参数变化只新增版本，不覆盖旧结果。

## 核心 API

```python
from sc_platform import build_dataset, QcParams, NormalizationParams, ResultStore

dataset = build_dataset(
    dataset_id="mini",
    matrix=[[1, 0], [9, 5]],
    gene_ids=["G1", "G2"],
    cell_ids=["A1", "B1"],
    batches={"batch_A": ["A1"], "batch_B": ["B1"], "batch_empty": []},
)
store = ResultStore("results")
_, _, _, qc, normalized = store.process(
    dataset,
    {
        "batch_A": QcParams(min_total_counts=5),
        "batch_B": QcParams(min_total_counts=5),
        "batch_empty": QcParams(),
    },
    {
        "batch_A": NormalizationParams(target_sum=100),
        "batch_B": NormalizationParams(target_sum=1000),
        "batch_empty": NormalizationParams(target_sum=100),
    },
)
```

批次规则以批次名为键分别传入；细胞 QC、基因检测数和归一化缩放因子均在各自批次内计算，不跨批次混用。
