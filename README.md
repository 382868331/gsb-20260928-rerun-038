# 单细胞数据研究平台（scplatform）

一个可独立运行的小型单细胞计数矩阵分析平台，提供：

- **质控筛选**：按批次分别配置 `min_total_counts` / `min_detected_genes`，每个细胞的保留/剔除及原因全部可追溯；全零细胞与全空批次显式报警，绝不静默吞掉。
- **归一化**：按批次使用不同 `target_sum` 的文库大小缩放 + `log1p`，细胞实际使用的规则随结果存档。
- **筛选参数并行比较**：同一原始矩阵上并跑两套筛选参数，逐个细胞列出进入（`entered`）、离开（`left`）及具体原因。
- **内容寻址的版本化结果**：版本身份绑定**实际原始矩阵字节、细胞/基因身份、批次归属与全部参数**，指纹由平台内部计算；相同内容重复运行幂等，内容或参数变化必然产生新版本，已发布结果不可被后来的试验覆盖。

## 环境与依赖

- Python 3.10+
- 运行依赖见 `requirements.txt`（numpy）
- 测试依赖见 `requirements-dev.txt`（pytest）

安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## 输入文件格式

CSV 计数表（首行为表头，首列细胞 ID，第二列批次，其余为基因列）：

```csv
cell_id,batch,GeneA,GeneB,GeneC,GeneD
cell_a1,batchA,600,300,100,0
cell_b1,batchB,40,30,20,10
```

约束（违反即报错，拒绝载入）：

- 细胞 ID 不允许重复，基因 ID 不允许重复；
- 计数必须是非负整数；
- 空文件（只有表头、没有细胞行）不允许静默通过；
- 批次没有对应筛选/归一化参数时报错。

## 快速开始

```bash
python demo.py
```

`demo.py` 会：

1. 在 `example_data/toy_matrix.csv` 写入内置 6×4 示例矩阵；
2. 用严格/宽松两套筛选参数并行比较，打印进入/离开细胞及原因，并写 `results/qc_comparison.json`；
3. 分别用严格、宽松规则跑完整流程，结果存入 `results/versions/`；
4. 发布严格基线版本（写入 `published.json`）。

重复执行不会清空或覆盖任何已有版本。

作为库使用：

```python
from scplatform import (
    QCParams, NormalizeParams, ResultStore,
    load_matrix_csv, filter_cells, normalize, compare_filters,
)

matrix = load_matrix_csv("example_data/toy_matrix.csv")
qc = {
    "batchA": QCParams(min_total_counts=500, min_detected_genes=3),
    "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
}
norm = {
    "batchA": NormalizeParams(target_sum=1e4, log1p=True),
    "batchB": NormalizeParams(target_sum=1e3, log1p=True),
}

comparison = compare_filters(matrix, qc, relaxed_qc,
                             label_before="strict", label_after="relaxed")
print(comparison.format_text())

filtered, report = filter_cells(matrix, qc)
normalized, rule_used = normalize(filtered, norm)

store = ResultStore("results")
version_dir = store.save_run(
    raw_matrix=matrix,
    source_name="toy_matrix.csv",
    params={"qc": {b: p.to_dict() for b, p in qc.items()}},
    qc_report=report.to_dict(),
    normalized=normalized,
    cell_ids=list(filtered.cell_ids),
    gene_ids=list(filtered.gene_ids),
    batch_of_cell=list(filtered.batch_of_cell),
    rule_used=rule_used,
)
store.publish(version_dir)
```

## 版本身份与不可变保证

每个版本目录为 `results/versions/vNNNN_<identity_hash>/`，其中 `identity_hash` 是对以下内容的 SHA-256 摘要前缀：

- 原始矩阵字节的 SHA-256、矩阵形状与 dtype；
- 全部细胞 ID、基因 ID、细胞-批次归属；
- 完整运行参数。

目录内容：

| 文件 | 内容 |
| --- | --- |
| `identity.json` | 版本哈希、输入指纹、来源文件名、输入身份载荷 |
| `params.json` | 完整运行参数 |
| `qc_report.json` | 每个细胞的保留/剔除原因、全零细胞、全空批次 |
| `raw.npz` | 原始计数矩阵（用于离线核对） |
| `normalized.npz` | 归一化矩阵 + 细胞/基因身份 + 规则映射 |
| `meta.json` | 创建时间、版本哈希、输入指纹、规模 |
| `published.json` | 发布标记（存在即不可再写入） |

因此：**沿用同一文件名或同一调用标记不会让不同内容混成同一版本**；相同输入+相同参数重复运行幂等返回原目录；任何覆盖已写入文件或已发布版本的尝试都会抛出 `ImmutableResultError`。

## 运行测试

```bash
pytest -q
```
