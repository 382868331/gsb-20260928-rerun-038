# 单细胞数据研究平台（scplatform）

按批次执行细胞质控筛选与归一化，结果带可核对的来源指纹和参数版本：

- 细胞、基因、批次均有稳定身份；重复身份、负计数等异常输入直接拒绝。
- 输入指纹由平台根据**原始矩阵内容**自行计算（计数 + 细胞 ID + 基因 ID +
  批次归属四部分分别摘要），调用方无法通过沿用同一标记把不同内容混成同一版本。
- 版本键同时绑定输入指纹与全部参数；内容或参数任一改变都产生新版本。
- 相同输入 + 相同参数重复运行是幂等的，复用已发布版本；已发布结果目录只读，
  任何后来的试验都不能覆盖。
- 支持多套筛选参数的并行比较，逐细胞列出进入（entered）/离开（left）及原因。
- **发布前身份核对**：结果中的细胞、基因、批次关系必须与来源原始矩阵逐项一致；
  非法身份（来源中不存在的细胞/基因）、重复身份、批次错配、形状错配都会聚合为
  清楚的拒绝说明并**整次拒绝发布**（在写入任何文件之前校验，不留下暂存或部分版本）。
- **多方案合并比较**：merge_qc 明确给出所有方案的交集、仅某方案保留的细胞及
  各自排除原因；细胞身份按“来源指纹 + 细胞 ID”限定，相同名称但不同来源的细胞
  绝不合并，同名碰撞在 name_collisions 中单独列出。
- **重开核对**：ResultStore.load_run 重新打开已发布版本并逐对象核对身份；
  结果顺序调整与重复保存不改变来源身份，提供原始矩阵时还会复核输入指纹。

## 依赖

- Python 3.10+
- `numpy>=1.24`（运行必需）、`pytest>=7.0`（测试）

安装：

```bash
python -m pip install -r requirements.txt
```

## 输入文件格式

计数矩阵 CSV：首列细胞 ID、首行基因 ID，其余为非负整数计数：

```
cell,GeneA,GeneB,GeneC,GeneD
cell_a1,600,300,100,0
cell_b1,40,30,20,10
```

批次表 CSV（可选，省略时全部归入 `default` 批次）：

```
cell_id,batch
cell_a1,batchA
cell_b1,batchB
```

整个批次全零计数时不会崩溃或被悄悄吞掉：相关细胞以 `all_zero_counts` 原因
剔除，批次出现在报告的 `empty_batches` 中。

## 使用流程

```python
from scplatform import (
     QCParams, NormalizeParams, ResultStore,
    filter_cells, normalize, compare_qc, load_matrix,
)

# 1. 载入原始矩阵（也可用 scplatform.toydata.make_toy_matrix 自带数据）
matrix = load_matrix("counts.csv", "batches.csv")

# 2. 每个批次可以使用各自的筛选与归一化参数
qc_params = {
    "batchA": QCParams(min_total_counts=500, min_detected_genes=200),
    "batchB": QCParams(min_total_counts=50, min_detected_genes=2),
}
norm_params = {
    "batchA": NormalizeParams(target_sum=1e4, log1p=True),
    "batchB": NormalizeParams(target_sum=1e3, log1p=True),
}

# 3. 筛选与归一化
filtered, report = filter_cells(matrix, qc_params)
normalized, rule_used = normalize(filtered, norm_params)

# 4. 与另一套筛选参数并行比较：哪些细胞进入/离开及原因
other_qc = {"batchA": QCParams(100, 3), "batchB": QCParams(50, 2)}
comparison = compare_qc(matrix, qc_params, other_qc,
                        baseline_label="strict", candidate_label="relaxed")
print(comparison.entered_ids, comparison.left_ids)
comparison.to_dict()  # 含每个细胞在两套规则下的 kept/reasons

# 4b. 多套方案合并比较：交集 / 仅某方案保留 / 各自排除原因
from scplatform import merge_qc
merged = merge_qc(
    {"strict": matrix, "relaxed": matrix},
    {"strict": qc_params, "relaxed": other_qc},
)
merged.intersection_ids()          # 所有方案都保留的细胞
merged.only_kept_ids("relaxed")    # 仅 relaxed 保留的细胞
merged.name_collisions             # 同名但不同来源的细胞（不合并）
merged.to_dict()                   # 含每个细胞被各方案排除的原因

# 5. 保存（指纹由平台从原始矩阵计算，无需也不能由调用方指定）
store = ResultStore("results")
params = {
    "qc": {b: p.to_dict() for b, p in qc_params.items()},
    "normalize": {b: p.to_dict() for b, p in norm_params.items()},
}
out_dir = store.save_run(
    source_matrix=matrix,
    params=params,
    qc_report=report.to_dict(),
    normalized=normalized,
    cell_ids=list(filtered.cell_ids),
    gene_ids=list(filtered.gene_ids),
    batch_of_cell=list(filtered.batch_of_cell),
    rule_used=rule_used,
    comparison=comparison.to_dict(),
)
```

## 结果目录与来源核对

每个版本位于 `results/versions/vNNNN_<input_hash>_<param_hash>/`：

- `input_identity.json`：整体输入指纹及计数、细胞 ID、基因 ID、批次归属
  四个组成部分的独立摘要，以及完整身份清单，用于逐项核对来源。
- `params.json`：本次运行的全部逐批次参数。
- `qc_report.json`：每个被剔除细胞的原因、全零细胞、空批次。
- `qc_comparison.json`：并行比较时逐细胞的进入/离开及两套规则下的原因。
- `normalized.npz`：归一化矩阵及随附的细胞/基因身份、批次规则使用记录。
- `meta.json`：创建时间、输入/参数哈希、`published` 标记。

核对方式：重新载入原始矩阵后比较 `input_identity.json` 中的摘要即可；
文件改名不影响指纹，任何计数或身份改动都会让摘要变化并产生新版本。

### 发布前拒绝与重开核对

`save_run` 在创建版本目录之前执行完整身份校验，任何不匹配都抛出
`scplatform.PublishRejected`，其 `errors` 列表聚合全部问题（例如同时存在
重复细胞身份和批次错配时，两条说明都会保留）：

- 细胞/基因身份在原始矩阵中不存在（非法身份）或在结果中重复；
- 结果细胞的批次归属或 `rule_used` 与其来源批次错配；
- 归一化矩阵行列数与随附身份不一致，或基因列被增删；
- `qc_report.kept` 与实际结果细胞集合不一致。

拒绝时不写入任何文件、不创建暂存目录，已发布版本不受影响。重新打开版本用：

```python
bundle = store.load_run(out_dir, source_matrix=load_matrix("counts.csv", "batches.csv"))
bundle["cell_ids"]          # 每个保留对象都能逐一对回来源
bundle["batch_of_cell"]
```

来源矩阵内容变化（指纹不一致）或版本内身份被破坏时，`load_run` 同样抛出
`PublishRejected` 并给出全部不匹配说明。结果行顺序变化不影响身份核对。

## 演示与测试

```bash
python demo.py            # 内置数据端到端演示（严格/宽松规则比较、幂等重跑）
python -m pytest -q       # 全部行为核对
```
