# 单细胞数据研究平台（scplatform）

按批次执行细胞质控筛选与归一化，结果带可核对的来源指纹和参数版本：

- 细胞、基因、批次均有稳定身份；重复身份、负计数等异常输入直接拒绝。
- 输入指纹由平台根据**原始矩阵内容**自行计算（计数 + 细胞 ID + 基因 ID +
  批次归属四部分分别摘要），调用方无法通过沿用同一标记把不同内容混成同一版本。
- 版本键同时绑定输入指纹与全部参数；内容或参数任一改变都产生新版本。
- 相同输入 + 相同参数重复运行是幂等的，复用已发布版本；已发布结果目录只读，
  任何后来的试验都不能覆盖。
- 支持多套筛选参数的并行比较，逐细胞列出进入（entered）/离开（left）及原因。
  合并视图同时给出交集、仅某方案保留的细胞及各自排除原因；同名但不同
  来源（指纹不同）的细胞绝不合并。
- 发布前强制核对：结果中的细胞、基因、批次关系必须与来源矩阵一致，
  非法身份、重复身份、批次错配都会以 `PublishRejectedError` 整次拒绝，
  不留下任何部分版本；`store.verify_version(...)` 可在重开后逐细胞复核。

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


## 跨批次联合分析集

不同采集批次可以合并为一个联合分析集，基因允许重叠但不完全相同：

- 每个值都带 `measured` 掩码：`True` 为真实测量，`False` 为该批次未测；
  缺测位置以 NaN 存储，**绝不把缺测解释成零表达**（真实零计数仍是测量值 0）。
- `join_batches(sources, "common")` 只用共同基因；`"union"` 保留全部基因，
  `missing_genes_by_batch()` 明确每个批次的缺测范围。
- `compare_join_modes(sources, qc)` 比较两种选择对保留细胞与质量判断的影响
  （共同基因更少时检出基因数阈值可能误判，差异逐细胞列出）。
- 细胞显示名相同仍按来源区分：内部身份为 `批次::细胞ID`，绝不按显示名合并。
- `store.save_joint(joint=..., sources=..., params=...)` 发布前用实际矩阵重新
  联合核对：采用细胞、基因顺序、缺测说明任一不符都整次拒绝，不留下混杂版本；
  批次补测基因后来源指纹变化，自动生成新联合版本而不覆盖旧分析。
- `store.load_joint(...)` / `store.verify_joint(...)` 重开后逐细胞、逐组测量
  追溯核对；`store.withdraw_batch(name)` 撤回批次只把依赖它的联合版本标记
  失效，其余版本不受影响。

```python
from scplatform import join_batches, compare_join_modes

sources = {"batchA": matrix_a, "batchB": matrix_b}  # 各自批次的 CountMatrix
joint = join_batches(sources, "union")
comparison = compare_join_modes(sources, qc_params)
out = store.save_joint(joint=joint, sources=sources, params={"note": "v1"})
```

## 同细胞多模态关联（基因表达 × 蛋白）

两个采集表的细胞集合不必相同，显示名也可能重复；关联只接受有明确
来源依据的配对：

- `suggest_pairing(modalities, candidate_keys=None)` 按**稳定细胞身份**
  （共享条码/孔位，可通过 `candidate_keys` 提供“细胞ID→共享条码”映射）
  提候选：所有模态唯一命中记为 `shared_key` 自动配对；同一键在某模态
  命中多个细胞形成一对多候选组，**保持未配对**，绝不按显示名合并；
  只在部分模态出现的细胞记为 `missing_modality`。重复显示名仅在
  `pairing.duplicate_display_names` 中登记核对。
- `pairing.with_resolved(group_id, {模态: 选中的细胞ID}, decided_by=...)`
  做人工选择：选择必须来自候选快照（来源不明的细胞直接拒绝），决定以
  `evidence="explicit"` 连同候选人与选择人一并保存。旧 `Pairing`
  不被改写，因此重新选择后旧联合版本依旧可解释。
- `ModalityMatrix` 为每个模态保留自己的字段（基因/蛋白）顺序、逐字段
  单位（如 counts / CLR）和 `measurement_version`。模态表内的零是
  **真实测得的零**；某行整模态缺测在联合矩阵中是 NaN 且 `measured=False`，
  两者严格区分，缺测绝不计作零。
- 两种分析范围可比较：`compare_scopes(...)` 分别给出只含已配对细胞的
  `paired` 范围与保留所有细胞的 `all` 范围，逐细胞标明缺少哪种测量，
  并列出范围选择对保留/剔除判断的影响。
- `store.save_multimodal(...)` 发布前用实际测量表按配对决定重新构造
  联合矩阵核对：行集合、各模态字段顺序、单位、测量版本、缺测掩码或
  测量值不符都以 `PublishRejectedError` 整次拒绝；相同输入幂等复用，
  补测或重新选择产生新版本而不覆盖旧版本。`load_multimodal` /
  `load_pairing` / `verify_multimodal` 支持发布中断后重开，矩阵范围、
  配对决定与缺测说明保持一致。
- `store.register_downstream_screen(...)` 登记基于某联合版本的下游筛选；
  `withdraw_association(version, pair_cell_id)` 撤回单条关联、
  `withdraw_modality(name)` 撤回整个采集批次（或 `fingerprint=` 指定
  补测版本）、`remeasure_modality(...)` 登记补测；这些操作返回
  `affected_multimodal_versions` 与 `affected_downstream_screens` 清单，
  未依赖该资料的测量与版本不受牵连。

```python
from scplatform import (
    ModalityMatrix, ModalityQCParams, ResultStore,
    suggest_pairing, build_multimodal, compare_scopes,
)

rna = ModalityMatrix(values=rna_values, cell_ids=("bc1", "bc2"),
                     feature_ids=("GeneA", "GeneB"), modality="rna",
                     units=("counts", "counts"), measurement_version="rna-2026-09")
protein = ModalityMatrix(values=prot_values, cell_ids=("pA", "pB"),
                         feature_ids=("CD3", "CD19"), modality="protein",
                         units=("CLR", "CLR"), measurement_version="prot-2026-09")
keys = {"rna": {...}, "protein": {"pA": "bc1", ...}}

pairing = suggest_pairing({"rna": rna, "protein": protein}, keys)
for group in pairing.ambiguous:          # 一对多候选由人工选择
    pairing = pairing.with_resolved(group.group_id, choice, decided_by="alice")

multi = build_multimodal({"rna": rna, "protein": protein}, pairing)
impact = compare_scopes(
    {"rna": rna, "protein": protein}, pairing,
    {"rna": ModalityQCParams(), "protein": ModalityQCParams()},
).to_dict()  # paired 与 all 两种范围的筛选与缺测差异

store = ResultStore("results")
version = store.save_multimodal(
    multi=multi, modalities={"rna": rna, "protein": protein},
    pairing=pairing, params={"scope": "paired"},
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

## 演示与测试

```bash
python demo.py            # 内置数据端到端演示（严格/宽松规则比较、幂等重跑）
python -m pytest -q       # 全部行为核对
```
