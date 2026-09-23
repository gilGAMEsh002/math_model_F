# Step 1 数据预处理

把 SlimPajama-Meta-rater 质量信号 A1/A2/A3 转成统一尺度的标准化质量矩阵 `Z ∈ [0,1]^(N×22)`。

## 目录结构

工作根目录为 `D:\26mathmodel\q1_preprocessing`，代码用相对路径定位，整个文件夹可独立拷贝。

```
q1_preprocessing\
├── real_attachments\   原始数据副本（551,191,693 字节，与原仓库逐字节一致）
├── src\preprocess\     代码
├── configs\            冻结参数
├── artifacts\q1\       产出
└── reports\            报告
```

本目录的这份是**唯一的数据副本**（2012 个文件，530M），请勿删改 `real_attachments/`。
原先 `D:\26mathmodel\math_model_F\real_attachments` 下另有一份逐字节相同的仓库副本，为省空间已删除。

如需取回仓库那份（无需联网，LFS 对象缓存在 `math_model_F/.git/lfs`，296M）：

```bash
cd /d/26mathmodel/math_model_F && git checkout -- real_attachments
```

删除后 `math_model_F` 的 `git status` 会显示 2012 条待提交删除，属预期；只要不 commit，
`git checkout -- real_attachments` 随时可还原（已实测）。

## 运行

```bash
python src/preprocess/quality_preprocess.py   # 预处理，约 20 秒
python src/preprocess/verify_preprocess.py    # 独立验证，46 项检查
```

## 输出

| 路径 | 内容 |
|---|---|
| `artifacts/q1/preprocessed/A1_processed.parquet` | 51,230 × 25（`id`, `domain`, 22 个 z 列, `source_set`） |
| `artifacts/q1/preprocessed/A2_processed.parquet` | 17,523 × 25（arxiv） |
| `artifacts/q1/preprocessed/A3_processed.parquet` | 203,752 × 25（github） |
| `configs/quality_preprocess_params.json` | 全部冻结参数 |
| `reports/q1_preprocess_report.md` | 缺失、参数、截断率、分布表 |

## 处理链

```
原始 JSONL (27/24 字段)
  └─ 字段筛选        仅保留 22 个质量指标
  └─ 缺失值处理      NaN/None -> A1 中位数（标量）或 A1 均值向量（列表型）
  └─ 列表型压缩      单元素直取 / 二分类 softmax 取高质类 / 四维逐维 MinMax 后平均 / 六分类 softmax 期望等级
  └─ 异常值处理      dsir_books|math|wiki -> sign(x)·log1p(|x|)
  └─ 方向统一        正向 MinMax；负向补转换 1-MinMax；非单调 → log1p 空间梯形隶属度
  └─ Z ∈ [0,1]^(N×22)
```

## 关键约定

**参考集冻结。** 所有可调参数（中位数、均值向量、MinMax 边界、qurater 逐维边界、梯形隶属度分位数）
一律在 A1 上估计后原样套用到 A2/A3。这样三套数据的 Z 处于同一尺度，赛题要求的
「扩展集域级 Q 与抽样集对照」才有意义。代价是 A2/A3 有极少量值超出 A1 的 [min,max]，
已截断到 [0,1]（A3 约 0.004%），并在报告第 3 节逐指标统计。

## 建模决策记录（非题面给定，需在论文中说明）

| 决策 | 取法 | 依据 |
|---|---|---|
| 负向指标 | 仅 `rps_doc_frac_no_alph_words`、`rps_doc_frac_chars_top_2gram`、`rps_doc_frac_chars_top_3gram` | 三者语义明确是噪声/冗余；其余 19 项按正向 |
| qurater 归一化范围 | 逐子维度在全数据集（A1）上 MinMax，再取平均 | 保留子维度间真实差异；行内归一化会强制每条样本拉满 [0,1] 并丢失绝对水平 |
| 非单调指标 | `rps_doc_word_count` / `rps_doc_num_sentences` / `rps_doc_mean_word_length` 用梯形隶属度，平台区 [P25, P75]，[P05, P95] 外记 0 | 过短/过长文档都不适合做训练语料，机械单调会误判 |
| 方向统一总体 | A1 为参考集 | 保证跨集可比 |

> 平台区取 [P25, P75] 使 A1 约 50% 的记录在这三个指标上得满分。若需收紧，改
> `quality_preprocess.py` 顶部的 `TRAP_Q` 即可，参数会自动重新冻结。

## 已知数据特征（会影响后续建模）

- **源数据缺失是 JSON `NaN` 字面量**，不是 `None`；判定必须用 `np.isfinite`。A1 缺 18 条（108 格），A3 缺 1 条。
- `rps_doc_frac_*` 系列名为 `frac` 但取值是 **0–100 百分数**；`chars_top_2gram/3gram` 最大可达 **194/284**，并非严格百分比。
- `dsir_*` 原始分无界且随文档长度漂移（A1 中位数 -2102，A2 达 -35136），必须做对数变换后再比。
- `fluency_en` / `ad_en` 的 index 1 才是高质量类；`ad_en` 官方顺序为 `[has_ad, no_ad]`。
- A2/A3 无 `content`，原文核验只能用 A1（及可选的 A18）。
- A1 域分布严重不均：book 仅 171 条，其域级估计不稳定，需在报告中标注。
