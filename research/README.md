# 三线比较与论文主线：研究环境（research/）

本轮只完成**环境准备与轻量验收**：不开展大规模分析、不批量重跑、不提前确定论文主线。
上一版三线分析报告（`baseline-a: reports/三线分析与论文主线_20260923/`）是**历史问题清单**；
其 B/C 第三问数值为修复前快照，结论一律以修复后的提交与新运行证据为准。

## 1. 启动

```bash
cd /home/sshuser/projects/数学建模26/F题-研究整合/research

# 输入与环境检查（三线提交号、关键输入哈希、依赖版本）
.venv/bin/python -m src.entry check

# 轻量验收（E0-A）：14 项接口/数值检查
.venv/bin/python -m src.entry experiment --name acceptance

# 单独“运行某条线/某一问”：默认 dry-run（只登记命令，不执行）
.venv/bin/python -m src.entry line --name C --question q3
# 真正执行（会用上游工作树里的原命令，结果写回上游目录）
.venv/bin/python -m src.entry line --name C --question q3 --execute

# 登记一个已声明实验（本轮 R1 仅登记计划，不执行）
.venv/bin/python -m src.entry experiment --config configs/r1_three_line.yaml

# 汇总历次运行（同口径列表）
.venv/bin/python -m src.entry aggregate
```

研究虚拟环境：`.venv`（`--system-site-packages`，复用系统包）。
实测版本见 `requirements.txt` 与每次运行的 `runs/<run_id>/meta.json`。

## 2. 目录

```
research/
  README.md           本文件
  upstream.lock.json  三线固定提交号、worktree 路径、关键输入 sha256、缺失实体清单
  protocol.yaml       单位、质量坐标、评价目标、切分、来源标签（研究阶段冻结口径）
  experiments.csv     实验登记（planned / pending / done_with_blockers；与执行记录分开）
  decisions.md        主线候选、修复状态三分类、阻塞与预先声明的选择判据
  requirements.txt    实测依赖
  src/
    common.py         Run 记录器（run_id、上游提交、代码版本+未提交 diff、输入哈希、日志、失败）
    adapters.py       A/B/C 适配器：调用各自原实现，声明物理单位边界
    checks.py         六项轻量验收
    entry.py          check / experiment / line / aggregate 入口
  configs/            每项实验配置（r1_three_line.yaml 已声明，run:false）
  runs/               每次运行独立目录（不覆盖）
  reports/            验收/比较报告与图表
```

## 3. 三线固定版本（`upstream.lock.json`）

| 线 | 分支 | 提交 | 参考工作树 |
|---|---|---|---|
| A | `baseline-a` | `c8b97cf4` | `../../F题-upstream/line-a` |
| B | `baseline-b` | `e498b2c6` | `../../F题-upstream/line-b` |
| C | `baseline-c` | `cf765761` | `../../F题-upstream/line-c` |

C 的分布（以实测提交为准，不按分支名推测）：
- Q1-C、Q2-C 来自 `baseline-q2-c`（`6f101de1`）；
- Q3-C 修复版来自 `baseline-q3-c`（`7e7e5908`）；
- 预处理来自 `baseline-q1-c-fix`（`efc08cb`，**路径复制**进 `baseline-c`，非 ancestry）；
- 合体版 = `baseline-c`（`cf765761`）。ancestry 结果见 `upstream.lock.json.component_in_baseline_c_by_ancestry`。

修复状态三分类（详见 `decisions.md`）：**代码已修** / **轻量验证通过** / **完整结果已重跑**。
只有三者齐备的结果才作为修复后证据；旧分支 Q3 数值不得自动视为已修复。

## 4. 数据与指针

- 原始附件实体在仓库根工作区：`/home/sshuser/projects/数学建模26/F题/real_attachments`（**只读**）。
- 本目录与 `line-b` 中的 `*.jsonl.xz` 是 **LFS 指针（134B）**，不可当压缩数据读取。
- 缺失实体：`line-b/data/quality_rebuild/*.jsonl.xz`（LFS 指针）——复核 B 质量重建前需 `git lfs pull`。
- 适配器通过文件接口读取上游产物，不复制大数据；不在上游工作树写入。

## 5. 运行记录约定

每次 `Run` 生成不可覆盖的 `runs/<run_id>/`，含：
`meta.json`（上游提交、研究代码 commit + 未提交 diff 的 sha256、配置、种子、输入 sha256、命令、耗时、状态）、
`metrics.json`、`log.txt`（`line --execute` 另存 `stdout.txt`/`stderr.txt`）。
失败运行记 `status="failed"` 并保留 traceback，绝不写成成功；计划与结果分离（`experiments.csv`）。

## 6. 轻量验收（E0-A）结果

run_id `20260923T165441Z_acceptance_2974389`（首轮）→ 最新 `20260923T171318Z_acceptance_2983929`，报告见 `reports/acceptance_*.md`：
**13 pass / 0 fail / 0 blocked / 1 n/a**（C Q1 预测器可执行后由 blocked 转为 pass）。

- 通过：A/B/C 在同一物理点（N=1e9,D=1e11,Q=1,h=1）第二问与第三问损失预测一致（均 2.385578）；
  C 成本按实际个数回算残差 3.1e-15；B 成本残差 2.2e-16；C 在 Q=Q0 的右导数与前向差分一致；
  C 主最优表 720 行全部 `in_trust_l1=True`；C Q1 描述符协议完整；跨问尺度冻结（Q2-C 标定存在）；
  计划/结果分离检查通过；修复证据齐备。
- **已解决**：C 的 Q1 配比预测器已重建为可执行对象（`artifacts/c_q1_predictor/`），对拍 `equivalent=true`（n=9828，max abs diff 1.78e-15）；验收 14 项 **13 pass / 0 fail / 0 blocked / 1 n/a**。
- 仍阻塞：B 的质量重建数据为 LFS 指针，未 pull；B 的质量评分标记为不同样本/复现未完成。

## 6b. 成稿交付（F0–F5）

`reports/F0_主张证据核对.md`、`F1_论文主线_一页说明.md`、`F2_最终章节大纲.md`、`F3_核心图表清单.md`、`F4_摘要草稿.md`、`F5_必须补做最小清单.md`。

## 7. 下一阶段（不本轮执行）

- **R1**：修复后三线统一口径比较（入口：`experiment --config configs/r1_three_line.yaml`；需先实现比较器）。
- **R2**：依据 R1 提出**至多两个**主线候选，各安排一个最小辨别实验。
- **R3**：必要时实现新方法/消融，形成主线选择记录与更新后的完整大纲。

约束：先区分实现错误与模型局限；端到端比较与固定上游消融分开；已用于挑方法的检验集不再称盲测；
半合成/插值/估算单独标记；允许无增益、无结构转移、不可辨识的结论；方法可新增/替换/舍弃。
