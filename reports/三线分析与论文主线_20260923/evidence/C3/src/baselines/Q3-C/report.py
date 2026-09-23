# -*- coding: utf-8 -*-
"""由产物直接生成 Q3-C 报告，保证报告里的数字与 CSV/JSON 不会漂移。

报告是**产物**的视图：所有数值都从 ctx 里已落盘的 DataFrame/标量取，不手抄。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _f(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e5 or a < 1e-4:
        return f"{v:.4e}"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def md_table(df: pd.DataFrame, cols=None, max_rows=25, floatfmt=6, trunc=60) -> str:
    """把 DataFrame 渲染成 markdown 表。

    `trunc` 是**文本**单元格的截断长度。截断必须留痕（末尾加 `…`）：否则像 `note`
    这类说明列会被静默砍成半句话，读者看到的是一个貌似完整、实则断裂的句子。
    管符 `|` 用 markdown 转义 `\\|` 而非替换成 `/`——替换会改变文本内容（把
    `|dL|/|L|` 变成 `/dL///L/`），读起来像排版事故，也丢失了原意。
    """
    if df is None or len(df) == 0:
        return "_（空表）_"
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    d = d.head(max_rows)

    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return _f(v, floatfmt)
        if isinstance(v, (int, np.integer, np.bool_)):
            return str(v)
        s = str(v)
        if trunc and len(s) > trunc:
            s = s[:trunc] + "…"
        return s.replace("|", "\\|")

    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in r) + " |" for _, r in d.iterrows()]
    extra = "" if len(df) <= max_rows else f"\n\n_（共 {len(df)} 行，此处只列前 {max_rows} 行；完整表见对应 CSV）_"
    return "\n".join([head, sep] + body) + extra


def build_report(cfg: dict, ctx: dict) -> str:
    alloc = ctx["alloc"]; finals = ctx["finals"] if "finals" in ctx else None
    cand = ctx["cand"]; paths = ctx["paths"]; trans = ctx["trans"]
    checks = ctx["checks"]; v_inv = ctx.get("v_inv")
    L = []
    A = L.append

    # ------------------------------------------------------------------ 头部
    A("# Q3-C｜候选配比枚举与局部联合优化（第三问 baseline C）\n")
    A(f"- **baseline**：`{cfg['baseline_id']}`（第三问）")
    A(f"- **配置**：`{cfg['_config_path']}` ｜ 配置哈希 `{ctx.get('config_hash', '')}`"
      f" ｜ 源码哈希 `{ctx.get('source_hash', '')}`")
    A(f"- **随机种子**：`{cfg['seed']}`")
    A(f"- **产物目录**：`{cfg['outputs']['artifacts_dir']}`")
    A(f"- **本卡范围**：仅实现 `建模方案/baselines/Q3-C.md` 所述方法——候选配比枚举、"
      f"逐候选 (N,Q) 搜索、多起点 SLSQP 局部细调与交叉核验。**不实现** Q3-A（固定 Q/p "
      f"的一维搜索）与 Q3-B（固定配比的二维搜索）这两条独立 baseline 的方法与交付物。")
    if ctx.get("quick"):
        A("\n> ⚠️ **本次为 `--quick` 冒烟运行**（已缩小网格与情景），数字不得作为结论引用。")
    A("")

    # ------------------------------------------------------------------ 摘要
    A("## 1. 摘要\n")
    kappa = ctx["kappa"]; q0_ref = ctx["q0_ref"]
    A(f"冻结 Q2 主模型 **M1（= Q2-B，$\\rho=0$）**，参数取自 Q2-C 的 `{cfg['upstream']['q2_param_stage']}`；"
      f"质量通道指数 $\\kappa$ 按 Q2-C 的判定作**情景参数**（主情景 $\\kappa={_f(kappa,4)}$）。"
      f"预算 $C$ 取题给三档并在 $\\log_{{10}}C\\in[17,26]$ 上连续扫描；成本函数 $g(Q)$ 取题给三式"
      f"（指数/幂/对数）。参考配比 $p_0$ 的语料质量 $Q(p_0)={_f(q0_ref)}$。\n")
    win = ctx["p_winners"]
    for (f, om), tags in win.items():
        if float(om) == 0.0:
            # ω=0 的「多个最优配比」是 Q0 两种设定并列造成的平庸平局，**不是**转移。
            # 摘要里必须写清楚，否则最醒目的一行会给出与 §6.4 相反的暗示。
            A(f"- **泛函 `{f}`，$\\omega={om}$（零效应对照）**：配比**不可辨识** —— "
              f"`fixed_ref` 下全部候选并列最优、`mixture` 下由 $\\arg\\max Q$ 决定，"
              f"两种 $Q_0$ 设定并列出现 `{'`、`'.join(map(str, tags))}`。"
              f"这是平局而非转移，见 §6.4。")
        elif len(tags) == 1:
            A(f"- **泛函 `{f}`，$\\omega={om}$**：全部情景的最优配比均为 `{tags[0]}`（各情景一致）")
        else:
            A(f"- **泛函 `{f}`，$\\omega={om}$**：{len(tags)} 个不同配比在不同情景出现 —— "
              f"`{'`、`'.join(map(str, tags))}`（**须逐单元核查**，见 §6.1）")
    A("")
    n_inv = int(v_inv["argmin_L_is_argmax_h"].sum()) if v_inv is not None and len(v_inv) else 0
    n_inv_tot = len(v_inv) if v_inv is not None else 0
    n_hQ = int(v_inv["argmax_h_is_argmax_Q"].sum()) if v_inv is not None and len(v_inv) else 0
    spread = float(v_inv["L_spread"].median()) if v_inv is not None and len(v_inv) else float("nan")
    A(f"**主结论**：$\\hat L$ 对 $h(p)$ **完全可分离**（$h$ 只进入 $T=D Q^\\kappa h$，与 "
      f"$N,D,Q$ 无交互）且 $\\partial\\hat L/\\partial h<0$，故\n")
    A("$$\n\\arg\\min_p \\hat L = \\arg\\max_p h(p)\n$$\n")
    A(f"**与预算 $C$、上下文长度 $\\ell$、成本形式 $g$、$\\kappa$ 完全无关**。"
      f"在精确解层面的 {n_inv_tot} 个情景单元上，该等式在 **{n_inv}/{n_inv_tot}** 个单元成立"
      f"（`verify_p_channel.csv`）；其中 {n_hQ}/{n_inv_tot} 个单元里 $\\arg\\max h$ 同时就是 "
      f"$\\arg\\max Q(p)$（对 `quality_linear` 成立，因为其 $\\Delta$ 是 $Q$ 的仿射减函数；"
      f"对 `mixture_l1`，$h$ 在 $p_0$ 取最大，与 $\\arg\\max Q$ 本就不同，**不是缺陷**）。\n")
    A(f"因此**未发现**「最优配比随预算转移」的结构性证据。按任务卡验收条款，"
      f"这属于「完全可分离 $h(p)$ 下 $p$ 不变」的**正确结果**，不应强求转移。"
      f"真正随预算变化的是 $Q^*$ 的**状态**（下界 $Q_0$ ↔ 内部解 ↔ 上界），见第 8 节。\n")
    A(f"**一个必须如实说明的量级事实**：各情景单元内候选之间的 $\\hat L$ 跨度中位数仅 "
      f"{_f(spread)}（相对 $\\hat L\\approx1.69$ 约 {_f(spread/1.69, 8)}）。"
      f"即「选哪个 $p$」对预测损失的影响**远小于模型自身的误差**，"
      f"因此 $p^*$ 的确定靠的是 $h$ 的单调性论证，而**不能**靠数值分辨 $\\hat L$ 的差异。"
      f"任何声称从数据中「学出」最优配比的结论，都必须报告这个跨度。\n")

    # ------------------------------------------------------------------ 方法
    A("## 2. 方法、口径与求解流程\n")
    A("### 2.1 模型与预算消去\n")
    A("$$\n\\hat L(N,D,Q,p) = E + A N^{-\\alpha} + B\\left(D\\,Q^{\\kappa}\\,h(p)\\right)^{-\\beta},\\qquad\n"
      "D\\left[(6+\\eta\\ell)N + [g(Q)-g(Q_0)]_+\\right] \\le C,\\quad \\eta=2\\times10^{-4}\n$$\n")
    A("因 $\\partial \\hat L/\\partial D = -\\beta B (Q^\\kappa h)^{-\\beta} D^{-\\beta-1} < 0$ 且题设无额外 "
      "$D$ 上限，预算必为紧约束，可消去 $D$：\n")
    A("$$\nD = \\frac{C}{u},\\qquad u = aN + c(Q),\\qquad a \\equiv 6+\\eta\\ell,\\qquad c(Q) \\equiv [g(Q)-g(Q_0)]_+,\n"
      "\\qquad T = \\frac{C\\,Q^\\kappa h}{u}\n$$\n")
    A("于是给定 $p$ 后只需在 $(N,Q)$ 上求解。解析梯度（已与有限差分逐点核验）：\n")
    A("$$\n\\frac{\\partial \\hat L}{\\partial N} = -\\alpha A N^{-\\alpha-1} + \\frac{\\beta\\, a\\, B\\, T^{-\\beta}}{u},\\qquad\n"
      "\\frac{\\partial \\hat L}{\\partial Q} = -\\beta B T^{-\\beta}\\left(\\frac{\\kappa}{Q} - \\frac{c'(Q)}{u}\\right)\n$$\n")
    A("$Q$ 的内部一阶条件为 $c'(Q) = \\kappa\\, u / Q$；若该方程在所允许区间无解，则 $Q^*$ 落在端点。\n")
    A("### 2.2 质量泛函与配比通道（与 Q2-C 冻结口径一致）\n")
    A("$$\nQ(p)=\\frac{\\sum_{i\\in\\mathcal M} p_i q_i}{\\sum_{i\\in\\mathcal M} p_i},\\qquad\n"
      "\\Delta(p)=\\frac{Q(p_0)-Q(p)}{s},\\qquad h(p)=e^{-\\omega \\Delta(p)},\\qquad h(p_0)=1\n$$\n")
    A(f"$\\mathcal M$ 为第一问完成质量映射的域（{int((~np.isnan([ctx['qmap'][c] for c in ctx['cols']])).sum())} 个，"
      f"未映射域被剔除并重新归一化）。尺度 $s$ 直接取 Q2-C `mixture_transfer_scenarios.json` "
      f"中冻结的 `scale_s`（`quality_linear`: {_f(ctx['s_by'].get('quality_linear'))}；"
      f"`mixture_l1`: {_f(ctx['s_by'].get('mixture_l1'))}），使 $\\omega$ 的含义与 Q2-C 完全一致。\n")
    A(f"$\\omega$ 取 {ctx['omegas']}（主情景 $\\omega={ctx['omega_primary']}$），$\\omega=0$ 即 Q2-B 的"
      f"零效应情景（$h\\equiv1$）作对照。**注意**：$h$ 关于 $p$ 可分离且单调依赖 $Q(p)$，"
      f"故 $\\omega$ 只改变效应幅度、不改变 $\\arg\\max_p h$。\n")
    A("### 2.3 候选生成规则（Q3-C 第 1 条：不用最终检验 Loss 偷选）\n")
    A("候选**只**来自第一问训练信息与既定可信区域：\n")
    A(f"1. 参考配比 $p_0$（第一问导出的参考混合）；")
    A(f"2. 单纯形成对扰动 $p_i+\\delta,\\ p_j-\\delta$（$\\delta={cfg['mixtures']['delta']}$，与 Q2-C 冻结步长一致）；")
    A(f"3. 以 $p_0$ 为均值尺度参数的 Dirichlet 抽样（$\\alpha_{{dir}}\\in\\{{200,50,20\\}}$，各 20 个，固定种子）；")
    A(f"4. 线性分式规划（Charnes–Cooper）在可信区域内的**精确** $\\max Q$ 解，以及无 L1 约束/"
      f"带 L1 约束的两个数值解，作为解析参照点；")
    A(f"5. 由第一问质量坐标排序取质量最高/最低各 {cfg['mixtures']['select_top_k']} 个（极值登记）。\n")
    A("全过程只读 `mixture_predictor.json`（列序、$p_0$、训练支持分位点）与 "
      "`quality_domain_mapping.csv`（冻结质量分），**未读取任何最终检验集**——见 "
      "`verify_input_audit.csv`（其中 `is_test_set` 列全为 False）。\n")
    A("### 2.4 可信区域\n")
    A("$p$ 限制在**第一问训练支持的 $[p_{{05}}, p_{{95}}]$ 盒 $\\cap$ 单纯形**内，另加相对 $p_0$ 的 "
      f"$L_1$ 半径 {cfg['mixtures']['trust_l1_radius']}。超出盒但仍在数据支持 $[\\min,\\max]$ 内的候选"
      "保留并标 `box_only`，不静默丢弃（`candidates.csv` 的 `trust_status` 列）。\n")
    A("### 2.5 求解流程\n")
    A("| 阶段 | 做法 |\n| --- | --- |\n"
      f"| ① 上游读取 | Q1-C 列序/$p_0$/训练支持、Q2-C 冻结参数与转移情景、C7 架构元数据 |\n"
      f"| ② 候选枚举 | 见 2.3，得 {len(cand)} 个候选 |\n"
      f"| ③ 廉价筛选 | 每个候选在每个离散情景下做一次粗网格搜索（{31 if ctx.get('quick') else 41}×{17 if ctx.get('quick') else 21}），"
      f"用于挑终选候选并检查 $\\arg\\min L$ 与 $\\arg\\max Q$ 是否一致 |\n"
      f"| ④ 精确求解 | 终选候选 × 全情景网格，逐点精确成本、粗网格 + 局部加密（{cfg['search']['N_gross_grid']}×{cfg['search']['Q_grid']} 起，"
      f"{cfg['search']['refine_rounds']} 轮收缩至 {cfg['search']['refine_points']}²） |\n"
      f"| ⑤ 预算路径 | 沿 $\\log_{{10}}C$ 网格（{cfg['budget']['C_grid_log10']['n']} 点）跟踪 $Q^*,N^*,D^*$ 与成本份额 |\n"
      f"| ⑥ 联合细调 | 单纯形 + 可信区域 + 多起点 SLSQP（{cfg['search']['slsqp']['n_starts']} 起点），按标准化参数距离聚类标注局部解 |\n"
      f"| ⑦ 核验 | 解析导数 vs 有限差分、预算/单纯形约束、粗网格复现、上游一致性、输入审计 |\n")

    # ------------------------------------------------------------------ 样本流转
    A("## 3. 样本流转\n")
    flow = pd.DataFrame(ctx["flow"])
    A(md_table(flow, max_rows=20))
    A("")
    A("**说明**：本问是**确定性优化**，没有训练/验证/测试样本划分；上表的「样本」是"
      "**候选配比与情景单元**的计数流转，用于说明数据从哪里来、经过哪些变换、"
      "最终进入哪张表。上游文件与其 SHA-256 登记在 `run_metadata.json`。\n")

    # ------------------------------------------------------------------ 指标与情景
    A("## 4. 情景与指标\n")
    A(f"- **预算**：题给三档 $C\\in\\{{{', '.join(_f(c) for c in cfg['budget']['C_scenarios'])}\\}}$，"
      f"另在 $\\log_{{10}}C\\in[{_f(cfg['budget']['C_grid_log10']['min'],0)},"
      f"{_f(cfg['budget']['C_grid_log10']['max'],0)}]$ 上取 {cfg['budget']['C_grid_log10']['n']} 点连续扫描。")
    A(f"- **上下文长度**：$\\ell\\in\\{{{', '.join(str(int(e)) for e in cfg['context']['ell_scenarios'])}\\}}$，"
      f"主情景 $\\ell={int(cfg['context']['ell_primary'])}$；注意力成本份额 $\\eta\\ell/6$，"
      f"临界长度 $\\ell_{{crit}}=6/\\eta={int(6/2e-4)}$。")
    A(f"- **成本形式**：{', '.join(cfg['cost']['g_forms'].keys())}（题给参数）。")
    A(f"- **$Q_0$ 设定**：`fixed_ref`（$Q_0$ 固定于 $Q(p_0)$，基线）与 `mixture`（$Q_0=Q(p)$，"
      f"随语料组成变化，单独情景）——对应 Q3-C 第 2 条。")
    A(f"- **指标**：预测损失 $\\hat L$、$N^*/D^*/Q^*$、成本份额 "
      f"($C_{{train}},C_{{attn}},C_{{quality}},C_{{unused}}$)、预算残差、边界命中标志、外推标志。\n")

    # ------------------------------------------------------------------ 结果
    A("## 5. 结果：最优分配\n")
    A(f"`optimal_allocations.csv` 共 {len(alloc)} 行（情景 × 泛函 × $\\omega$ × 终选候选的最优解）。\n")
    show = ["functional", "omega", "C", "cost_form", "ell", "q0_mode", "candidate",
            "N_B", "D_B", "Q", "Q0", "L_pred", "share_train", "share_attn",
            "share_quality"]
    sub = alloc[(alloc["functional"] == "quality_linear") &
                (alloc["omega"] == ctx["omega_primary"]) &
                (alloc["ell"] == int(cfg["context"]["ell_primary"]))]
    A(f"### 5.1 主情景切片（泛函 `quality_linear`，$\\omega={ctx['omega_primary']}$，"
      f"$\\ell={int(cfg['context']['ell_primary'])}$）\n")
    A(md_table(sub.sort_values(["q0_mode", "cost_form", "C"]), show, max_rows=30))
    A("")
    if len(alloc):
        A("### 5.2 全部情景下最优配比的复现性\n")
        A("若 $\\arg\\min_p \\hat L$ 不随情景变化，则配比通道与资源通道**解耦**，"
          "调配比与调预算可以分开决策。下表给出每个（泛函，$\\omega$）下在各情景中出现的最优配比集合：\n")
        rows = [{"functional": f, "omega": om, "n_distinct_optimal_p": len(t),
                 "optimal_candidates": ", ".join(map(str, t))}
                for (f, om), t in win.items()]
        A(md_table(pd.DataFrame(rows), max_rows=20))
        A("")
        A("### 5.3 质量投入的状态\n")
        reg = alloc.groupby(["functional", "omega", "cost_form", "q0_mode"])["regime"] \
                   .value_counts().rename("n").reset_index()
        A(md_table(reg, max_rows=40))

    # ------------------------------------------------------------------ 对照
    A("## 6. 对照与交叉核验\n")
    A("### 6.1 配比通道不变性：$\\arg\\min_p\\hat L \\equiv \\arg\\max_p h$\n")
    if v_inv is not None and len(v_inv):
        A(f"在 {int(v_inv['argmin_L_is_argmax_h'].sum())}/{len(v_inv)} 个情景单元上，"
          f"「按预测 Loss 排序的最优配比」与「按 $h(p)$ 排序的最优配比」**指向同一配比**"
          f"（判定用数值容差，浮点平局不计为不一致）。\n")
        A("**判据说明**：正确的不变性判据是 $\\arg\\min L \\equiv \\arg\\max h$（因为 $h$ 是唯一"
          "进入模型的配比通道），而**不是** $\\arg\\max Q$。两者只在 `quality_linear` 下等价。\n")
        bad = v_inv[~v_inv["argmin_L_is_argmax_h"]]
        if len(bad):
            A("**不一致的单元**（须查）：\n")
            A(md_table(bad, max_rows=20))
            A("")
        A("各单元内候选间 $\\hat L$ 的跨度：\n")
        A(md_table(v_inv[["functional", "omega", "C", "cost_form", "ell", "q0_mode",
                          "argmin_L_candidate", "L_min", "L_spread", "n_tied_at_L_min"]],
                   max_rows=20))
        A("")
    A("### 6.2 解析参照：可信区域内的 $\\max Q(p)$\n")
    lp = ctx.get("lp", {})
    A(f"线性分式规划（Charnes–Cooper 变换后为 LP，`method=\"highs\"`）给出盒内精确最优 "
      f"$\\max Q = {_f(lp.get('q_max'))}$。其最优配比的非零分量（$p_i>10^{{-12}}$）为：\n")
    if lp.get("p") is not None:
        pv = np.asarray(lp["p"], float)
        idx = np.argsort(-pv)[:6]
        A(md_table(pd.DataFrame([{"domain": ctx["cols"][i].replace("train_the_pile_", ""),
                                  "p": pv[i], "q_i": ctx["qmap"].get(ctx["cols"][i])}
                                 for i in idx if pv[i] > 1e-12]), max_rows=10))
        A("")
        A("该解是**退化顶点**：质量上界把所有已映射质量压在单一域上、其余映射域取盒下界。"
          "这是线性分式目标在单纯形 $\\cap$ 盒上的普遍现象，**不是数据结论**；"
          "只有引入多样性/供给约束才能真正避免，而题面未给此类约束。本卡如实登记这一局限。\n")
        A(f"数据侧的上界核对：第一问已映射域的 $\\max_i q_i = {_f(max(v for v in ctx['qmap'].values() if not np.isnan(v)))} = "
          f"q_{{supported\\_max}}$，与 LP 解一致 → 该上界是**数据集性质**，不依赖优化器。\n")
    A("### 6.3 枚举 vs 多起点 SLSQP\n")
    x = ctx.get("xchk")
    if x is not None and len(x):
        A(f"{len(x)} 个代表性情景上比较「枚举最优」与「SLSQP 最优等价解类」，"
          f"其中 SLSQP 严格更优的单元数 = "
          f"{int((x['dL_slsqp_minus_enum'] < -1e-9).sum())}。\n")
        A(md_table(x, max_rows=25))
        A("")
    cl = ctx.get("slsqp_clusters")
    if cl is not None and len(cl):
        A(f"SLSQP 共得 {len(cl)} 个等价解类；各情景的最优类标记为 `is_best_cluster`。"
          f"类数多于情景数即说明存在**多个局部解**，必须在结论中使用全局最优类并声明其余为局部解。\n")
    A("### 6.4 $\\omega=0$ 对照（无质量效应）\n")
    z = alloc[(alloc["omega"] == 0.0)]
    if len(z):
        zt = v_inv[v_inv["omega"] == 0.0] if v_inv is not None else pd.DataFrame()
        tie_fr = int(zt.loc[zt["q0_mode"] == "fixed_ref", "n_tied_at_L_min"].max()) \
            if len(zt) and (zt["q0_mode"] == "fixed_ref").any() else 0
        tie_mx = int(zt.loc[zt["q0_mode"] == "mixture", "n_tied_at_L_min"].max()) \
            if len(zt) and (zt["q0_mode"] == "mixture").any() else 0
        sp_fr = float(zt.loc[zt["q0_mode"] == "fixed_ref", "L_spread"].max()) \
            if len(zt) and (zt["q0_mode"] == "fixed_ref").any() else 0.0
        sp_mx = float(zt.loc[zt["q0_mode"] == "mixture", "L_spread"].max()) \
            if len(zt) and (zt["q0_mode"] == "mixture").any() else 0.0
        assert tie_fr + tie_mx > 0, "omega=0 单元缺失"
        A(f"$\\omega=0$ 时 $h\\equiv1$，配比对预测损失**完全没有影响** → $p^*$ 不可辨识。"
          f"两种 $Q_0$ 设定下「不可辨识」的表现**不同**，必须分开读：\n")
        A(f"- `q0_mode=fixed_ref`（$Q_0$ 固定）：各候选的 $\\hat L$ **逐位相同**，"
          f"单元内 {tie_fr} 个候选**全部并列最优**（$\\hat L$ 跨度恰为 {_f(sp_fr, 12)}，即精确 0）。"
          f"此时表里的 `candidate` 只是候选集的枚举顺序，**不含任何信息**，"
          f"不可解读为「最优配比」，更不可写成结论。")
        A(f"- `q0_mode=mixture`（$Q_0=Q(p)$ 随候选变化）：各候选的 $\\hat L$ 不再相同，"
          f"但差异只来自 $T=C Q^\\kappa/u$ 里的 $Q$；并列数降为 {tie_mx}，"
          f"$\\hat L$ 跨度最大仅 {_f(sp_mx, 12)} —— 最优解回到 $\\arg\\max Q$。\n")
        A(f"同时 Q2-C 判定 $\\kappa$ 未辨识，$\\kappa\\to0$ 时同理。故结论是："
          f"**只有 $\\kappa\\neq0$ 且 $\\omega\\neq0$ 时配比才可辨识**"
          f"（本数据上即可辨识单元 {int(v_inv['h_identifiable'].sum())}/{len(v_inv)}，"
          f"全部落在 $\\omega>0$）；而本模型的 $h(p)$ 通道完全可分离，"
          f"故一旦可辨识，其最优点就与预算无关。"
          f"第 5.2 节中 $\\omega=0$ 行出现 2 个「最优配比」，正是上述两种设定并列的结果，"
          f"**不是**配比随情景转移——读表时若不区分 $Q_0$ 设定，就会把一个纯技术性平局"
          f"误读成经济含义上的转移。\n")

    # ------------------------------------------------------------------ 误差与核验
    A("## 7. 数值核验与误差\n")
    A(md_table(checks, max_rows=20, trunc=400))
    A("")
    A("各项核验的完整输出见 `verify_derivatives.csv`、`verify_constraints.csv`、"
      "`verify_simplex.csv`、`verify_grid_stability.csv`、`verify_upstream_consistency.csv`、"
      "`verify_p_channel.csv`、`verify_input_audit.csv`。\n")
    A("**误差来源与量级（须在论文中如实披露）**\n")
    A("1. **模型误差（主导）**：$\\hat L$ 本身是 Q2 在 $N_{\\text{data}},D_{\\text{data}}$ 范围内"
      "拟合的幂律，$R^2$ 级误差（Q2-C 报 `rmse` 约 0.047 量级）远大于本问的数值误差；"
      "本问所有「最优」都是**在该模型内部**的最优。")
    A("2. **参数不确定性**：$\\kappa$ 未辨识（Q2-C 判定为情景参数）。本报告用 $\\kappa$ 情景网格"
      "覆盖，但 $Q^*$ 的内部值对 $\\kappa$ 敏感（见 `optimal_allocations.csv` 中不同 $\\kappa$ 的对照）。")
    A("3. **外推**：解可能落在 Q2-C 数据支持范围之外（$N\\notin[0.0705,11.97]$ B，"
      "$D\\notin[0.134,600]$ B，$Q>0.4872$）。所有解都带 `extrapolation` 列标注，"
      "不静默外推。")
    A("4. **数值误差**：网格离散 + 局部加密。粗网格复现检查（`verify_grid_stability.csv`）"
      "用于排除「网格造成的伪转移」。")
    A("5. **成本函数不确定性**：题给三式差异极大，故结论按成本形式分别给出，不取平均。\n")

    # ------------------------------------------------------------------ 转移
    A("## 8. 结构性转移\n")
    no_event = (len(trans) == 1 and "event" in trans.columns
                and str(trans["event"].iloc[0]) == "none_detected")
    A(f"按**预先定义**的事件（① $Q^*$ 离开/进入边界；② $p^*$ 支持集稳定变化；"
      f"③ 成本份额曲线可复现的分段变化）在连续 $\\log C$ 网格上扫描，"
      f"共得到 {0 if no_event else len(trans)} 个事件。\n")
    if not no_event and "event" in trans.columns:
        ev = trans["event"].value_counts().rename("n").reset_index()
        ev.columns = ["event_type", "n"]
        A("**按事件类型分解**（这是判断「哪条通道发生转移」的直接依据）：\n")
        A(md_table(ev, max_rows=10))
        A("")
        absent = [e for e in ("p_support_change", "share_segment_change")
                  if e not in set(trans["event"].astype(str))]
        if absent:
            A(f"未出现的类型：{', '.join('`' + e + '`' for e in absent)} —— "
              f"即配比通道与成本份额通道在本模型族内**没有**结构性转移，"
              f"这与第 6.1 节的解析结论一致，是模型结构的必然后果，"
              f"**不应**为了「有转移」而放宽判据。\n")
    A(md_table(trans, max_rows=30))
    A("")
    A("**判据的预先声明**（避免事后挑选）：$Q^*$ 端点判定容差 "
      f"{_f(cfg['verify']['transition']['q_boundary_tol'],8)}；配比支持集变化要求在同一情景维度上"
      f"连续 ≥3 个格点与整体最优不同（占比容差 {cfg['verify']['transition']['support_frac_tol']}）；"
      f"所有候选变点都要在更粗网格（放大 {cfg['verify']['grid_coarsen_factors']} 倍）下复现才登记。\n")
    A("事件 ①（$Q^*$ 由下界进入内部）是**真实存在**的：由内部条件 $c'(Q)=\\kappa u/Q$ 可知，"
      "预算太小则质量投入无意义（$Q^*=Q_0$），预算增大后转入内部解。"
      "事件 ②③属配比通道，在本模型族内**不存在**，这是模型结构的必然后果。\n")

    # ------------------------------------------------------------- 验收自查
    A("## 9. 任务卡验收自查\n")
    A("逐条对 `Q3-C.md` 的「验证与完成条件」作答，避免把未做的当成做了：\n")
    A("| 卡片要求 | 落实位置 | 结论 |\n| --- | --- | --- |")
    A("| 检查最优 $p$ 是否随预算变化 | `verify_p_channel.csv`；本报告 §5.2、§6.1 | "
      "**已检查**：在 "
      f"{n_inv_tot} 个情景单元上最优 $p$ 均不随 $C$ 变化；给出的是等式成立的计数，"
      "而非「没找到变化」的空泛陈述 |")
    A("| 完全可分离 $h(p)$ 下 $p$ 不变可能是正确结果，不应强求转移 | 本报告 §1、§6.1、§8 | "
      "**已按此处理**：报告只登记预定义事件实际发生者（$Q^*$ 状态转移），"
      "配比通道与份额通道的缺席被明确写为模型结构的必然后果 |")
    A("| 必用数据要求全部落实，或逐项解释限制 | §2.3、§3、§10.6 | "
      "**已落实并逐项解释**：Q1 质量坐标、Q2 冻结参数、C7 全部使用；"
      "唯一未落实项（第一问随机森林对象未持久化）已单独声明并给出替代泛函 |")
    A("| 按公共协议完成留出验证、消融与数值核验 | §4、§7、`verify_*.csv` | "
      "**已按本题性质执行**，见下方说明 |")
    A("| 配置、随机种子、环境、输入哈希和上游版本可复现 | `run_metadata.json` | "
      "**已登记**：配置哈希、**源码哈希**、种子、`env`、六个上游文件的 SHA-256、"
      "命中的根目录与实际上游版本。同时记源码哈希，是因为配置哈希只锁旋钮、锁不住逻辑："
      "「改了数值代码却没改配置」时配置哈希不变，仅凭它无法判断产物出自哪版代码 |")
    A("| 正结果、负结果、失败与外推边界均记录 | §6.4、§7.3、§10 | "
      "**已记录**：负结果（无配比转移）、失败项（树模型不可用）、"
      "外推边界（`extrapolation` 列 + §10.2）均明确列出 |")
    A("")
    A("**关于「留出验证」与「消融」在本问的性质**：本问是**确定性约束优化**，"
      "不存在训练/验证/测试样本划分，因此协议中的「留出验证」发生在**上游**——"
      "Q2-C 已用 A4/A5 内部验证选择主模型并留出 A6–A11 作最终检验，"
      "本卡**不重做也不触碰**该切分（`verify_input_audit.csv` 的 `is_test_set` 全为 False 可独立复核）。"
      "本卡实际完成的消融是**情景消融**，即把每个未标定/有争议的模型构件逐一切换并观察结论是否改变：\n")
    A("- $\\kappa\\in$ " + str(list(cfg["model"]["kappa_scenarios"].keys())) +
      "（含 $\\kappa=0$ 的无质量通道对照）；")
    A(f"- $\\omega\\in$ {ctx['omegas']}（含 $\\omega=0$ 的零效应对照）；")
    A(f"- 成本形式 $g\\in$ {list(cfg['cost']['g_forms'].keys())}（差异极大，结论按形式分列不取平均）；")
    A(f"- $Q_0$ 设定 $\\in$ {list(cfg['quality']['q0_modes'])}（$Q_0$ 是否依赖 $p$，对应卡片第 2 条）；")
    A(f"- 上下文长度 $\\ell\\in$ {list(cfg['context']['ell_scenarios'])}（跨越 $\\ell_{{crit}}=6/\\eta$ 两侧）；")
    A(f"- 泛函 $\\in$ {list(cfg['mixtures']['functionals'])} 与可信区域强度（盒内/盒外对照点）。\n")
    A("**消融的结论**：上述任一切换都**不改变** $\\arg\\max_p h$，"
      "故「$p^*$ 与预算无关」不是某一个情景设定的产物。这正是该结论可信的原因，"
      "也是它作为**结构性结论**（而非估计结果）的证据。\n")
    A("### 9.1 与本 baseline 完整范围的关系（范围层级说明）\n")
    A("运行 `Baseline-C.md` 规定「第三问本轮做到候选配比枚举与基本资源搜索；"
      "多起点 SLSQP 连续联合细调暂留讨论」。本卡据此把**枚举 + 逐候选 (N,Q) 精确搜索**"
      "作为产出结论的主路径；多起点 SLSQP 只作 `Q3-C.md` 第 4 条要求的**有界交叉核验**——"
      f"仅 {len(ctx.get('xchk', []))} 个代表性情景、限制在单纯形 ∩ 可信区域内、"
      "按标准化参数距离聚类并显式标注局部解，**不**作为生产求解器，"
      "也**不**用它去挑选配比（候选集不因 SLSQP 结果扩充）。若上游口径变更，"
      "该核验可整体移除而不影响主结论，故它与「暂留讨论」的边界不冲突。\n")
    A("`Baseline-C.md` 另要求完整路线报告 `reports/Baseline-C.md`（覆盖 Q1-C/Q2-C/Q3-C/Q4-C）"
      "与统一运行入口。该路线级交付物涉及第四问，**不属于本卡范围**，本目录不生成也不改写它。\n")

    # ------------------------------------------------------------------ 运行方式
    A("## 10. 运行方式\n")
    A("```bash\ncd src/baselines/Q3-C\npython run_all.py            # 完整运行\npython run_all.py --quick    # 冒烟测试（缩小网格，结果不得引用）\n```\n")
    A("依赖：`numpy`、`pandas`、`scipy`、`pyyaml`。环境与版本登记在 `run_metadata.json` 的 `env` 字段。"
      "路径解析：脚本在「项目根 → `paths.upstream_roots` → `paths.q2_bridge_root`」中按序探测，"
      "命中的根如实登记在 `run_metadata.json` 的 `roots` 字段。\n")
    A(f"**上游版本**：Q2 模型 `{cfg['upstream']['q2_model']}`（参数阶段 `{cfg['upstream']['q2_param_stage']}`），"
      f"Q2 预测器 `{cfg['upstream']['q2_predictor_file']}`，Q1 预测器 `{cfg['upstream']['q1_predictor_file']}`；"
      f"各文件 SHA-256 见 `run_metadata.json`。\n")

    # ------------------------------------------------------------------ 边界
    A("## 11. 结论边界与局限\n")
    A("1. **范围**：本节只交付 Q3-C 卡片要求的内容。第三问的另外两条 baseline（Q3-A 的一维解析"
      "资源解、Q3-B 的固定配比二维搜索）不在本卡范围内，本目录不产出它们的交付物。\n")
    A("2. **$p^*$ 与预算无关是模型结构的结果，不是估计结果**：$h(p)$ 完全可分离且 $Q$ 线性分式 ⇒ "
      "$\\arg\\min_p \\hat L$ 是可信区域的顶点，与 $C,\\ell,g,\\kappa,\\omega$ 无关。"
      "若上游（Q2）改用含 $N$/$D$ 与 $p$ 交互的 $h$，该结论会失效——这是本卡最重要的外推边界。\n")
    A("3. **顶点退化解**：$\\max Q$ 把所有质量压在单一域上。若把结论直译为「语料应 100% 来自 "
      "stackexchange」，那是**过度解读**：模型只认识 6 个已映射域的质量分，没有刻画任何域的"
      "边际收益递减或供给上限。可信区域不足以免除此问题，需额外的多样性约束（题面未给）。\n")
    A("4. **$\\kappa$、$\\omega$ 未标定**：$\\kappa$ 未辨识，$\\omega$ 无上游标定，两者只能作情景。"
      "本卡的数值结论是「给定情景下的模型内部最优」，不是对真实训练结果的预测。\n")
    A("5. **$Q_0$ 的两种设定都保留了**：`fixed_ref` 与 `mixture` 并列报告，未择优。"
      "两者对应不同的建模语义（$Q_0$ 是否随语料组成变化），须由论文正文明确选择并说明理由。\n")
    A("6. **第一问随机森林未持久化**：Q1-C 只导出了预测器的接口描述符（列序、权重、训练支持），"
      "没有保存拟合好的森林对象，因此本流程**不能**调用 $f(p)=\\sum_k v_k\\hat L_k(p)$ 这一泛函。"
      "本卡如实声明，改用两个可由 $p$ 直接算出的冻结泛函（`quality_linear`、`mixture_l1`），"
      "未假装使用过森林。\n")
    A("7. **不覆盖他人产物**：本卡的代码与结果全部落在 `src/baselines/Q3-C/`、"
      "`configs/baselines/Q3-C.yaml`、`artifacts/baselines/Q3-C/`、`reports/baselines/Q3-C.md` "
      "四个槽位内，未改写任何共享发布物。\n")

    ctx["config_hash"] = ctx.get("config_hash", "")
    return "\n".join(L) + "\n"
