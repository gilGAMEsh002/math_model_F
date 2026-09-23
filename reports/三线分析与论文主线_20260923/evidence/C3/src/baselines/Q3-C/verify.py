# -*- coding: utf-8 -*-
"""数值核验（总计划 §6 第 3、4、5 条；Q3-C「按公共协议完成必要的数值核验」）。

覆盖四类会改变结论的错误：
  1. 单位与量级：N/D 采用「个」为单位、成本数量级与 eta*l/6 份额；
  2. 导数：解析梯度与有限差分一致（差分的步长与容差写进配置）；
  3. 优化：成本不超预算、配比可行（单纯形 + 可信区域）、一阶条件与边界符号自洽；
  4. 稳定性：网格加密/初值变化不制造伪转移；口径与上游预测器逐点一致。

另含一项输入审计：把本流程实际读入的文件（哈希）全部列出，用于证明
「未读取任何最终检验集（A6–A11）」——即不存在用检验 Loss 偷选配方。
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd

from common import hash_if_exists, q1_path, q2_path, which_root
from scenarios import classify_q_regime as _regime


# ------------------------------------------------------------------ 1. 上游一致
def _load_upstream_predictor(code_path: str, param_path: str):
    """载入 Q2→Q3 官方接口：**代码**在 .py 里，**参数**在同目录的 .json 里。

    两者必须分开传入：`Q2CPredictor.__init__(path)` 打开的是 JSON 参数字典，
    而类的定义在被执行的模块里。只传其一都会得到"JSON 解析失败"这类误导性报错。
    """
    spec = importlib.util.spec_from_file_location("predict_q2c_loss", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Q2CPredictor(param_path)


def upstream_consistency(cfg: dict, model, kappa: float, rng_seed: int = 20260923,
                         n: int = 40) -> pd.DataFrame:
    """与 Q2→Q3 官方接口 predict_q2c_loss.py 逐点比对（M1：rho=0）。

    这是**口径核验**，不是新拟合：两边算的必须是同一个数。先核对参数本身
    （上游 JSON 的 E/A/alpha/B/beta 与本地 config 的 fixed_params 是否逐位相同），
    再逐点核对预测值——只比预测值会掩盖"参数抄错但恰好接近"的情况。
    """
    up = _load_upstream_predictor(q2_path(cfg, cfg["upstream"]["q2_code_file"]),
                                  q2_path(cfg, cfg["upstream"]["q2_predictor_file"]))
    local = cfg["model"]["fixed_params"]
    param_match = {k: bool(abs(float(up.d["parameters"][k]) - float(local[k])) <= 0.0)
                   for k in ("E", "A", "alpha", "B", "beta")}
    up.rh = 0.0                      # Q2-C 结论：主模型退回 M1(rho=0)
    up.ka = float(kappa)
    rng = np.random.default_rng(rng_seed)
    N = 10.0 ** rng.uniform(8.0, 11.5, n)
    D = 10.0 ** rng.uniform(9.0, 12.5, n)
    Q = rng.uniform(0.2, 1.0, n)
    h = rng.uniform(0.5, 2.0, n)
    a = model.predict(N, D, Q, h, kappa)
    b = up.predict(N, D, Q, h, dgrp=np.zeros(n))
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    ok = bool(all(param_match.values())) and rel.max() < 1e-12
    return pd.DataFrame([{
        "n": n, "kappa": float(kappa),
        "params_identical_EAalphaBbeta": bool(all(param_match.values())),
        "param_mismatch": ";".join(k for k, v in param_match.items() if not v),
        "max_rel_diff": float(rel.max()), "median_rel_diff": float(np.median(rel)),
        "verdict": ("参数逐位相同且预测逐点一致（同口径）" if ok
                    else "与上游预测器不一致，须查明"),
    }])


# ------------------------------------------------------------------ 2. 导数核验
def derivative_check(model, cfg: dict, cases: list[dict]) -> pd.DataFrame:
    """解析梯度 vs 中心有限差分（仅用于核验导数，不把差分当弹性）。"""
    h = float(cfg["verify"]["fd_rel_step"])
    rows = []
    for cs in cases:
        N, Q = cs["N"], cs["Q"]
        kw = dict(h=cs["h"], Q0=cs["Q0"], form=cs["form"], C=cs["C"], ell=cs["ell"],
                  kappa=cs["kappa"])
        o = model.loss_at_budget(np.array([N]), np.array([Q]), need_grad=True, **kw)
        gN_an = float(o["dL_dN"][0]); gQ_an = float(o["dL_dQ"][0])
        lp = model.loss_at_budget(np.array([N * (1 + h)]), np.array([Q]), **kw)["L"][0]
        lm = model.loss_at_budget(np.array([N * (1 - h)]), np.array([Q]), **kw)["L"][0]
        gN_fd = (lp - lm) / (2 * N * h)
        qp = min(Q * (1 + h), 1.0); qm = max(Q * (1 - h), 1e-9)
        Lp = model.loss_at_budget(np.array([N]), np.array([qp]), **kw)["L"][0]
        Lm = model.loss_at_budget(np.array([N]), np.array([qm]), **kw)["L"][0]
        gQ_fd = (Lp - Lm) / (qp - qm)
        rows.append({
            "N_B": N / 1e9, "Q": Q, "cost_form": cs["form"], "ell": int(cs["ell"]),
            "kappa": float(cs["kappa"]),
            "dL_dN_analytic": gN_an, "dL_dN_fd": gN_fd,
            "dL_dN_rel_err": abs(gN_an - gN_fd) / max(abs(gN_an), 1e-300),
            "dL_dQ_analytic": gQ_an, "dL_dQ_fd": gQ_fd,
            "dL_dQ_rel_err": abs(gQ_an - gQ_fd) / max(abs(gQ_an), 1e-300),
        })
    df = pd.DataFrame(rows)
    df["pass"] = ((df["dL_dN_rel_err"] < float(cfg["verify"]["fd_rel_tol"])) &
                  (df["dL_dQ_rel_err"] < float(cfg["verify"]["fd_rel_tol"])))
    return df


# ------------------------------------------------------------------ 3. 约束核验
def constraint_check(model, cfg: dict, sols: pd.DataFrame) -> pd.DataFrame:
    """逐解回算：预算是否紧、成本份额是否加和、配比是否可行。

    输入 sols 需含 C, ell, cost_form, kappa, N_B, D_B, Q, h, Q0, 以及 share_* 列。
    """
    rows = []
    for _, r in sols.iterrows():
        N = float(r["N_B"]) * 1e9
        Q = float(r["Q"]); Q0 = float(r["Q0"]); h = float(r["h"])
        form = r["cost_form"]; C = float(r["C"]); ell = float(r["ell"])
        a = model.a_of_ell(ell)
        c = float(model.quality_cost(Q, np.array([Q0]), form)[0])
        u = a * N + c
        D_implied = C / u
        used = D_implied * u
        rows.append({
            "C": C, "ell": int(ell), "cost_form": form, "Q0_mode": r.get("q0_mode", ""),
            "budget_rel_residual": abs(C - used) / C,
            "D_recomputed_vs_reported_rel": abs(D_implied - float(r["D_B"]) * 1e9) /
                                            max(float(r["D_B"]) * 1e9, 1e-300),
            # 四个份额分量一并留存：否则"份额加和 = 1"这条核验无法只凭本文件复核，
            # 必须再去读 optimal_allocations.csv 拼列——核验表应当自洽可审。
            "share_train": float(r["share_train"]),
            "share_attn": float(r["share_attn"]),
            "share_quality": float(r["share_quality"]),
            "share_unused": float(r["share_unused"]),
            "share_sum": float(r["share_train"] + r["share_attn"] +
                               r["share_quality"] + r["share_unused"]),
            "share_train_plus_attn_ratio_attn_over_train": (
                float(r["share_attn"]) / float(r["share_train"])
                if float(r["share_train"]) > 0 else np.nan),
            "attn_share_formula_check": abs(
                (float(r["share_attn"]) / float(r["share_train"])
                 if float(r["share_train"]) > 0 else np.nan) - model.attn_share(ell)),
            "Q_ge_Q0": bool(Q >= Q0 - 1e-9),
            "Q_le_upper": bool(Q <= float(cfg["quality"]["q_extended_max"]) + 1e-9),
        })
    df = pd.DataFrame(rows)
    df["pass_budget"] = df["budget_rel_residual"] < float(cfg["verify"]["budget_rel_tol"])
    df["pass_share_sum"] = np.isclose(df["share_sum"], 1.0, atol=1e-9)
    df["pass_attn_formula"] = df["attn_share_formula_check"] < 1e-12
    return df


def simplex_check(p: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float,
                  in_trust_box: bool = True) -> dict:
    """配比可行性：单纯形约束（永远必须）与可信区域盒约束（仅对盒内候选要求）。

    为什么要把两件事分开：候选集里有意保留了若干**盒外对照点**（`trust_status`
    为 `box_only` / `outside`），它们是用来量化「可信区域到底约束了什么」的反事实，
    本来就不在盒内。若把盒约束也当成通过条件，就会把"故意越界"记成"不可行"，
    从而掩盖真正的可行性信息。故：
        pass_simplex := Σp=1 且 p>=0（对所有候选都要求，这是优化问题的硬约束）
        pass_box     := 在盒内（仅当 in_trust_box=True 时参与判定）
        pass         := pass_simplex 且 (pass_box 或 该候选本就登记为盒外)
    同时仍如实报告 `box_violation_max`，让盒外的偏离量可见。
    """
    p = np.asarray(p, float)
    sum_err = float(p.sum() - 1.0)
    viol = float(np.max(np.maximum(lo - p, p - hi).clip(min=0.0))) if p.size else 0.0
    pass_simplex = bool(abs(sum_err) < tol and p.min() >= -tol)
    pass_box = bool(viol <= tol)
    return {"sum_minus_1": sum_err, "min_p": float(p.min()),
            "box_violation_max": viol,
            "in_trust_box_declared": bool(in_trust_box),
            "pass_simplex": pass_simplex, "pass_box": pass_box,
            "pass": bool(pass_simplex and (pass_box or not in_trust_box))}


# ------------------------------------------------------------------ 4. 稳定性
def grid_stability(model, cfg: dict, search_fn, h: float, Q0: float, form: str,
                   C: float, ell: float, kappa: float) -> pd.DataFrame:
    """换个更粗的网格重解，检查**结论**是否改变（§3.4 反伪转移）。

    判据是什么：这个检查的目的是排除"网格造成的伪转移"，即**结论**随网格分辨率变化。
    结论 = (a) 预测损失、(b) $Q^*$ 的状态（下界/内部/上界）。故
        stable := |dL| <= 1e-6  且  regime 与配置网格相同
    $N^*$ 与 $Q^*$ 的**位置**只作诊断量报告，不参与判定：本模型的 $\\hat L$ 在最优点
    附近极平坦（候选间跨度仅 ~1e-3），要求 $\\Delta\\log_{10}N<0.05$ 会把"平坦谷底里
    位置略有移动"误判为不稳定。
    """
    q_hi = float(cfg["quality"]["q_extended_max"])
    tol_b = float(cfg["verify"]["transition"]["q_boundary_tol"])
    base = search_fn(h=h, Q0=Q0, form=form, C=C, ell=ell, kappa=kappa, q_lo=Q0, q_hi=q_hi)
    reg0 = _regime(base["Q"], Q0, q_hi, tol_b)
    rows = [{"variant": "configured",
             "n_grid": int(cfg["search"]["N_gross_grid"]), "q_grid": int(cfg["search"]["Q_grid"]),
             "L": base["L"], "N_B": base["N"] / 1e9, "Q": base["Q"], "regime": reg0}]
    for f in cfg["verify"]["grid_coarsen_factors"]:
        nd, qd = int(cfg["search"]["N_gross_grid"] * f), int(cfg["search"]["Q_grid"] * f)
        r = search_fn(h=h, Q0=Q0, form=form, C=C, ell=ell, kappa=kappa,
                      q_lo=Q0, q_hi=q_hi, n_grid=nd, q_grid=qd)
        rows.append({"variant": f"coarsen_x{f}", "n_grid": nd, "q_grid": qd,
                     "L": r["L"], "N_B": r["N"] / 1e9, "Q": r["Q"],
                     "regime": _regime(r["Q"], Q0, q_hi, tol_b)})
    df = pd.DataFrame(rows)
    df["dL"] = df["L"] - base["L"]
    df["dlog10N"] = (np.log10(df["N_B"]) - np.log10(base["N"] / 1e9)).abs()
    df["dQ"] = (df["Q"] - base["Q"]).abs()
    df["L_rel"] = df["dL"].abs() / abs(base["L"])
    df["regime_same"] = df["regime"] == reg0
    df["stable"] = (df["L_rel"] <= float(cfg["verify"]["grid_l_rel_tol"])) & df["regime_same"]
    return df


def cell_ranking(tab: pd.DataFrame, keys: list[str], lcol: str = "L") -> pd.DataFrame:
    """按情景单元汇总：argmin L、argmax h、argmax Q(p) 是否指向同一配比。

    这是 Q3-C 要求的核心核验——**最优配比是否随预算变化**。理论预期是：
    $\\hat L$ 对 $h(p)$ 可分离且单调下降，故 $\\arg\\min_p \\hat L = \\arg\\max_p h$，
    与 $C,\\ell,g,\\kappa,\\omega$ 无关（不随预算转移）。因此不能只报告"没发现转移"，
    而要给出等式在哪几个单元上成立。

    判据一律用**数值容差**而非标签相等：LP 顶点与多起点 SLSQP 可能落在同一点
    （$Q$ 只差 $\\sim10^{-16}$），比标签会产生"不一致"的假警报。同时记录单元内
    候选间 $L$ 的跨度——跨度极小说明"哪个 $p$ 最优"无法由数值分辨 $L$ 得出，
    只能由 $h$ 的单调性论证给出，这一点必须如实写进报告。

    需要列：keys、candidate、h、Q_cand 与 lcol。
    """
    rows = []
    for k, g in tab.groupby(keys):
        kk = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        iL = g[lcol].idxmin(); ih = g["h"].idxmax(); iQ = g["Q_cand"].idxmax()
        Lmin = float(g.loc[iL, lcol]); hmax = float(g.loc[ih, "h"])
        h_at_L = float(g.loc[iL, "h"]); Q_at_h = float(g.loc[ih, "Q_cand"])
        hmin = float(g["h"].min())
        # ω=0 时 h≡1，候选间 h 无差异 ⇒ argmax h 是全体的平局，"p 是否最优"根本不可辨识。
        # 这类单元必须单独标出：否则"argmax h 是不是 argmax Q"会拿平局中任意一个来判，
        # 得出的是随机标签而非结论。
        h_ident = bool(hmax - hmin > 1e-12)
        rows.append({
            **kk,
            "argmin_L_candidate": str(g.loc[iL, "candidate"]), "L_min": Lmin,
            "argmax_h_candidate": str(g.loc[ih, "candidate"]), "h_max": hmax,
            "h_min": hmin, "h_identifiable": h_ident,
            "argmax_Q_candidate": str(g.loc[iQ, "candidate"]),
            "Q_cand_max": float(g.loc[iQ, "Q_cand"]),
            "L_spread": float(g[lcol].max() - g[lcol].min()),
            "n_tied_at_L_min": int((g[lcol] <= Lmin + 1e-12).sum()),
            "argmin_L_is_argmax_h": bool(h_at_L >= hmax * (1.0 - 1e-9)),
            "argmax_h_is_argmax_Q": bool(abs(Q_at_h - float(g["Q_cand"].max())) <= 1e-9),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. 输入审计
def input_audit(cfg: dict) -> pd.DataFrame:
    """列出实际读入的全部文件及其哈希，并标注它属于哪一类数据角色。

    「只读训练/接口文件、不读最终检验集」这条声明在本表上可被独立复核：
    role 列里不应出现 A6–A11 或 test_* 之类的最终检验文件。
    """
    items = [
        ("Q1-C", cfg["upstream"]["q1_predictor_file"], "q1"),
        ("Q1-C", "quality_domain_mapping.csv", "q1"),
        ("Q2-C", cfg["upstream"]["q2_predictor_file"], "q2"),
        ("Q2-C", "mixture_transfer_scenarios.json", "q2"),
        ("Q2-C", "scaling_parameters.json", "q2"),
        ("C7", cfg["context"]["c7_file"], "c7"),
    ]
    rows = []
    for owner, name, role in items:
        try:
            p = (q1_path(cfg, name) if role == "q1" else
                 q2_path(cfg, name) if role == "q2" else
                 os.path.join(which_root(cfg, os.path.join(cfg["paths"]["c7_dir"], name)),
                              cfg["paths"]["c7_dir"], name))
            rel = os.path.join(cfg["paths"]["q1_artifacts"] if role == "q1" else
                               cfg["paths"]["q2_artifacts"] if role == "q2" else
                               cfg["paths"]["c7_dir"], name)
            rows.append({"owner": owner, "role": role, "file": name,
                         "resolved_root": which_root(cfg, rel),
                         "sha256": hash_if_exists(p), "exists": os.path.isfile(p)})
        except FileNotFoundError as e:
            rows.append({"owner": owner, "role": role, "file": name,
                         "resolved_root": "", "sha256": None, "exists": False,
                         "note": str(e)[:120]})
    df = pd.DataFrame(rows)
    df["is_test_set"] = df["file"].astype(str).str.contains(
        r"test_mixture|test_pile_loss|est_mixture|est_pile_loss|leaderboard|"
        r"detailed_results", regex=True)
    return df
