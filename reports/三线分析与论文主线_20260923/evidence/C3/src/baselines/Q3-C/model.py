# -*- coding: utf-8 -*-
"""冻结的 Q2 主模型 M1 与第三问的预算/成本结构（§3.1）。

模型（M1 = Q2-B，rho = 0）：
    L(N,D,Q,p) = E + A*N^(-alpha) + B*(D * Q^kappa * h(p))^(-beta)

约束（题给）：
    D * { (6 + eta*l)*N + [g(Q) - g(Q0)]_+ } <= C,   eta = 2e-4

因为 dL/dD = -beta*B*Q^(-kappa*beta)*h^(-beta)*D^(-beta-1) < 0（beta > 0），
Loss 对 D 严格单调下降，且题设无额外 D 上限，故预算必为紧约束（§3.1），可消去 D：

    D = C / u,      u = a*N + c(Q),   a = 6 + eta*l,   c(Q) = [g(Q) - g(Q0)]_+

代入后的一维（给定 p）目标：

    L(N,Q) = E + A*N^(-alpha) + B*T^(-beta),   T = C*Q^kappa*h / u

关于 Q 的取值区间：题目只考虑质量提升且 Loss 对 Q 单调改善（kappa > 0 时
dL/dQ = -beta*B*T^(-beta)*(kappa/Q - c'(Q)/u)），故限定 Q >= Q0；上界取
数据支持范围或 Q2 预测器支持上界（外推须显式标注，§3.1）。

本模块提供解析导数与成本函数，供 search.py 的 SLSQP 与 verify.py 的有限差分核验使用。
"""

from __future__ import annotations

import numpy as np

# 成本函数形式：题给 g(Q) 与其导数 g'(Q) = c'(Q)（c = g(Q)-g(Q0)，常数项导数为 0）
_G_FORMS = {
    "exponential":  ("gamma * exp(lam*Q)",        lambda g, l, Q: g * np.exp(l * Q),
                     lambda g, l, Q: g * l * np.exp(l * Q)),
    "power":        ("gamma * Q^lam",             lambda g, l, Q: g * np.power(Q, l),
                     lambda g, l, Q: g * l * np.power(np.maximum(Q, 1e-300), l - 1.0)),
    "logarithmic":  ("gamma * ln(1 + lam*Q)",     lambda g, l, Q: g * np.log1p(l * Q),
                     lambda g, l, Q: g * l / (1.0 + l * Q)),
}


def g_forms_available() -> list[str]:
    return sorted(_G_FORMS)


class Q3Model:
    """M1 主模型 + 预算约束。所有 Q、N、D 均为原始单位（N 以「个」计，D 以「个」计）。"""

    def __init__(self, cfg: dict):
        p = cfg["model"]["fixed_params"]
        self.E = float(p["E"])
        self.A = float(p["A"])
        self.alpha = float(p["alpha"])
        self.B = float(p["B"])
        self.beta = float(p["beta"])
        self.eta = float(cfg["cost"]["eta"])
        self.g_forms = {k: (float(v["gamma"]), float(v["lam"]))
                        for k, v in cfg["cost"]["g_forms"].items()}

    # ------------------------------------------------------------- 成本
    def a_of_ell(self, ell: float) -> float:
        """训练+注意力 每 token FLOPs 系数 a = 6 + eta*l。"""
        return 6.0 + self.eta * float(ell)

    def attn_share(self, ell: float) -> float:
        """C_attn / C_train = eta*l/6（§3.3）。"""
        return self.eta * float(ell) / 6.0

    def g(self, Q, form: str):
        gamma, lam = self.g_forms[form]
        return _G_FORMS[form][1](gamma, lam, np.asarray(Q, float))

    def g_prime(self, Q, form: str):
        gamma, lam = self.g_forms[form]
        return _G_FORMS[form][2](gamma, lam, np.asarray(Q, float))

    def quality_cost(self, Q, Q0, form: str):
        """c(Q) = [g(Q) - g(Q0)]_+（题给的正部）。"""
        return np.maximum(self.g(Q, form) - self.g(Q0, form), 0.0)

    # ------------------------------------------------------------- 损失
    def predict(self, N, D, Q, h=1.0, kappa=0.0):
        """M1 原始形式；dgrp 偏移不适用（Q3 不在质量实验族内），故不含 delta。"""
        N = np.asarray(N, float); D = np.asarray(D, float)
        Q = np.asarray(Q, float); h = np.asarray(h, float)
        return (self.E
                + self.A * np.power(N, -self.alpha)
                + self.B * np.power(D * np.power(Q, kappa) * h, -self.beta))

    def D_from_budget(self, N, Q, Q0, form: str, C: float, ell: float):
        """消去 D：D = C / u（预算紧约束）。返回 (D, u, c)。"""
        N = np.asarray(N, float); Q = np.asarray(Q, float)
        a = self.a_of_ell(ell)
        c = self.quality_cost(Q, Q0, form)
        u = a * N + c
        return C / u, u, c

    def loss_at_budget(self, N, Q, h, Q0, form: str, C: float, ell: float,
                       kappa: float, need_grad: bool = False):
        """给定 (N,Q,p→h) 与预算，返回消去 D 后的预测 Loss（及可选解析梯度）。

        返回 dict：L, D, u, c, dL_dN, dL_dQ（need_grad 时）。
        """
        N = np.atleast_1d(np.asarray(N, float))
        Q = np.atleast_1d(np.asarray(Q, float))
        h = np.atleast_1d(np.asarray(h, float))
        a = self.a_of_ell(ell)
        c = self.quality_cost(Q, Q0, form)
        u = a * N + c
        D = C / u
        T = C * np.power(Q, kappa) * h / u
        L = self.E + self.A * np.power(N, -self.alpha) + self.B * np.power(T, -self.beta)
        out = {"L": L, "D": D, "u": u, "c": c, "T": T}
        if need_grad:
            # dL/dN = -alpha*A*N^(-alpha-1) + beta*a*B*T^(-beta)/u
            dL_dN = (-self.alpha * self.A * np.power(N, -self.alpha - 1.0)
                     + self.beta * a * self.B * np.power(T, -self.beta) / u)
            # dL/dQ = -beta*B*T^(-beta) * (kappa/Q - c'(Q)/u)
            cp = self.g_prime(Q, form) * (c > 0)          # c>0 时才启用质量成本导数
            dL_dQ = -self.beta * self.B * np.power(T, -self.beta) * (kappa / Q - cp / u)
            out["dL_dN"] = dL_dN
            out["dL_dQ"] = dL_dQ
        return out

    # ------------------------------------------------- 解析最优性的参考解
    def interior_Q_condition(self, N, Q0, form: str, C: float, ell: float, kappa: float):
        """内部解的一阶条件（仅给定 N 时）：kappa/Q = c'(Q)/u。

        返回 F(Q) = c'(Q)*Q - kappa*u(Q) 的符号函数；过零点即内部最优 Q。
        用于给数值解提供一个独立的一致性参照（不替代联合优化）。
        """
        a = self.a_of_ell(ell)
        def F(Q):
            Q = np.asarray(Q, float)
            c = self.quality_cost(Q, Q0, form)
            u = a * N + c
            return self.g_prime(Q, form) * Q - kappa * u
        return F

    def cost_shares(self, N, Q, Q0, h, form: str, C: float, ell: float):
        """成本份额（分母统一为总预算 C，另报告未使用份额，§3.4）。"""
        a = self.a_of_ell(ell)
        c = float(self.quality_cost(Q, Q0, form))
        N = float(N)
        D = C / (a * N + c)
        train = D * 6.0 * N
        attn = D * self.eta * float(ell) * N
        quality = D * c
        used = train + attn + quality
        return {"C_train": train, "C_attn": attn, "C_quality": quality,
                "used": used, "unused": C - used,
                "share_train": train / C, "share_attn": attn / C,
                "share_quality": quality / C, "share_unused": (C - used) / C,
                "D_used": D}
