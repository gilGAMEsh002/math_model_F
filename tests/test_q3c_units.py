# -*- coding: utf-8 -*-
"""Q3-C 单位与导数回归测试（P0 修复）。

背景：Q2-C 的 A、B 系数以「十亿参数 / 十亿 Token」坐标拟合，而 Q3-C 成本端用实际个数。
修复前 model.predict / loss_at_budget 把实际个数直接代入十亿坐标系数，使 Loss 被压到
不可约项附近（N=1e9,D=1e11,Q=1 时 1.691141，正确值 2.385578）。

运行：pytest tests/test_q3c_units.py -q
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines" / "Q3-C"))

from model import Q3Model  # noqa: E402

CFG = {
    "model": {"fixed_params": {"E": 1.689797562918097, "A": 0.35398032067156865,
                               "alpha": 0.33997658189164076, "B": 1.2403055835454702,
                               "beta": 0.2798781285456506}},
    "cost": {"eta": 2.0e-4,
             "g_forms": {"exponential": {"gamma": 1.0e7, "lam": 6.0},
                         "power": {"gamma": 5.0e9, "lam": 4.0},
                         "logarithmic": {"gamma": 2.0e9, "lam": 10.0}}},
}


def test_physical_unit_anchor():
    m = Q3Model(CFG)
    L = float(np.asarray(m.predict(1e9, 1e11, 1.0, 1.0, 0.804707)).ravel()[0])
    assert abs(L - 2.385578) < 1e-5, f"got {L}"
    assert abs(L - 1.691141) > 0.5, "旧错误单位值仍出现"


def test_q1_degenerate_kappa():
    m = Q3Model(CFG)
    L0 = float(np.asarray(m.predict(1e9, 1e11, 1.0, 1.0, 0.0)).ravel()[0])
    for k in (0.26, 0.38, 0.80, 1.05):
        Lk = float(np.asarray(m.predict(1e9, 1e11, 1.0, 1.0, k)).ravel()[0])
        assert abs(Lk - L0) < 1e-12


def test_loss_at_budget_scales_with_magnitudes():
    m = Q3Model(CFG)
    L_small = float(m.loss_at_budget(np.array([1e9]), np.array([0.416633]),
                                     1.0, 0.416633, "power", 1e19, 4096, 0.8)["L"][0])
    # 修复前该点约为 ~1.69（贴近 E）；修复后应落在合理量级
    assert L_small > 1.75, f"loss collapsed toward E: {L_small}"


def test_right_derivative_at_q0():
    """Q=Q0 处应取右导数 g'(Q0)，而不是把 c=0 处的导数置零。"""
    m = Q3Model(CFG)
    Q0 = 0.416633
    o = m.loss_at_budget(np.array([1e9]), np.array([Q0]), 1.0, Q0, "power",
                         1e22, 4096, 0.804707, need_grad=True)
    g_an = float(o["dL_dQ"][0])
    h = 1e-6
    Lp = float(m.loss_at_budget(np.array([1e9]), np.array([Q0 * (1 + h)]), 1.0, Q0,
                                "power", 1e22, 4096, 0.804707)["L"][0])
    g_fd = (Lp - float(o["L"][0])) / (Q0 * h)
    assert abs(g_an - g_fd) / max(abs(g_fd), 1e-30) < 1e-3, (g_an, g_fd)
    # g'(Q0) > 0，故右导数比"置零"版本更负
    # 质量边际成本进入右导数：与"把 c=0 处导数置零"的旧实现可区分
    g_zero = -m.beta * m.B * float(np.power(o["T"], -m.beta)[0]) * (0.804707 / Q0)
    assert abs(g_an - g_zero) > 1e-6
    assert g_an > g_zero
