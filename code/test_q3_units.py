# -*- coding: utf-8 -*-
"""B 线第三问单位回归测试（P0 修复）。

背景：Q2-B 的 A、B 系数以「十亿参数 / 十亿 Token」为坐标拟合，而第三问的成本项
6ND、ηNDℓ 使用实际个数。修复前 q3_optimize.loss() 把实际个数直接代入十亿坐标系数，
使 Loss 被压到不可约项附近（N=1e9,D=1e11,Q=1 时得到 1.691141，正确值 2.385578）。

运行：pytest code/test_q3_units.py -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q3_optimize as q


def test_physical_unit_anchor():
    """N=1e9, D=1e11, Q=1, h=1 的经典项应为 2.385578，而非错误单位下的 1.691141。"""
    L = float(q.loss(1e9, 1e11, 1.0))
    assert abs(L - 2.385578) < 1e-5, f"got {L}"
    assert abs(L - 1.691141) > 0.5, "旧错误单位值仍出现"


def test_q1_degenerate_kappa():
    """Q=1 时质量指数 κ 不应影响损失（退化测试）。"""
    L = float(q.loss(1e9, 1e11, 1.0))
    for kappa in (0.0, 0.3, 0.8047, 1.0524):
        assert abs(float(q.loss(1e9, 1e11, 1.0, {**q.TH, "kappa": kappa})) - L) < 1e-12


def test_analytic_matches_numeric_at_fixed_Q():
    """固定 Q=Q0 时解析 N* 应与数值搜索一致（同单位下）。"""
    cf = q.COST_FUNCS["指数型"]
    for C, ell in [(1e19, 2048), (1e19, 8192), (1e22, 4096)]:
        r = q.solve_2d(C, ell, cf, q.Q0_DEFAULT, fix_Q=q.Q0_DEFAULT)
        na = q.analytic_Nstar(C, ell, cf, q.Q0_DEFAULT)
        assert abs(r["N"] / na - 1) < 1e-5, (C, ell, r["N"], na)


def test_loss_decreases_with_N_and_D():
    assert q.loss(2e9, 1e11, 1.0) < q.loss(1e9, 1e11, 1.0)
    assert q.loss(1e9, 2e11, 1.0) < q.loss(1e9, 1e11, 1.0)
