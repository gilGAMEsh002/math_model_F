# -*- coding: utf-8 -*-
"""Q2-C 损失预测器（Q2→Q3 接口）。

用法：
    from predict_q2c_loss import Q2CPredictor
    m = Q2CPredictor("loss_predictor.json")
    L = m.predict(N=7.0, D=300.0, Q=0.8, h=1.0, dgrp=0.0)

注意（见 loss_predictor.json 的 warnings）：
    * κ、ρ 未通过可辨识性判定时，本预测器只用于情景分析；
    * ω 未经标定，h 只在情景网格下取值；
    * 族外来源存在未吸收的水平差。
"""

import json

import numpy as np


class Q2CPredictor:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            self.d = json.load(fh)
        p = self.d["parameters"]
        self.E, self.A, self.al = p["E"], p["A"], p["alpha"]
        self.B, self.be = p["B"], p["beta"]
        self.ka, self.rh, self.de = p["kappa"], p["rho"], p["delta"]

    def predict(self, N, D, Q, h=1.0, dgrp=0.0):
        N = np.asarray(N, float); D = np.asarray(D, float)
        Q = np.asarray(Q, float); h = np.asarray(h, float)
        dgrp = np.asarray(dgrp, float)
        return (self.E + self.de * dgrp
                + self.A * np.power(N * np.power(Q, self.rh), -self.al)
                + self.B * np.power(D * np.power(Q, self.ka) * h, -self.be))

    def h_from_delta(self, delta, omega):
        """h = exp(-omega*Delta)；omega=0 即零效应（嵌套回 Q2-B）。"""
        return np.exp(-np.asarray(omega, float) * np.asarray(delta, float))
