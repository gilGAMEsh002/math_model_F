"""Adapters that call the ORIGINAL baseline implementations.

Rules:
- upstream trees are read-only; adapters never write into them;
- each adapter loads the frozen module/JSON from the fixed commit recorded in
  upstream.lock.json, and states the physical-unit convention at its boundary;
- any behavioural fix belongs in research/ (new code), not by editing upstream.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

from .common import line_worktree, load_lock

UNIT = 1e9


@contextmanager
def _sys_path(p: Path):
    old = list(sys.path)
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path[:] = old


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- line A
class LineA:
    """Q1-A/Q2-A/Q3-A. Loss coordinate: billions; Q3-A fixes Q=Q0, p=p0."""

    def __init__(self, lock=None):
        self.lock = lock or load_lock()
        self.root = line_worktree(self.lock, "A")
        self.loss = json.loads((self.root / "artifacts/baselines/Q2-A/loss_predictor.json").read_text("utf-8"))
        self.mix = json.loads((self.root / "artifacts/baselines/Q1-A/mixture_predictor.json").read_text("utf-8"))
        self.p0 = json.loads((self.root / "artifacts/baselines/Q1-A/p0.json").read_text("utf-8"))

    @property
    def E(self): return float(self.loss["E"])

    @property
    def A(self): return float(self.loss["A"])

    @property
    def alpha(self): return float(self.loss["alpha"])

    @property
    def B(self): return float(self.loss["B"])

    @property
    def beta(self): return float(self.loss["beta"])

    @property
    def gamma(self): return float(self.loss["gamma"])

    def loss_at(self, N_raw: float, D_raw: float, Q: float | None = None):
        """A's Q2 loss at actual N,D (internally billions). Uses frozen gamma."""
        n, d = N_raw / UNIT, D_raw / UNIT
        q = float(self.p0["Q0"]) if Q is None else float(Q)
        return self.E + self.A * n ** (-self.alpha) + self.B * d ** (-self.beta) + self.gamma * (1.0 - q)


# --------------------------------------------------------------------- line B
class LineB:
    """Q1-B..Q4-B. Q3-B loss uses billions after the unit fix; cost uses raw counts."""

    def __init__(self, lock=None):
        self.lock = lock or load_lock()
        self.root = line_worktree(self.lock, "B")
        self.q2 = json.loads((self.root / "results/q2_scaling_parameters.json").read_text("utf-8"))
        self._q3 = None

    def _load_q3(self):
        if self._q3 is None:
            with _sys_path(self.root / "code"):
                self._q3 = _load_module(self.root / "code/q3_optimize.py", "upstream_b_q3")
        return self._q3

    def loss_at(self, N_raw: float, D_raw: float, Q: float = 1.0):
        """Call B's own fixed Q3 loss (billions internally)."""
        return float(self._load_q3().loss(N_raw, D_raw, Q))

    def anchor(self) -> float:
        return float(self._load_q3().unit_anchor_check())


# --------------------------------------------------------------------- line C
class LineC:
    """Q1-C/Q2-C/Q3-C. Q2-C predictor expects billions; Q3-C model expects raw counts."""

    def __init__(self, lock=None):
        self.lock = lock or load_lock()
        self.root = line_worktree(self.lock, "C")
        self.q1 = json.loads((self.root / "artifacts/baselines/Q1-C/mixture_predictor.json").read_text("utf-8"))
        self._q2 = None
        self._q2_model = None
        self._q3 = None

    def q2_predictor(self):
        """Q2-C official executable predictor module (billion-coordinate N,D)."""
        if self._q2 is None:
            self._q2 = _load_module(
                self.root / "artifacts/baselines/Q2-C/predict_q2c_loss.py", "upstream_c_q2")
        return self._q2

    def q2_config(self) -> dict:
        import yaml
        return yaml.safe_load((self.root / "configs/baselines/Q2-C.yaml").read_text("utf-8"))

    def q3_config(self) -> dict:
        import yaml
        return yaml.safe_load((self.root / "configs/baselines/Q3-C.yaml").read_text("utf-8"))

    def q2_scaling(self) -> dict:
        return json.loads((self.root / "artifacts/baselines/Q2-C/scaling_parameters.json").read_text("utf-8"))

    def q2_stage(self, name: str) -> dict:
        st = self.q2_scaling().get("stages", {})
        if isinstance(st, dict):
            return st.get(name, {})
        for s in st:
            if s.get("name") == name:
                return s
        return {}

    def m1_kappa(self) -> float:
        st = self.q2_stage("stage2_M1_staged")
        if "kappa" in st:
            return float(st["kappa"])
        return float(self.q2_scaling().get("parameters", {}).get("kappa", 0.0))

    def q3_model(self):
        """Q3-C frozen model (raw-coordinate N,D; converts internally)."""
        if self._q3 is None:
            with _sys_path(self.root / "src/baselines/Q3-C"):
                model_mod = _load_module(self.root / "src/baselines/Q3-C/model.py", "upstream_c_q3_model")
                self._q3 = model_mod.Q3Model(self.q3_config())
        return self._q3

    def q2_loss_at(self, N_raw: float, D_raw: float, Q: float, h: float = 1.0,
                   model: str = "M1") -> float:
        """Official Q2-C predictor; N,D passed in billion coordinates (per its docstring)."""
        if self._q2_model is None:
            pred = self.q2_predictor()
            self._q2_model = pred.Q2CPredictor(str(self.root / "artifacts/baselines/Q2-C/loss_predictor.json"))
        m = self._q2_model
        m.rh = 0.0                       # recommended M1 (rho=0)
        if model == "M1":
            m.ka = self.m1_kappa()
        return float(m.predict(N_raw / UNIT, D_raw / UNIT, Q, h, dgrp=0.0))

    def q3_loss_at_budget(self, N_raw: float, Q: float, h: float, Q0: float, form: str,
                          C: float, ell: float, kappa: float, need_grad: bool = False) -> dict:
        out = self.q3_model().loss_at_budget([N_raw], [Q], [h], Q0, form, C, ell, kappa,
                                             need_grad=need_grad)
        return {k: (float(v[0]) if hasattr(v, "__len__") else float(v)) for k, v in out.items()}

    def q1_predictor_status(self) -> dict:
        """Q1-C exposes only a descriptor; no persisted executable model object.

        Recorded as a blocker rather than silently 'passing': arbitrary-p calls are
        impossible until the forest coefficients/model are persisted.
        """
        m = self.q1
        required = ["input_columns", "output_columns", "target_weights_v",
                    "p0_reference_mixture_mean", "training_support", "row_normalization"]
        missing = [k for k in required if k not in m]
        has_model_obj = any((self.root / "artifacts/baselines/Q1-C").glob("*.npz")) or \
            any((self.root / "artifacts/baselines/Q1-C").glob("*.pkl")) or \
            any((self.root / "artifacts/baselines/Q1-C").glob("*.joblib"))
        return {
            "descriptor_present": True,
            "descriptor_keys_ok": not missing,
            "missing_keys": missing,
            "n_inputs": m.get("n_inputs"), "n_outputs": m.get("n_outputs"),
            "row_normalization": m.get("row_normalization"),
            "executable": bool(has_model_obj),
            "blocker": None if has_model_obj else
                       "仅描述符；未持久化森林系数/模型对象，无法对任意合法 p 调用",
        }




def get_line(name: str, lock=None):
    name = name.upper()
    return {"A": LineA, "B": LineB, "C": LineC}[name](lock)
