"""Independent continuous sensitivity check for R3b scale-parameter scenarios.
Uses the frozen line-C Q3-C model/config and alpha/beta 95% endpoints.
Writes only next to this audit script.
"""
from pathlib import Path
import importlib.util
import yaml
import pandas as pd
from scipy.optimize import differential_evolution

AUDIT = Path(__file__).resolve().parent
CROOT = Path('/home/sshuser/projects/数学建模26/F题-upstream/line-c')
MODEL = CROOT / 'src/baselines/Q3-C/model.py'
CONFIG = CROOT / 'configs/baselines/Q3-C.yaml'
BOOT = CROOT / 'artifacts/baselines/Q2-C/validation_bootstrap_M0.csv'

cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('frozen_q3_model_audit', MODEL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
boot = pd.read_csv(BOOT).set_index('param')
base_model = mod.Q3Model(cfg)
Q0 = 0.416633
KAPPA = float(cfg['model']['kappa_scenarios']['M1_staged'])
FORM = 'exponential'
ROWS = []

for C in (1e19, 1e22, 1e24):
    for ell in (2048, 8192):
        def solve(alpha=None, beta=None):
            model = mod.Q3Model(cfg)
            if alpha is not None:
                model.alpha = float(alpha)
            if beta is not None:
                model.beta = float(beta)
            def objective(x):
                N = (10.0 ** x[0]) * 1e9
                Q = x[1]
                return float(model.loss_at_budget([N], [Q], [1.0], Q0, FORM, C, ell, KAPPA)['L'][0])
            opt = differential_evolution(objective, [(-3.0, 3.0), (Q0, 1.0)],
                                         seed=3, tol=1e-11, polish=True)
            return {'N_B': 10.0 ** opt.x[0], 'Q': opt.x[1], 'L': opt.fun}
        baseline = solve()
        ROWS.append({'C': C, 'ell': ell, 'scenario': 'baseline', 'param': '', 'value': '',
                     **baseline, 'N_rel_change': 0.0})
        for param in ('alpha', 'beta'):
            for endpoint in ('lo95', 'hi95'):
                value = float(boot.loc[param, endpoint])
                changed = solve(**{param: value})
                ROWS.append({'C': C, 'ell': ell, 'scenario': endpoint, 'param': param,
                             'value': value, **changed,
                             'N_rel_change': changed['N_B'] / baseline['N_B'] - 1.0})

out = pd.DataFrame(ROWS)
out.to_csv(AUDIT / 'q3_continuous_sensitivity.csv', index=False)
# Derive the final R3b grid spacing from its two-round algorithm.
last_log10_step = (2 * (6.0 / 20.0) / 4.0) / 39.0
summary = pd.DataFrame([{
    'r3b_initial_log10_range': 6.0,
    'refine_rounds': 2,
    'points_per_round': 40,
    'last_log10_step': last_log10_step,
    'implied_relative_N_step': 10.0 ** last_log10_step - 1.0,
    'continuous_max_abs_N_rel_change': out.loc[out.param.ne(''), 'N_rel_change'].abs().max(),
    'continuous_min_abs_nonzero_N_rel_change': out.loc[out.param.ne('') & out.N_rel_change.ne(0), 'N_rel_change'].abs().min(),
}])
summary.to_csv(AUDIT / 'q3_continuous_sensitivity_summary.csv', index=False)
print(summary.to_string(index=False))
print(out.to_string(index=False))
