"""Independent audit of new B/C saved artifacts; no baseline outputs are changed."""
from pathlib import Path
from itertools import combinations
import importlib.util,json,hashlib,subprocess
import numpy as np
import pandas as pd
import yaml
R=Path(__file__).resolve().parent
C3=R/'evidence/C3'
C2=R/'evidence/C2'
B=R/'evidence/B'
cfg=yaml.safe_load((C3/'configs/baselines/Q3-C.yaml').read_text())
spec=importlib.util.spec_from_file_location('audit_q3_model',C3/'src/baselines/Q3-C/model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);model=m.Q3Model(cfg)
ka=cfg['model']['kappa_scenarios']['M1_staged']
spec=importlib.util.spec_from_file_location('audit_q2_model',C2/'artifacts/baselines/Q2-C/predict_q2c_loss.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
up=u.Q2CPredictor(C2/'artifacts/baselines/Q2-C/loss_predictor.json');up.rh=0.;up.ka=ka
cases=[]
for n,d,q in [(1.,100.,1.),(7.,300.,.8),(.2,7.,.416633)]:
 wrong=float(model.predict(n*1e9,d*1e9,q,1.,ka))
 right=float(up.predict(n,d,q,1.,0.))
 cases.append(dict(N_B=n,D_B=d,Q=q,Q3_saved_unit_prediction=wrong,Q2_same_physical_point_prediction=right,absolute_gap=abs(wrong-right)))
alloc=pd.read_csv(C3/'artifacts/baselines/Q3-C/optimal_allocations.csv')
pred_right=up.predict(alloc.N_B,alloc.D_B,alloc.Q,alloc.h_true,0.)
pred_wrong=model.predict(alloc.N_B*1e9,alloc.D_B*1e9,alloc.Q,alloc.h_true,ka)
assert np.allclose(pred_wrong,alloc.L_pred,atol=1e-12)
cand=pd.read_csv(C3/'artifacts/baselines/Q3-C/candidates.csv')
ref=cand[cand.is_p0].iloc[0]
pcols=[c for c in cand if c.startswith('train_the_pile_')]
vertex=cand[cand.candidate=='lp_vertex_box'].iloc[0]
l1=float(np.abs(vertex[pcols].astype(float)-ref[pcols].astype(float)).sum())
# The analytic reference remains fixed-Q, h=1 and uses billion-unit coefficients.
p=cfg['model']['fixed_params'];C=1e19;ell=4096;q0=.416633
k=6+2e-4*ell;K=C/(1e18*k)
n=(p['alpha']*p['A']/(p['beta']*p['B'])*K**p['beta']*q0**(ka*p['beta']))**(1/(p['alpha']+p['beta']))
d=K/n
cost=(6+2e-4*alloc.ell)*alloc.N_B*alloc.D_B*1e18
cost+=alloc.D_B*1e9*alloc.c_quality
assert np.allclose(cost,alloc.C,rtol=1e-10)
summary={'C3_units_cases':cases,'C3_saved_L_matches_raw_unit_code':True,
 'C3_correct_unit_rescore_only_not_reoptimization':{'median_abs_gap':float(np.median(np.abs(pred_right-alloc.L_pred))), 'max_abs_gap':float(np.max(np.abs(pred_right-alloc.L_pred)))},
 'C3_budget_feasibility_recomputed':True,'C3_rows':len(alloc),
 'C3_winners_outside_declared_L1':int((~alloc.in_trust_l1).sum()),
 'C3_vertex_l1_distance':l1,'C3_declared_l1_radius':cfg['mixtures']['trust_l1_radius'],
 'C3_unique_kappa_allocations':alloc.kappa.unique().tolist(),
 'C3_unique_kappa_budget_paths':pd.read_csv(C3/'artifacts/baselines/Q3-C/budget_paths.csv').kappa.unique().tolist(),
 'correct_unit_fixed_Q_h1_analytic_reference':{'C':C,'ell':ell,'Q':q0,'N_B':n,'D_B':d,'L':float(up.predict(n,d,q0,1.,0.))},
 'limits':'No full retraining or reoptimization. Correct-unit rescoring is a diagnostic, not a corrected optimal allocation.'}
(R/'new_branch_verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
pd.DataFrame(cases).to_csv(R/'c3_unit_counterexamples.csv',index=False)
cols=['functional','omega','C','cost_form','ell','q0_mode','N_B','D_B','Q','h_true','L_pred','trust_status']
x=alloc[cols].copy();x['L_correct_unit_rescore']=pred_right;x.to_csv(R/'c3_rescore_diagnostic.csv',index=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))

bp=json.loads((B/'results/q3_optimization_meta.json').read_text())['parameters']
ba=pd.read_csv(B/'results/q3_optimal_allocations.csv')
def b_loss(n,d,q):
    return bp['E']+bp['A']*n**(-bp['alpha'])+bp['B']*(d*q**bp['kappa'])**(-bp['beta'])
braw=b_loss(ba.N,ba.D,ba.Q)
assert np.allclose(braw,ba.L,atol=6e-6)
summary['B3_same_unit_error']={'saved_L_matches_raw_unit_formula_to_csv_rounding':True,
 'one_B_params_100_B_tokens_Q1_raw':float(b_loss(1e9,1e11,1)),
 'one_B_params_100_B_tokens_Q1_correct':float(b_loss(1,100,1)),
 'median_correct_unit_rescore_gap':float(np.median(np.abs(b_loss(ba.N/1e9,ba.D/1e9,ba.Q)-ba.L))),
 'rows':len(ba)}
(R/'new_branch_verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')

from scipy.stats import spearmanr
coef=np.load(B/'results/q1_mixture_model.npz')
recorded=pd.read_csv(B/'results/q1_mixture_eval.csv')
regrets=pd.read_csv(B/'results/q1_mixture_regret.csv')
combined=pd.read_csv(R/'q1_comparison_verified.csv').to_dict('records')
repo=R.parents[1]
mapping={'test_1m':('1m','same_scale_1m'),'test_60m':('60m','cross_scale_60m'),
 'test_1B':('1B','cross_scale_1B'),'est_10b':('10b','est_10b'),'est_70b':('70b','est_70b')}
raw_inputs=[]
for split,(suffix,dataset) in mapping.items():
    prefix='est' if split.startswith('est') else 'test'
    inp=repo/'real_attachments/A_data_value/regmix_tables'
    for filename in [f'{prefix}_mixture_{suffix}.csv',f'{prefix}_pile_loss_{suffix}.csv']:
        rel=(inp/filename).relative_to(repo).as_posix()
        data=(inp/filename).read_bytes()
        frozen=subprocess.check_output(['git','show','03921456b512d2c8dbd5ed052ef0f22d8de91412:'+rel],cwd=repo)
        assert data==frozen
        raw_inputs.append({'path':rel,'sha256':hashlib.sha256(data).hexdigest(),'matches_frozen_B_commit':True})
    mix=pd.read_csv(inp/f'{prefix}_mixture_{suffix}.csv')
    target=pd.read_csv(inp/f'{prefix}_pile_loss_{suffix}.csv')
    assert np.array_equal(mix['index'],target['index'])
    P=mix.iloc[:,1:].to_numpy(float);P=P/P.sum(axis=1,keepdims=True)
    y=target.iloc[:,1:].to_numpy(float).mean(axis=1)
    for name,key,quad in [('B_linear','coef_lin',False),('B_quadratic','coef_quad',True)]:
        X=np.hstack([P,17*np.column_stack([P[:,i]*P[:,j] for i,j in combinations(range(17),2)])]) if quad else P
        pred=(X@coef[key].T).mean(axis=1)
        vals={'n':len(y),'rmse':float(np.sqrt(np.mean((pred-y)**2))),
          'r2':float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
          'spearman':float(spearmanr(y,pred).statistic),
          'top1_regret':float(y[np.argmin(pred)]-y.min()),'random_expected_regret':float(y.mean()-y.min())}
        label='二次混料' if quad else '线性混料'
        saved=recorded[(recorded.split==split)&(recorded.model==label)&(recorded.domain=='__综合(eval)__')]
        assert any(np.isclose(saved.rmse,vals['rmse'],rtol=1e-5,atol=1e-6))
        rr=regrets[(regrets.split==split)&(regrets.model==label)].iloc[0]
        assert np.isclose(rr.regret_top1,vals['top1_regret'],atol=1e-6)
        combined.append({'line':'B','dataset':dataset,'variant':name,**vals})
pd.DataFrame(combined).to_csv(R/'q1_three_lines_verified.csv',index=False)
summary['B_Q1_original_csv_hashes']=raw_inputs
summary['B_Q1_saved_coefficient_metrics_verified']=10
(R/'new_branch_verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print('B saved coefficients independently reproduce 10 held-out/extrapolation summaries. B Q3 unit mismatch confirmed.')
