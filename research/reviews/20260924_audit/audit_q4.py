"""Read-only Q4 audit; writes only beside this script, never to upstream results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UP = ROOT.parent / 'F题-upstream/line-b'
DATA = ROOT / 'real_attachments/C_efficiency_evolution'
raw = pd.read_csv(DATA / 'leaderboard_cleaned.csv')
raw['date'] = pd.to_datetime(raw['Submission Date'], errors='coerce')
t0 = pd.Timestamp('2025-03-13')
d = raw.dropna(subset=['date', '#Params (B)', 'Average ⬆️']).copy()
d = d[d.date <= t0]
d['t'] = (d.date-t0).dt.days/365.25
d['logN'] = np.log10(d['#Params (B)'].clip(lower=1e-3))
y = d['Average ⬆️'].to_numpy(float)
old_chat = d.Type.str.contains('chat', case=False, na=False).astype(float)
post = d.Type.str.contains('chat|fine-tuned', case=False, na=False).astype(float)
categories = pd.get_dummies(d.Type, drop_first=True, dtype=float)
base = np.column_stack([np.ones(len(d)), d.logN, d.t])
g = .628031
wN = .27987813/(.33997658+.27987813)
s0 = float(np.percentile(d.loc[d['Hub License'].notna(), 'Average ⬆️'], 90))
rows = []
for name, extra in [('original_chat_only',old_chat),('chat_or_finetuned_binary_sensitivity',post),('all_type_categories_sensitivity',categories)]:
    X = np.column_stack([base, extra])
    beta = np.linalg.solve(X.T@X + 1e-8*np.eye(X.shape[1]), X.T@y)
    row = dict(model=name,n=len(d),b_logN=beta[1],b_t=beta[2],rmse=np.sqrt(np.mean((X@beta-y)**2)),S0=s0)
    for months in [12,24]:
        row[f'forecast_{months}m_continued'] = s0+(beta[1]*wN*g+beta[2])*months/12
        row[f'forecast_{months}m_slowdown'] = s0+(beta[1]*wN*g*.5+beta[2])*months/12
    rows.append(row)
pd.DataFrame(rows).to_csv(HERE/'q4_type_sensitivity.csv',index=False)
summary = dict(n_valid=len(d),raw_type_counts=raw.Type.value_counts().to_dict(),
    wrongly_called_pretrained_finetuned=int(raw.Type.str.contains('fine-tuned',case=False,na=False).sum()),
    original_false_chat_count=int((~raw.Type.str.contains('chat',case=False,na=False)).sum()),
    all_period_open_p90=s0,
    latest_quarter_open_p90=float(np.percentile(d.loc[(d.date.dt.year==2025)&d['Hub License'].notna(),'Average ⬆️'],90)),
    time_range=[str(d.date.min().date()),str(d.date.max().date())],
    note='Type recoding is sensitivity evidence, not a validated replacement forecast.')
(HERE/'q4_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
