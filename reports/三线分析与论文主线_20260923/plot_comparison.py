from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
d=pd.read_csv(R/'q1_three_lines_verified.csv')
sets=['same_scale_1m','cross_scale_60m','cross_scale_1B','est_10b','est_70b']
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for variant,label,col in [('ridge_A','A: ridge','#526D82'),('B_quadratic','B: quadratic','#CF843A'),('forest_none','C: forest (no Q)','#2A877D')]:
 x=d[d.variant==variant].set_index('dataset').loc[sets]
 axes[0].plot(range(3),x.spearman.iloc[:3],'o-',label=label,color=col)
 axes[0].plot(range(2,5),x.spearman.iloc[2:],'o--',color=col)
 axes[1].plot(range(3),(x.top1_regret/x.random_expected_regret).iloc[:3],'o-',label=label,color=col)
for ax in axes:
 ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].axvspan(2.5,4.4,color='#F1E1D5',alpha=.65)
axes[0].axhline(0,color='#999',lw=.8);axes[0].set_xticks(range(5),['1M','60M','1B','10B*','70B*'])
axes[0].set_ylabel('Spearman correlation');axes[0].set_title('Ranking: observed vs estimated (*)');axes[0].legend(fontsize=9)
axes[1].axhline(1,color='#999',ls='--',label='Random-pick expectation')
axes[1].set_xticks(range(3),['1M','60M','1B']);axes[1].set_ylabel('Top-1 regret / random expected regret');axes[1].set_title('Observed candidate-selection performance');axes[1].legend(fontsize=9)
fig.savefig(R/'q1_comparison.png',dpi=220);fig.savefig(R/'q1_comparison.svg')
