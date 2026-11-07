#!/usr/bin/env python3
"""Report scientific target comparisons without tuning models to reference numbers."""
from pathlib import Path
import sys,json,re,hashlib
import pandas as pd,numpy as np
R=Path(sys.argv[1]);ROOT=Path(__file__).resolve().parents[2]
s=(ROOT/'recovery-audit/target-artifacts/target-artifacts.md').read_text().split('## 2024-06-04-5xfad-behavior-CO28154')[1].split('\n## ')[0]
block=s.split('### Artifact 5')[1].split('```')[1];rows=[];group=''
cols=['mean','sd','hdi_2.5%','hdi_97.5%','mcse_mean','mcse_sd','ess_bulk','ess_tail','r_hat','mean %change','hdi_2.5% %change','hdi_97.5% %change']
for line in block.strip().splitlines()[1:]:
 t=line.split()
 try:vals=list(map(float,t[-12:]))
 except ValueError:continue
 label=' '.join(t[:-12]);param=next((p for p in ['Intercept','Day','censored(Latency, Status)_alpha','1|IRN_sigma'] if label.endswith(p)),None)
 if not param:continue
 prefix=label[:-len(param)].strip()
 if prefix:group=prefix
 rows.append(dict(experiment=group,parameter=param,**dict(zip(cols,vals))))
ref=pd.DataFrame(rows).set_index(['experiment','parameter']);actual=pd.read_csv(R/'target-5-summary.csv',index_col=[0,1]);actual.index.names=ref.index.names;actual=actual.rename(index={'alpha':'censored(Latency, Status)_alpha'},level=1);comp=pd.DataFrame(index=ref.index)
for c in cols:
 comp[c+'_reference']=ref[c];comp[c+'_new']=actual[c];comp[c+'_delta']=actual[c]-ref[c]
comp.to_csv(R/'target-5-comparison.csv')
# Extract all available numeric treatment means from Artifact 3 (truncated reference).
b=pd.read_csv(R/'target-3-BF_all.csv',index_col=[0,1],header=[0,1]);treat=pd.read_csv(R/'treatment-summary.csv',index_col=[0,1]);means=[];experiment=''
for line in s.split('### Artifact 3')[1].split('```')[1].splitlines():
 tokens=line.split()
 if '...' not in tokens:continue
 ell=tokens.index('...')
 try:mean=float(tokens[ell-1]); reference_bf=list(map(float,tokens[ell-10:ell-1]))
 except (ValueError,IndexError):continue
 prefix=' '.join(tokens[:ell-10])
 for exp in ['Amiloride 10 TG','CO28152 TG','TUDCA + HCQ TG','Arundine TG']:
  if prefix.startswith(exp+' '):experiment=exp;prefix=prefix[len(exp):].strip();break
 key=(experiment,prefix)
 if key in b.index:
  value=float(b.loc[key,('log change of escape latency','mean')]);means.append({'experiment':experiment,'treatment':prefix,'reference_mean':mean,'new_mean':value,'delta':value-mean,**{hyp+' '+stat+' reference':reference_bf[j*3+k] for j,hyp in enumerate(['full rescue','partial rescue','slight improvement']) for k,stat in enumerate(['2 log BF','t1','t0'])},**{hyp+' '+stat+' new':float(b.loc[key,(hyp,stat)]) for hyp in ['full rescue','partial rescue','slight improvement'] for stat in ['2 log BF','t1','t0']}})
pd.DataFrame(means).to_csv(R/'target-3-mean-comparison.csv',index=False)
lines=['# CO28154 behavioral notebook reproduction','','## Execution','', 'Fresh bayes kernel, with isolated working directory and figures. Existing external outputs were not loaded or overwritten. See manifest.json and conda-packages.json for inputs, code hashes and runtime. Saved posterior fits are local, ignored NetCDF files.','','## Artifact 5: non-treatment posterior summaries','','| Experiment | Parameter | Reference mean | New mean | Difference |','|---|---|---:|---:|---:|']
for (exp,param),row in comp.iterrows():lines.append(f"| {exp} | {param} | {row['mean_reference']:.3f} | {row['mean_new']:.3f} | {row['mean_delta']:+.3f} |")
lines+=['','Full comparison (including intervals, SD, diagnostics and derived columns): target-5-comparison.csv. The current Bambi parameter alpha is matched to the historical censored(Latency, Status)_alpha label for comparison only. ESS values are stochastic diagnostics, not exact reproducibility targets. No acceptance tolerance was supplied; the table reports observed differences.','','## Artifact 3: BF_all posterior means','','| Experiment | Treatment | Reference mean | New mean | Difference |','|---|---|---:|---:|---:|']
for row in means:lines.append(f"| {row['experiment']} | {row['treatment']} | {row['reference_mean']:.3f} | {row['new_mean']:.3f} | {row['delta']:+.3f} |")
lines+=['','The reference BF_all display omits columns and truncates its final row. target-3-BF_all.csv preserves the complete new result. Tail-count-based values labeled Bayes factors can vary substantially even when posterior means agree. This run preserves the existing pseudocount and computation; it does not validate the scientific interpretation of those values.','','## Artifacts 1, 2 and 4','','Artifact 1 is exported as target-1-ess.csv. Figures are figures/model-latency-CO28154-TUDCA_HCQ-TG.png and figures/CI-BF-all.png. Compare them with the screenshots identified in target-section.md.','','## Known experiment-definition concerns','','The CO26990 two-level WT entries use CO28152 data; the corresponding male-only WT entry also uses the full CO28152 dataset. The CO28152 three-level entries named TUDCA WT select TUDCA TG. These pre-existing definitions do not supply the requested target artifacts and were not changed silently. They require scientific review.','','## Reuse','','The isolated notebook cache uses fpaths.csv under workspace/results. Model-fit caches are also saved immediately under fits. Source and package versions must be checked before reusing a run. Raw reference cells remain unexecuted, matching their original notebook type.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Target 5 maximum mean difference',comp['mean_delta'].abs().max());print('Target 3 compared means',len(means))
