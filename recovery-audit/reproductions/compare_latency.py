#!/usr/bin/env python3
"""Compare a completed latency reproduction with the user-specified targets."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np,arviz as az
R=Path(sys.argv[1]);reference=pd.DataFrame(json.loads((R/'reference-day-summary.json').read_text())).set_index('experiment');actual=pd.read_csv(R/'day_summary.csv',index_col=0)
comparison=pd.DataFrame(index=reference.index)
for c in ['mean','sd','hdi_3%','hdi_97%','mcse_mean','ess_bulk','ess_tail','r_hat']:
 comparison[c+'_reference']=reference[c];comparison[c+'_reproduced']=actual[c];comparison[c+'_delta']=actual[c]-reference[c]
comparison.to_csv(R/'day-summary-comparison.csv')
idata=az.from_netcdf(R/'fits/fit-02.nc');v='C(Condition, levels=lvl)';a=idata.posterior[v];dim=next(d for d in a.dims if d not in ['chain','draw'])
figure=[]
for i,label in enumerate(a[dim].values):
 vals=a.isel({dim:i}).values.ravel();interval=az.hdi(vals,hdi_prob=.95)
 figure.append(dict(parameter=str(label),mean=float(vals.mean()),hdi_low=float(interval[0]),hdi_high=float(interval[1]),rope_percent=float(100*np.mean((vals>=-.1)&(vals<=1)))))
(R/'figure-posterior-values.json').write_text(json.dumps(figure,indent=2)+'\n')
fits=[json.loads(l) for l in (R/'fit-diagnostics.jsonl').read_text().splitlines()]
manifest=json.loads((R/'manifest.json').read_text())
lines=['# Latency notebook reproduction','',f"Execution: **{manifest['execution_status']}**. Elapsed: {manifest['elapsed_seconds']:.1f} seconds.",'','Fresh bayes kernel; original notebook cells executed unchanged. Figure writes were redirected to this run directory. Each completed fit was immediately saved as NetCDF. No original figures or notebook outputs were overwritten.','', '## Target 1: day_summary','', 'Numerical differences are reported for review; no acceptance tolerance has been approved. The original MCSEs are rounded, including values printed as zero. ESS and R-hat need not reproduce exactly.','', '| Experiment | Reference mean | New mean | Difference | Reference SD | New SD |','|---|---:|---:|---:|---:|---:|']
for name in reference.index:lines.append(f"| {name} | {reference.loc[name,'mean']:.3f} | {actual.loc[name,'mean']:.3f} | {actual.loc[name,'mean']-reference.loc[name,'mean']:+.3f} | {reference.loc[name,'sd']:.3f} | {actual.loc[name,'sd']:.3f} |")
lines+=['','Full numerical comparison: day-summary-comparison.csv.','', '## Target 2: condition-posteriors-study1.png','', 'Generated PNG and PDF are in figures/. The reference screenshot is Pasted image 20261106213159.png in target-artifacts. Figure posterior values below use the same 95% HDI and ROPE [-0.1, 1] as the plotting cell.','']
for f in figure:lines.append(f"- {f['parameter']}: mean {f['mean']:.4f}; 95% HDI [{f['hdi_low']:.4f}, {f['hdi_high']:.4f}]; ROPE {f['rope_percent']:.2f}%.")
lines+=['','## Sampling diagnostics','',f"{len(fits)} fits saved; total divergences: {sum(f['divergences'] for f in fits)}. Largest reported R-hat: {max(f['max_rhat'] for f in fits):.2f}; smallest reported bulk ESS: {min(f['min_ess_bulk'] for f in fits):.0f}. These use rounded ArviZ summaries; inspect fit-diagnostics.jsonl and full saved fits for details.",'','## Provenance and reuse','','manifest.json records source/input hashes, Git revision and timing; conda-packages.json records installed packages. source.ipynb preserves the input notebook. executed.ipynb contains fresh outputs. The run harness saves results but does not yet automatically reuse caches; reuse must check input/code/environment compatibility. Figures may differ in fonts, rendering and Monte Carlo details. A scientific reproduction pass requires review against the target artifacts.']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
