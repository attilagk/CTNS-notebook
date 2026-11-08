from pathlib import Path
import json
allfits={'kidney_gamma':idata_kidney,'kidney_weibull':idata_kidney_wb,**idata_train1,**{'direct_'+k:v for k,v in idata_1.items()}}
def scalar_max(x):
    a=np.concatenate([np.asarray(v).ravel() for v in x.data_vars.values()]);a=a[np.isfinite(a)];return float(a.max()) if len(a) else None
def scalar_min(x):
    a=np.concatenate([np.asarray(v).ravel() for v in x.data_vars.values()]);a=a[np.isfinite(a)];return float(a.min()) if len(a) else None
diagnostics={}
for name,idata in allfits.items():
    if idata is None:diagnostics[name]={'fit':'caught SamplingError returned None'};continue
    diagnostics[name]={'divergences':int(idata.sample_stats.diverging.sum()),'max_finite_rhat':scalar_max(az.rhat(idata)),'min_finite_bulk_ess':scalar_min(az.ess(idata,method='bulk')),'min_finite_tail_ess':scalar_min(az.ess(idata,method='tail')),'finite_log_likelihood':all(bool(np.isfinite(v).all()) for v in idata.log_likelihood.data_vars.values()) if hasattr(idata,'log_likelihood') else None}
Path('diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
