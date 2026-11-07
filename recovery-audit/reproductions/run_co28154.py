#!/usr/bin/env python3
"""Fresh-kernel CO28154 reproduction. --resume RUN reuses isolated fit caches."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,sys,time,traceback,os
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[2];NAME='2024-06-04-5xfad-behavior-CO28154';SOURCE=ROOT/NAME/(NAME+'.ipynb')
RUN=Path(sys.argv[2]).resolve() if len(sys.argv)>2 and sys.argv[1]=='--resume' else Path(__file__).parent/('co28154-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
RUN.mkdir(exist_ok=True);(RUN/'fits').mkdir(exist_ok=True);(RUN/'figures').mkdir(exist_ok=True)
(RUN/'.gitignore').write_text('fits/\nworkspace/\nexecuted.ipynb\nsource.ipynb\n')
W=RUN/'workspace';(W/'notebooks'/NAME).mkdir(parents=True,exist_ok=True);(W/'results'/NAME).mkdir(parents=True,exist_ok=True)
if not (W/'resources').exists():(W/'resources').symlink_to(ROOT.parent/'resources',target_is_directory=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[ROOT.parent/'resources/QPS/5xFAD-mouse'/f'{s}-behavioral-data-cleaned.xlsx' for s in ['CO26990','CO28152','CO28154']]
m=dict(started_at=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),source_sha256=sha(SOURCE),inputs={str(p):sha(p) for p in inputs},python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),target_sha256=sha(ROOT/'recovery-audit/target-artifacts/target-artifacts.md'),helper_hashes={str(p):sha(p) for p in [ROOT/'src/mwm.py',ROOT/'src/cellbayesassay.py',Path('/home/attila/src/attila_utils.py')]},execution_modifications='Bootstrap adds import paths, redirects figures, and persists/reuses identical model fits; isolated working tree redirects relative results paths. Notebook science cells unchanged.')
if (RUN/'manifest.json').exists():
 previous=json.loads((RUN/'manifest.json').read_text())
 for key in ['source_sha256','inputs','helper_hashes']:
  if previous[key]!=m[key]:raise RuntimeError(f'Cannot resume: {key} changed; start a new run')
(RUN/'source.ipynb').write_bytes(SOURCE.read_bytes());(RUN/'conda-packages.json').write_bytes(subprocess.check_output(['conda','list','-n','bayes','--json']))
n=nbformat.read(SOURCE,as_version=4)
boot=f'''from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[{str(ROOT/'src')!r},'/home/attila/src']
import attila_utils,bambi as bmb,arviz as az,numpy as np,pandas as pd
run_dir=Path({str(RUN)!r})
_original_savefig=attila_utils.savefig
def _savefig(fig,basename,dirpath='named-figure/',formats=['png','pdf']):
    paths=_original_savefig(fig,basename,str(run_dir/'figures')+'/',formats)
    return paths
attila_utils.savefig=_savefig
_original_fit=bmb.Model.fit
def _fit(self,*args,**kwargs):
    key=hashlib.sha256(pd.util.hash_pandas_object(self.data,index=True).values.tobytes()+str(self.data.dtypes).encode()+str(self.formula).encode()+str(self.family).encode()+repr(args).encode()+repr(kwargs).encode()).hexdigest()
    path=run_dir/'fits'/(key+'.nc')
    start=time.monotonic()
    reused=path.exists()
    if reused:result=az.from_netcdf(path)
    else:
        result=_original_fit(self,*args,**kwargs)
        result.to_netcdf(path)
    rh=az.rhat(result).to_array();ess=az.ess(result).to_array()
    report={{'key':key,'formula':str(self.formula),'rows':len(self.data),'seconds':time.monotonic()-start,'reused':reused,'divergences':int(result.sample_stats.diverging.sum()),'max_rhat':float(rh.max()),'min_ess_bulk':float(ess.min())}}
    with (run_dir/'fit-diagnostics.jsonl').open('a') as f:f.write(json.dumps(report)+'\\n')
    return result
bmb.Model.fit=_fit
'''
n.cells.insert(0,nbformat.v4.new_code_cell(boot))
n.cells.append(nbformat.v4.new_code_cell("mwm.get_diagnostics(idatas_CO28152, az.ess).data.to_csv(run_dir/'target-1-ess.csv')\nBF_all.to_csv(run_dir/'target-3-BF_all.csv')\nget_summary(idatas_CO28154,treatment=False).to_csv(run_dir/'target-5-summary.csv')\nget_summary(idatas_CO28154,treatment=True).to_csv(run_dir/'treatment-summary.csv')\n"))
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
def progress(cell,cell_index,**kwargs):print('CELL',cell_index,flush=True)
client=NotebookClient(n,km=km,timeout=3600,resources={'metadata':{'path':str(W/'notebooks'/NAME)}},on_cell_start=progress)
start=time.monotonic();print('RUN_DIRECTORY',RUN,flush=True)
try:client.execute();m['execution_status']='completed'
except Exception:m['execution_status']='failed';m['error']=traceback.format_exc();print(m['error'],flush=True)
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 m.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
 nbformat.write(n,RUN/'executed.ipynb');(RUN/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('RESULT',m['execution_status'],RUN,flush=True)
if m['execution_status']!='completed':sys.exit(1)
