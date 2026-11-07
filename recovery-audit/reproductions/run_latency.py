#!/usr/bin/env python3
"""Execute the latency notebook unchanged, with isolated figures and saved fits.
Run with the bayes environment's Python. Output is an independent run directory.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time,traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'2023-11-07-5xfad-maze-latency/2023-11-07-5xfad-maze-latency.ipynb'
RUN=Path(__file__).parent/('latency-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));RUN.mkdir();(RUN/'fits').mkdir();(RUN/'figures').mkdir()
(RUN/'.gitignore').write_text('fits/\nexecuted.ipynb\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[ROOT.parent/'results/2023-10-13-5xfad-maze'/f'data_train-{s}.csv' for s in ['CO26990','CO28152']]
manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),source=str(SOURCE),source_sha256=sha(SOURCE),python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inputs={str(p):sha(p) for p in inputs},target_sha256=sha(ROOT/'recovery-audit/target-artifacts/target-artifacts.md'),modifications='Added bootstrap for isolated figure paths and fit persistence; original cells execute unchanged.')
(RUN/'source.ipynb').write_bytes(SOURCE.read_bytes())
(RUN/'conda-packages.json').write_bytes(subprocess.check_output(['conda','list','-n','bayes','--json']))
n=nbformat.read(SOURCE,as_version=4)
bootstrap=f'''from pathlib import Path
import sys,json,time,hashlib
sys.path[:0]=[{str(ROOT/'src')!r}, '/home/attila/src']
import attila_utils,bambi as bmb,arviz as az,numpy as np,pandas as pd
run_dir=Path({str(RUN)!r})
_original_savefig=attila_utils.savefig
def _isolated_savefig(fig, basename, dirpath='named-figure/', formats=['png','pdf']):
    return _original_savefig(fig, basename, str(run_dir/'figures')+'/', formats)
attila_utils.savefig=_isolated_savefig
_original_fit=bmb.Model.fit
_fit_counter=0
def _persisted_fit(self,*args,**kwargs):
    global _fit_counter
    _fit_counter+=1
    start=time.monotonic()
    result=_original_fit(self,*args,**kwargs)
    result.to_netcdf(run_dir/'fits'/f'fit-{{_fit_counter:02}}.nc')
    summary=az.summary(result)
    report={{'fit':_fit_counter,'formula':str(self.formula),'seconds':time.monotonic()-start,'divergences':int(result.sample_stats.diverging.sum()),'max_rhat':float(summary.r_hat.max()),'min_ess_bulk':float(summary.ess_bulk.min())}}
    with (run_dir/'fit-diagnostics.jsonl').open('a') as f:f.write(json.dumps(report)+'\\n')
    return result
bmb.Model.fit=_persisted_fit
'''
n.cells.insert(0,nbformat.v4.new_code_cell(bootstrap))
n.cells.append(nbformat.v4.new_code_cell("day_summary.to_csv(run_dir/'day_summary.csv')\ncondition0_summary.to_csv(run_dir/'condition0_summary.csv')\ncondition1_summary.to_csv(run_dir/'condition1_summary.csv')\n"))
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
def progress(cell,cell_index,**kwargs):print('Executing cell',cell_index,flush=True)
client=NotebookClient(n,km=km,timeout=1800,resources={'metadata':{'path':str(SOURCE.parent)}},on_cell_start=progress)
start=time.monotonic();print('RUN_DIRECTORY',RUN,flush=True)
try:
 client.execute();manifest['execution_status']='completed'
except Exception:
 manifest['execution_status']='failed';manifest['error']=traceback.format_exc();print(manifest['error'],flush=True)
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
 nbformat.write(n,RUN/'executed.ipynb');(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('RESULT',manifest['execution_status'],RUN,flush=True)
if manifest['execution_status']!='completed':sys.exit(1)
