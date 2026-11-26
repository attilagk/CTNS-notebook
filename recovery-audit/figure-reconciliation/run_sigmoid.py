#!/usr/bin/env python3
"""Focused fresh-kernel reproduction of sigmoid-4 figure, original scientific cells intact."""
import os,sys,json,hashlib,shutil,subprocess,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];name='2023-09-26-cell-bayes-assays'
RUN=ROOT/'recovery-audit/reproductions'/('figure-recovery-'+name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));RUN.mkdir(parents=True)
(RUN/'.gitignore').write_text('work/\n*.ipynb\nfits/\nreferences/\nmplconfig/\n')
cwd=RUN/'work/notebooks'/name;cwd.mkdir(parents=True);res=RUN/'work/resources/cell-based-assays';res.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={}
for fn in ['TI21_22_26_rawdata_sum-edited.xlsx','TI21_22_26_rawdata_sum-controls.csv']:
 p=Path('/home/attila/CTNS~/resources/cell-based-assays')/fn;shutil.copy2(p,res/fn);inputs[str(p)]=sha(p)
(RUN/'references').mkdir();refroot=Path('/home/attila/CTNS~/notebooks')/name/'named-figure'
for ext in ['png','pdf']:shutil.copy2(refroot/('prior-posterior-curves-sigmoid-4.'+ext),RUN/'references')
os.environ.update(PYTHONPATH=str(ROOT/'src')+':/home/attila/src',MPLCONFIGDIR=str(RUN/'mplconfig'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
source=ROOT/name/(name+'.ipynb');nb=nbformat.read(source,4);selected=[1,2,5,6,21,22,23,32,33]
nb.cells=[nb.cells[i-1] for i in selected]
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
# The sigmoid-2 comparison is not a dependency of sigmoid-4. Its original
# HalfStudentT(sigma=20,nu=10) prior currently evaluates to NaN in bayes.
# Preserve the failed unmodified attempt; execute only the requested model.
for c in nb.cells:
 if 'model_2, idata_2 =' in c.source:
  c.source='\n'.join(line for line in c.source.splitlines() if not line.startswith('model_2, idata_2 ='))
 if c.source.startswith('def prior_posterior_curves'):
  c.source=c.source.split('\nfig, ax = prior_posterior_curves(model_2')[0]
(RUN/'fits').mkdir()
persist=nbformat.v4.new_code_cell(f"from pathlib import Path\n_fit_dir=Path({str(RUN/'fits')!r})\nidata_2.to_netcdf(_fit_dir/'sigmoid-2.nc')\nidata_4.to_netcdf(_fit_dir/'sigmoid-4.nc')\nimport json\n_report={{}}\nfor _name,_idata in [('sigmoid-2',idata_2),('sigmoid-4',idata_4)]:\n    _summary=az.summary(_idata)\n    _summary.to_csv(_fit_dir/(_name+'-summary.csv'))\n    _rhat=az.rhat(_idata).to_array().values\n    _ess=az.ess(_idata).to_array().values\n    _report[_name]={{'divergences':int(_idata.sample_stats.diverging.sum()),'max_rhat':float(np.nanmax(_rhat)),'min_ess':float(np.nanmin(_ess)),'log_likelihood_finite':bool(np.isfinite(_idata.log_likelihood.to_array()).all())}}\n(_fit_dir/'diagnostics.json').write_text(json.dumps(_report,indent=2))\n")
nb.cells.insert(7,persist)
persist.source=persist.source.replace("idata_2.to_netcdf(_fit_dir/'sigmoid-2.nc')\n",'').replace("[('sigmoid-2',idata_2),('sigmoid-4',idata_4)]","[('sigmoid-4',idata_4)]")
manifest=dict(notebook=name,selected_cells_one_based=selected,scope='focused sigmoid figures only; not a full notebook rerun',environment='bayes',environment_resolution='no target declaration; dependency probes pass; pandas 2 supports original applymap; original scientific cells unchanged',python=sys.executable,source_sha256=sha(source),inputs=inputs,helpers={str(p):sha(p) for p in [ROOT/'src/cellbayesassay.py',Path('/home/attila/src/attila_utils.py')]},reference_sha256={str(p):sha(p) for p in (RUN/'references').iterdir()},started_at=datetime.now(timezone.utc).isoformat(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),git_dirty=subprocess.check_output(['git','status','--porcelain'],text=True),packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
(RUN/'source.ipynb').write_bytes(source.read_bytes());(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
try:
 NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(cwd)}},on_cell_start=lambda cell,cell_index,**kw:print('CELL',cell_index,flush=True)).execute();manifest['execution']='passed'
except Exception:
 manifest['execution']='failed';manifest['error']=traceback.format_exc();print(manifest['error'],flush=True)
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 nbformat.write(nb,RUN/'executed.ipynb');manifest['finished_at']=datetime.now(timezone.utc).isoformat();manifest['artifacts']={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.suffix in ['.png','.pdf','.nc'] and 'references' not in p.parts};manifest['warnings']=[o.get('text','') for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='stream' and o.name=='stderr'];(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('RUN',RUN,'STATUS',manifest['execution'],flush=True)
if manifest['execution']!='passed':sys.exit(1)
