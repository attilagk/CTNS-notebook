"""Fresh-kernel isolated NFL reproduction. Run using base Python."""
from pathlib import Path
import json,sys,os,hashlib,subprocess,traceback
from datetime import datetime,timezone
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
ROOT=Path(__file__).resolve().parents[2];NAME='2024-05-20-nfl-modeling'
RUN=ROOT/'recovery-audit/reproductions'/('nfl-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));RUN.mkdir()
(RUN/'.gitignore').write_text('workspace/\nsource.ipynb\nexecuted.ipynb\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/NAME/(NAME+'.ipynb');n=nbformat.read(source,4);(RUN/'source.ipynb').write_bytes(source.read_bytes())
inputs=[ROOT.parent/'resources/QPS/5xFAD-mouse'/x for x in ['CO28152-animal-list.csv','CO28154-animal-list.csv','CO28154-biochemistry-cleaned.xlsx','CO28152_NFL-GFAP_datasum.xlsx']]
modules=['arviz','statsmodels.formula.api','pymc','bambi','pandas','numpy','matplotlib.pyplot','seaborn','scipy.stats','nfl','attila_utils','h5netcdf','h5py']
sys.path[:0]=[str(ROOT/'src'),'/home/attila/src']
import importlib
probes={x:importlib.import_module(x).__file__ for x in modules}
manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),python=sys.executable,branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_paths=subprocess.check_output(['git','status','--porcelain'],text=True),source_sha256=sha(source),inputs={str(p):sha(p) for p in inputs},helpers={str(p):sha(p) for p in [ROOT/'src/nfl.py',ROOT/'src/mcmc_cache.py',Path('/home/attila/src/attila_utils.py')]},target_sha256=sha(ROOT/'recovery-audit/target-artifacts/target-artifacts.md'),dependency_probes=probes,criteria='Four deterministic target figures: compare source data and numeric t-test/OLS outputs at historical displayed precision; visually compare curves, labels and scientific interpretation. Bayesian models exploratory; diagnostics recorded separately.',sampling='model.fit() defaults, no original seed; unchanged')
(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(RUN/'conda-packages.json').write_bytes(subprocess.check_output(['/home/attila/miniconda3/bin/conda','list','-n','base','--json']))
setup='''from pathlib import Path
import sys, hashlib
sys.path[:0] = [str(Path.cwd().parent / 'src'), '/home/attila/src']
import mcmc_cache
force_refit = False
_cache_inputs = [Path('../../resources/QPS/5xFAD-mouse') / x for x in ['CO28152-animal-list.csv', 'CO28154-animal-list.csv', 'CO28154-biochemistry-cleaned.xlsx', 'CO28152_NFL-GFAP_datasum.xlsx']]
import nbformat
_cache_context = {'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _cache_inputs}, 'helpers': {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [nfl, mcmc_cache]}, 'analysis': mcmc_cache.analysis_code_hash(nbformat.read('2024-05-20-nfl-modeling.ipynb', 4).cells)}
_cache_dir = Path('../../results/2024-05-20-nfl-modeling/fit-cache')
'''
n.cells[0].source="from pathlib import Path\nimport sys\nsys.path[:0] = [str(Path.cwd().parent / 'src'), '/home/attila/src']\n"+n.cells[0].source+'\n'+setup
for c in n.cells:
 if c.cell_type=='code':
  c.source=c.source.replace('.applymap(','.map(').replace("results.posterior.variables['Treatment']","results.posterior['Treatment']").replace('results = model.fit()', 'results = mcmc_cache.fit_or_load(model, _cache_dir, settings={}, context=_cache_context, force=force_refit)')
  if '%connect_info' in c.source:c.source='# Connection details intentionally omitted.'
  c.outputs=[];c.execution_count=None
work=RUN/'workspace';cwd=work/'notebooks'/NAME;cwd.mkdir(parents=True);(work/'notebooks/src').symlink_to(ROOT/'src',target_is_directory=True);(work/'resources').symlink_to(ROOT.parent/'resources',target_is_directory=True);(work/'results'/NAME).mkdir(parents=True)
nbformat.write(n,cwd/(NAME+'.ipynb'))
# Additional evidence exports only in execution copy.
n.cells.append(nbformat.v4.new_code_cell(f'''import json
from pathlib import Path
assert sys.executable == {sys.executable!r}
_out=Path({str(RUN)!r})
for _name in ['results_CO28152','results_CO28154a','results_CO28154b']:
    _idata=globals()[_name]
    _summary=az.summary(_idata, round_to='none')
    _summary.to_csv(_out/(_name+'-diagnostics.csv'))
    with (_out/'diagnostics.jsonl').open('a') as _f:
        _f.write(json.dumps({{'model':_name,'divergences':int(_idata.sample_stats.diverging.sum()),'max_rhat':float(_summary.r_hat.max()),'min_ess_bulk':float(_summary.ess_bulk.min()),'min_ess_tail':float(_summary.ess_tail.min())}})+'\\n')
dataw_CO28152.to_csv(_out/'dataw_CO28152.csv')
dataw_CO28154.to_csv(_out/'dataw_CO28154.csv')
'''))
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
print('RUN',RUN,flush=True)
try:
 NotebookClient(n,km=km,timeout=1800,resources={'metadata':{'path':str(cwd)}},on_cell_start=lambda cell,cell_index,**k:print('CELL',cell_index,flush=True)).execute();manifest['execution']='completed'
except Exception:
 manifest['execution']='failed';manifest['error']=traceback.format_exc();print(manifest['error'],flush=True)
finally:
 nbformat.write(n,RUN/'executed.ipynb');manifest['finished_at']=datetime.now(timezone.utc).isoformat();(RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['execution'],flush=True)
