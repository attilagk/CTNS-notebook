#!/usr/bin/env python3
"""Run canonical source in an isolated persistent workspace; never promote automatically."""
import os,sys,json,hashlib,subprocess,shutil,traceback,csv,ast
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2];name=sys.argv[1];env=sys.argv[2]
RUN=ROOT/'recovery-audit/reproductions'/('figure-recovery-'+name+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
RUN.mkdir(parents=True);(RUN/'.gitignore').write_text('work/\n*.ipynb\nmplconfig/\n')
work=RUN/'work';cwd=work/'notebooks'/name;cwd.mkdir(parents=True)
(work/'resources').symlink_to(ROOT.parent/'resources',target_is_directory=True)
(work/'results'/name).mkdir(parents=True)
# Copy persistent caches into isolation; never write back into the original caches.
original=ROOT.parent/'results'/name
if original.exists():
 for p in original.iterdir():
  if p.is_dir() and ('cache' in p.name or p.name.startswith('idata')):shutil.copytree(p,work/'results'/name/p.name)
source=ROOT/name/(name+'.ipynb');shutil.copy2(source,cwd/source.name)
os.environ.update(PYTHONPATH=str(ROOT/'src')+':/home/attila/src',MPLCONFIGDIR=str(RUN/'mplconfig'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
nb=nbformat.read(source,4)
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
nb.cells=[c for c in nb.cells if '%connect_info' not in c.get('source','')]
if '--cache-only' in sys.argv:
 nb.cells.insert(0,nbformat.v4.new_code_cell("import pymc as pm, bambi as bmb, functools\n_original_sample=pm.sample\n@functools.wraps(_original_sample)\ndef _no_sample(*args, **kwargs):\n    raise RuntimeError('Figure recovery needs an unavailable compatible cache; fresh MCMC not started')\npm.sample=_no_sample\nbmb.Model.fit=_no_sample\n"))
inputs={}
for r in csv.DictReader(open(ROOT/'recovery-audit/input-inventory.csv')):
 if r['notebook'].split('/')[0]==name and Path(r['resolved_path']).is_file():inputs[r['resolved_path']]=sha(r['resolved_path'])
manifest=dict(notebook=name,environment=env,python=sys.executable,started_at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(source),inputs=inputs,helpers={str(p):sha(p) for p in [*list((ROOT/'src').glob('*.py')),Path('/home/attila/src/attila_utils.py')]},git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),git_dirty=subprocess.check_output(['git','status','--porcelain'],text=True),purpose='recover missing exported figures; execution distinct from reference agreement',packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
(RUN/'source.ipynb').write_bytes(source.read_bytes())
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
try:
 client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(cwd)}},on_cell_start=lambda cell,cell_index,**kw:print('CELL',cell_index,flush=True))
 client.execute();manifest['execution']='passed'
except Exception:
 manifest['execution']='failed';manifest['error']=traceback.format_exc();print(manifest['error'],flush=True)
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 nbformat.write(nb,RUN/'executed.ipynb')
 manifest['finished_at']=datetime.now(timezone.utc).isoformat();manifest['artifacts']={str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.pdf','.svg']}
 manifest['warnings']=[o.get('text','') for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='stream' and o.name=='stderr']
 (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('RUN',RUN,'STATUS',manifest['execution'],flush=True)
if manifest['execution']!='passed':sys.exit(1)
