import os,sys,json,hashlib,subprocess,shutil,csv,traceback
from pathlib import Path
from datetime import datetime,timezone
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(sys.argv[1]); name=sys.argv[2]; mode=sys.argv[3]
run=root/'recovery-audit/reproductions'/('rerun-'+name+'-20261108')
run.mkdir(parents=True,exist_ok=True)
(run/'.gitignore').write_text('work/\nmplconfig/\nsource.ipynb\nexecuted.ipynb\n')
work=run/'work'; repo=work/'repo'; cwd=repo/name; cwd.mkdir(parents=True,exist_ok=True)
resources=work/'resources'
if not resources.exists():resources.symlink_to('/home/attila/projects/CTNS/resources',target_is_directory=True)
(work/'results'/name).mkdir(parents=True,exist_ok=True)
(cwd/'named-figure').mkdir(exist_ok=True)
source=root/name/(name+'.ipynb')
nb=nbformat.read(source,as_version=4)
nb.cells=[c for c in nb.cells if '%connect_info' not in c.get('source','')]
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
copy=cwd/source.name;nbformat.write(nb,copy)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={r['resolved_path']:sha(r['resolved_path']) for r in csv.DictReader(open(root/'recovery-audit/input-inventory.csv')) if r['notebook'].split('/')[0]==name and r['role']=='source data / resource' and Path(r['resolved_path']).is_file()}
manifest=dict(notebook=name,acceptance='execution-only; no target comparisons requested',environment='base',python=sys.executable,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),git_status=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),source_sha256=sha(source),inputs=inputs,helpers={str(p.relative_to(root)):sha(p) for p in (root/'src').glob('*.py')},started_at=datetime.now(timezone.utc).isoformat(),mode=mode)
manifest['packages']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
os.environ['PYTHONPATH']=str(root/'src')+':/home/attila/src';os.environ['MPLCONFIGDIR']=str(run/'mplconfig');os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
try:
 client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(cwd)}},on_cell_start=lambda cell,cell_index,**k: print('CELL',cell_index,flush=True))
 client.execute(); manifest['execution']='passed'
except Exception:
 manifest['execution']='failed';manifest['error']=traceback.format_exc();print(manifest['error'])
finally:
 nbformat.write(nb,run/'executed.ipynb');manifest['finished_at']=datetime.now(timezone.utc).isoformat();manifest['executed_cells']=sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells);manifest['warnings']=[o.get('text','') for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='stream' and o.name=='stderr'];(run/(mode+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
if manifest['execution']!='passed':sys.exit(1)
if mode=='reproduction':shutil.copy2(run/'executed.ipynb',source)
