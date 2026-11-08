import sys,os,json,hashlib,subprocess,datetime,traceback
from pathlib import Path
from importlib.metadata import version
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(sys.argv[1]); name=sys.argv[2]; phase=sys.argv[3]
run=root/'recovery-audit/reproductions'/('rerun-'+name+'-20261108')
run.mkdir(parents=True,exist_ok=True)
(run/'.gitignore').write_text('work/\nmplconfig/\nsource.ipynb\nexecuted.ipynb\nverification.ipynb\n')
source=root/name/(name+'.ipynb'); nb=nbformat.read(source,4)
nb.cells=[c for c in nb.cells if not (c.cell_type=='code' and '%connect_info' in c.source)]
for c in nb.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
work=run/'work'/phase; cwd=work/'repo'/name;cwd.mkdir(parents=True,exist_ok=True)
(work/'resources').symlink_to('/home/attila/projects/CTNS/resources',target_is_directory=True) if not (work/'resources').exists() else None
(work/'results'/name).mkdir(parents=True,exist_ok=True)
(cwd/'named-figure').mkdir(exist_ok=True)
os.environ.update(PYTHONPATH=str(root/'src')+':/home/attila/src',MPLCONFIGDIR=str(run/'mplconfig'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputp=Path('/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28152_NFL-GFAP_datasum.xlsx')
manifest=dict(notebook=name,phase=phase,environment='base',acceptance='all analysis cells complete without unhandled error; no reference comparisons',reference_reproduction='not assessed',inference_quality='not assessed; no MCMC',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(source),input_sha256={str(inputp):sha(inputp)},helpers={'attila_utils.py':sha('/home/attila/src/attila_utils.py')},git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),git_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),interpreter=sys.executable,versions={x:version(x) for x in ['pandas','numpy','matplotlib','seaborn','statsmodels','openpyxl','nbclient','nbformat']},removed_cells=['%connect_info'])
subprocess.run(['/home/attila/miniconda3/bin/conda','list','--json'],stdout=open(run/'conda-packages.json','w'),check=True)
# Kernel interpreter is fixed explicitly; no kernelspec inference.
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(cwd)}},on_cell_start=lambda cell,cell_index:print('CELL',cell_index,flush=True))
try:
 client.execute();manifest['execution']='passed'
except Exception:
 manifest['execution']='failed';manifest['error']=traceback.format_exc();traceback.print_exc()
finally:
 if km.has_kernel:km.shutdown_kernel(now=True)
 nbformat.write(nb,run/('executed.ipynb' if phase=='reproduction' else 'verification.ipynb'))
 manifest['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 manifest['warnings']=[o.get('text','') for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='stream' and o.name=='stderr']
 manifest['code_cells']=sum(c.cell_type=='code' for c in nb.cells)
 manifest['artifacts']={str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and 'resources' not in p.relative_to(work).parts}
 (run/(phase+'-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['execution'],flush=True)
if manifest['execution']!='passed':sys.exit(1)
