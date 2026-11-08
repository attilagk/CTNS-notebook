import os,sys,json,time,hashlib,traceback,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2];n='2023-10-13-5xfad-maze';work=R/'work/repo'/n
os.environ['PYTHONPATH']=str(root/'src')+':/home/attila/src';os.environ['MPLCONFIGDIR']=str(R/'mplconfig');os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
b=nbformat.read(work/(n+'.ipynb'),as_version=4)
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
status={'environment':'bayes','python':sys.executable,'acceptance':'execution-only','cache_only':'--cache-only' in sys.argv,'reference_agreement':'not assessed','started_at':time.time(), 'source_sha256':hashlib.sha256((R/'source.ipynb').read_bytes()).hexdigest(), 'executed_source_sha256':hashlib.sha256((work/(n+'.ipynb')).read_bytes()).hexdigest()}
status['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
status['git_dirty']=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root,text=True)
status['git_branch']=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip()
def progress(cell,cell_index,**kw):print('CELL',cell_index,flush=True)
try:
 client=NotebookClient(b,km=km,timeout=None,resources={'metadata':{'path':str(work)}},on_cell_start=progress)
 with client.setup_kernel():
  client.reset_execution_trackers()
  client.code_cells_executed=0
  for index,cell in enumerate(b.cells):
   client.execute_cell(cell,index)
   if index==0 and '--cache-only' in sys.argv:
    guard=nbformat.v4.new_code_cell("import functools\ndef _no_sampling(*args,**kwargs): raise RuntimeError('Cache verification attempted sampling')\n_original_sample=pm.sample\npm.sample=functools.wraps(_original_sample)(_no_sampling)\nbmb.Model.fit=_no_sampling\n")
    b.cells.append(guard);client.execute_cell(guard,len(b.cells)-1);b.cells.pop()

  check=nbformat.v4.new_code_cell((R/'diagnostics-code.py').read_text())
  b.cells.append(check);client.execute_cell(check,len(b.cells)-1);b.cells.pop()
 status['execution']='passed'
except Exception as e:status['execution']='failed';status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:
 nbformat.write(b,R/'executed.ipynb');status['finished_at']=time.time();(R/'manifest.json').write_text(json.dumps(status,indent=2));
 if km.has_kernel:km.shutdown_kernel(now=True)
