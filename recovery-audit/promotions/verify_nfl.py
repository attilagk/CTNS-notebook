"""Verify canonical NFL notebook with sampling forbidden; keep harness out of source."""
from pathlib import Path
import sys,json,hashlib
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parents[2];name='2024-05-20-nfl-modeling';source=root/name/(name+'.ipynb');n=nbformat.read(source,4)
n.cells.insert(0,nbformat.v4.new_code_cell("import bambi\ndef _no_sampling(*args,**kwargs):\n    raise RuntimeError('Unexpected sampling during cache verification')\nbambi.Model.fit=_no_sampling"))
km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
NotebookClient(n,km=km,timeout=1800,resources={'metadata':{'path':str(source.parent)}},on_cell_start=lambda cell,cell_index,**k:print('CELL',cell_index,flush=True)).execute()
hits=sum(o.get('text','').count('Cache hit:') for c in n.cells for o in c.get('outputs',[]));assert hits==3,hits
n.cells.pop(0);nbformat.write(n,source)
Path(__file__).with_name('nfl-verification.json').write_text(json.dumps({'execution':'completed in base fresh kernel','sampling_forbidden':True,'cache_hits':hits,'canonical_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
print('Verified 3 cache hits; sampling forbidden',flush=True)
