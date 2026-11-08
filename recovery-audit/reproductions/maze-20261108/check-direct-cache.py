"""Focused checks of the notebook-local direct PyMC cache, without sampling."""
import os,sys,json,tempfile
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[2];sys.path[:0]=[str(ROOT/'src'),'/home/attila/src'];os.environ['MPLCONFIGDIR']=str(R/'mplconfig')
import numpy as np,arviz as az,pymc as pm,mwm,mcmc_cache,importlib.metadata
p=ROOT/'2023-10-13-5xfad-maze/2023-10-13-5xfad-maze.ipynb';book=json.load(open(p));setup=''.join(book['cells'][3]['source']).split('\nfrom pathlib import Path\nimport hashlib, json, inspect',1)[1]
os.chdir(R/'work/repo/2023-10-13-5xfad-maze')
fpath1='../../resources/QPS/5xFAD-mouse/CO26990-behavioral-data-cleaned.xlsx';fpath2='../../resources/QPS/5xFAD-mouse/CO28152-behavioral-data-cleaned.xlsx'
exec('from pathlib import Path\nimport hashlib,json,inspect\n'+setup)
calls=[];real=mwm.model_A;data=az.from_dict(posterior={'x':np.array([[1.,2.],[3.,4.]])})
def mock(*args,**kwargs):calls.append(True);return data
mwm.model_A=mock
try:
 with tempfile.TemporaryDirectory() as td:
  maze_cache_dir=Path(td);y=np.array([1.,2.]);x=np.array([1.,2.])
  cached_model_A(y,x,y_inf_A_val=0,censored=False);assert len(calls)==1
  cached_model_A(y,x,y_inf_A_val=0,censored=False);assert len(calls)==1
  cached_model_A(y+1,x,y_inf_A_val=0,censored=False);assert len(calls)==2
  cached_model_A(y,x,y_inf_A_val=10,censored=False);assert len(calls)==3
  old=mwm.mcmc_random_seed;mwm.mcmc_random_seed=old+1
  cached_model_A(y,x,y_inf_A_val=0,censored=False);assert len(calls)==4;mwm.mcmc_random_seed=old
  force_refit=True;cached_model_A(y,x,y_inf_A_val=0,censored=False);assert len(calls)==5;force_refit=False
  for f in maze_cache_dir.glob('*.nc'):f.write_bytes(b'corrupt')
  try:cached_model_A(y,x,y_inf_A_val=0,censored=False)
  except ValueError:pass
  else:raise AssertionError('Corruption accepted')
 print('Direct cache: miss, hit, data/argument/seed invalidation, forced refit and corruption rejection passed')
finally:mwm.model_A=real
