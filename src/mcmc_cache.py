"""Conservative, checksum-verified local Bambi fit cache."""
import hashlib,json,os,tempfile,platform,re
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import arviz as az

def encode(x):
    if isinstance(x,dict):return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [encode(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'args') and hasattr(x,'name'):return {'name':x.name,'args':encode(x.args)}
    if x is None or isinstance(x,(str,int,float,bool)):return x
    raise TypeError(f'Unsupported cache identity value: {type(x)}')

def identity(model,settings,context):
    priors={}
    for name,c in model.components.items():
        priors[name]=encode(c.prior) if hasattr(c,'prior') else {n:encode(t.prior) for n,t in c.terms.items()}
    design={}
    for name,component in model.components.items():
        matrices=getattr(component,'design',None)
        if matrices is None:continue
        design[name]={}
        for kind in ['common','group']:
            part=getattr(matrices,kind,None)
            if part is None:continue
            matrix=part.design_matrix
            if hasattr(matrix,'toarray'):matrix=matrix.toarray()
            matrix=np.asarray(matrix)
            design[name][kind]={'shape':list(matrix.shape),'dtype':str(matrix.dtype),'sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),'terms':{n:encode(getattr(t,'levels',None)) for n,t in part.terms.items()}}
    data=model.data
    spec=dict(schema=1,design=design,context=context,formula=str(model.formula),family=model.family.name,links={k:v.name for k,v in model.family.link.items()},center=model.center_predictors,priors=priors,data_hash=hashlib.sha256(pd.util.hash_pandas_object(data,index=True).values.tobytes()).hexdigest(),columns=list(data.columns),index_names=list(data.index.names),dtypes=[str(t) for t in data.dtypes],categories={c:{'values':list(data[c].cat.categories),'ordered':data[c].cat.ordered} for c in data.select_dtypes('category')},settings=encode(settings),python=platform.python_version(),packages={p:version(p) for p in ['bambi','pymc','pytensor','arviz','numpy','pandas']})
    return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),spec

def save(directory,key,spec,result,provenance=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=directory,suffix='.nc');os.close(fd)
    try:
        result.to_netcdf(tmp);sha=hashlib.sha256(Path(tmp).read_bytes()).hexdigest();os.replace(tmp,directory/(key+'.nc'))
        manifest=dict(spec=spec,sha256=sha,provenance=provenance)
        fd,tmp=tempfile.mkstemp(dir=directory,suffix='.json');os.close(fd);Path(tmp).write_text(json.dumps(manifest,indent=2)+'\n');os.replace(tmp,directory/(key+'.json'))
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def fit_or_load(model,directory,settings,context,force=False,adopt=None):
    key,spec=identity(model,settings,context);p=Path(directory)/(key+'.nc');meta=p.with_suffix('.json')
    if not force and p.exists() and meta.exists():
        m=json.loads(meta.read_text())
        if m['spec']==spec and m['sha256']==hashlib.sha256(p.read_bytes()).hexdigest():
            print('Cache hit:',key[:12]);return az.from_netcdf(p)
        raise ValueError('Cache checksum or identity mismatch; inspect '+str(p))
    if adopt is not None and not force:
        result=az.from_netcdf(adopt);save(directory,key,spec,result,{'adopted_from':str(adopt),'source_sha256':hashlib.sha256(Path(adopt).read_bytes()).hexdigest()});return result
    result=model.fit(**settings);save(directory,key,spec,result);return result


def analysis_code_hash(cells):
    """Hash analysis code while normalizing the operational refit toggle."""
    code=[]
    for cell in cells:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        source=re.sub(r'^force_refit[ \t]*=[ \t]*(?:True|False)[ \t]*(?:#.*)?$', 'force_refit = False', source, flags=re.MULTILINE)
        code.append(source)
    return hashlib.sha256(json.dumps(code).encode()).hexdigest()
