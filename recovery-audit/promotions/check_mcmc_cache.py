import sys,tempfile,pathlib,json
sys.path.insert(0,'src')
import mcmc_cache as c
import bambi as b,pandas as pd,arviz as az
m=b.Model('y ~ x',pd.DataFrame({'y':[1.,2.,3.],'x':[0.,1.,2.]}));settings={'draws':10};context={'input':'original'}
k,s=c.identity(m,settings,context)
assert c.identity(m,{'draws':11},context)[0]!=k
assert c.identity(m,settings,{'input':'changed'})[0]!=k
m.data.loc[0,'y']=2.;assert c.identity(m,settings,context)[0]!=k;m.data.loc[0,'y']=1.
m.set_priors({'x':b.Prior('Normal',mu=0,sigma=2)});assert c.identity(m,settings,context)[0]!=k
k,s=c.identity(m,settings,context);d=az.from_dict(posterior={'x':[[1.,2.,3.],[1.,2.,3.]]})
with tempfile.TemporaryDirectory() as td:
 c.save(td,k,s,d);assert c.fit_or_load(m,td,settings,context).posterior.equals(d.posterior)
 pathlib.Path(td,k+'.nc').write_bytes(b'broken')
 try:c.fit_or_load(m,td,settings,context)
 except ValueError:pass
 else:raise AssertionError('corrupt cache accepted')
print('Identity, valid hit, and corruption rejection checks passed')

a=b.Model('y ~ C(g, levels=lev)',pd.DataFrame({'y':[1.,2.,3.,4.],'g':['a','b','a','b']}),extra_namespace={'lev':['a','b']})
bb=b.Model('y ~ C(g, levels=lev)',pd.DataFrame({'y':[1.,2.,3.,4.],'g':['a','b','a','b']}),extra_namespace={'lev':['b','a']})
assert c.identity(a,settings,context)[0]!=c.identity(bb,settings,context)[0]
print('Explicit factor-order changes invalidate cache')

with tempfile.TemporaryDirectory() as td:
    m.fit=lambda **kw:d
    loaded=c.fit_or_load(m,td,settings,context)
    assert loaded.posterior.equals(d.posterior)
    called=[]
    def replacement(**kw):
        called.append(True)
        return d
    m.fit=replacement
    c.fit_or_load(m,td,settings,context,force=True)
    assert called
    changed={'input':'failed fit'}
    badkey,_=c.identity(m,settings,changed)
    def fail(**kw):raise RuntimeError('simulated sampling failure')
    m.fit=fail
    try:c.fit_or_load(m,td,settings,changed)
    except RuntimeError:pass
    else:raise AssertionError('failure not propagated')
    assert not pathlib.Path(td,badkey+'.nc').exists()
    original_version=c.version
    try:
        c.version=lambda package:original_version(package)+'-changed'
        assert c.identity(m,settings,context)[0]!=k
    finally:c.version=original_version
print('Missing entry, forced refit, failed fit, and version invalidation checks passed')

normal=[{'cell_type':'code','source':['force_refit = False\n','prior = 1\n']}]
forced=[{'cell_type':'code','source':['force_refit = True\n','prior = 1\n']}]
assert c.analysis_code_hash(normal)==c.analysis_code_hash(forced)
forced[0]['source'][1]='prior = 2\n'
assert c.analysis_code_hash(normal)!=c.analysis_code_hash(forced)
print('Refit toggle preserves identity; scientific edits invalidate it')
