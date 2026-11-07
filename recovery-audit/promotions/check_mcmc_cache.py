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
