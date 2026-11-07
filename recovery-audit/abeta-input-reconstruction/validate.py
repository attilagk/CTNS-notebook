"""Check export values, notebook imports and historical target coefficients."""
from pathlib import Path
from html.parser import HTMLParser
import sys,json,math
import pandas as pd,numpy as np
import statsmodels.formula.api as smf
stage=Path(sys.argv[1]);root=Path(__file__).resolve().parents[2]
n=json.loads((root/'2024-08-08-biochemistry/2024-08-08-biochemistry.ipynb').read_text());ns={'pd':pd,'np':np,'smf':smf}
for study in ['CO28152','CO28154']:
 values=json.loads((stage/(study+'.json')).read_text());data=pd.read_excel(stage/(study+'-biochemistry-cleaned-Abeta.xlsx'),sheet_name='Abeta',header=[0,1],index_col=[0,1,2,3,4]);assert data.shape==(60,8)
 expected=pd.DataFrame([r[5:] for r in values[2:]],index=pd.MultiIndex.from_tuples([tuple(r[:5]) for r in values[2:]]),columns=data.columns)
 # pandas reads numeric IRNs as ints; compare their string form to source identifiers.
 assert [tuple(str(v) for v in x) for x in data.index]==[tuple(str(v) for v in x) for x in expected.index]
 assert np.allclose(data.to_numpy(dtype=float),expected.to_numpy(dtype=float),equal_nan=True,rtol=1e-14,atol=1e-12)
 assert data.index.is_unique
for i in [2,3,4,9,10,13]:
 code=''.join(n['cells'][i]['source'])
 if i==2:
  code=code.replace("fpathl = ['../../resources/QPS/5xFAD-mouse/CO2815' + i + '-biochemistry-cleaned-Abeta.xlsx' for i in ['2', '4']]",'fpathl = '+repr([str(stage/(s+'-biochemistry-cleaned-Abeta.xlsx')) for s in ['CO28152','CO28154']]))
 if i==13:code=code.split('with pd.ExcelWriter')[0]
 exec(code,ns)
class T(HTMLParser):
 def __init__(self):super().__init__();self.values=[];self.v=None
 def handle_starttag(self,t,a):
  if t=='td':self.v=''
 def handle_data(self,x):
  if self.v is not None:self.v+=x
 def handle_endtag(self,t):
  if t=='td' and self.v is not None:self.values.append(self.v.strip());self.v=None
report={}
for study,exp,cell in [('CO28154','TUDCA and HCQ',14),('CO28152','TUDCA or HCQ',15)]:
 actual=ns['results_params'][(study,exp)];p=T()
 for o in n['cells'][cell].get('outputs',[]):p.feed(''.join(o.get('data',{}).get('text/html',[])))
 assert len(p.values)==actual.size,(len(p.values),actual.size)
 for printed,value in zip(p.values,actual.to_numpy().ravel()):
  # Printed six decimals or scientific notation: half one unit of last printed digit.
  mantissa,*exponent=printed.lower().split('e');dec=len(mantissa.split('.')[1]) if '.' in mantissa else 0
  tol=.50001*10**(-dec+(int(exponent[0]) if exponent else 0))
  assert math.isclose(float(printed),float(value),rel_tol=0,abs_tol=tol),(study,printed,value)
 report[study]={'historical_parameter_cells_matched':actual.size,'rows':60,'numeric_cells':int(ns['Abeta'].xs(study,level='Study').notna().sum().sum())}
 actual.to_csv(stage/(study+'-validated-params.csv'))
report.update(status='passed',wide_shape=list(ns['Abeta'].shape),long_shape=list(ns['Abeta_long'].shape),scope='Input round trip and notebook import/data-engineering/selected OLS cells only; not full reproduction')
(stage/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
