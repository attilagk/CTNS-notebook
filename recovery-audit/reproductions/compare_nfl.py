"""Compare the historical and reproduced deterministic HTML tables."""
import json,sys,math
from pathlib import Path
from html.parser import HTMLParser
class Cells(HTMLParser):
 def __init__(self):super().__init__();self.values=[];self.current=None
 def handle_starttag(self,t,a):
  if t=='td':self.current=''
 def handle_data(self,s):
  if self.current is not None:self.current+=s
 def handle_endtag(self,t):
  if t=='td' and self.current is not None:self.values.append(self.current.strip());self.current=None
run=Path(sys.argv[1]);original=json.loads((run/'source.ipynb').read_text());new=json.loads((run/'executed.ipynb').read_text());report={}
for cell in [1,2,17,49]:
 def values(n):
  p=Cells()
  for o in n['cells'][cell].get('outputs',[]):p.feed(''.join(o.get('data',{}).get('text/html',[])))
  return p.values
 a,b=values(original),values(new);assert a and len(a)==len(b),(cell,len(a),len(b))
 for x,y in zip(a,b):
  try:
   xf,yf=float(x),float(y)
   if math.isnan(xf):assert math.isnan(yf)
   else:
    precision=len(x.split('.')[1].split('e')[0]) if '.' in x else 0
    tolerance=0.50001*10**(-precision)
    if 'e' in x.lower():tolerance*=10**int(x.lower().split('e')[1])
    assert abs(xf-yf)<=tolerance,(cell,x,y)
  except ValueError:assert x==y,(cell,x,y)
 report[str(cell)]={'verdict':'matches historical displayed precision','cells_compared':len(a)}
(run/'table-comparisons.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
