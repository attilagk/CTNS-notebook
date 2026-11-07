"""Extract exact amyloid values and independently reconcile CO28152 to PDF pp64-66.
Run with base Python: extract.py RESOURCE_DIR DOWNLOADED_CO28154_XLSX PDF_LAYOUT_TEXT STAGE_DIR
"""
import csv,json,re,sys,math
from pathlib import Path
import openpyxl
resources,raw54,textpath,stage=map(Path,sys.argv[1:]);stage.mkdir(exist_ok=True,parents=True)
report={'flagged_cells':[],'pdf_comparison':{},'sources':{}}
rows_by_study={}
for study,path,sheet in [('CO28152',resources/'CO28152_IV_data all.xlsx','Abeta'),('CO28154',raw54,'Aß')]:
 w=openpyxl.load_workbook(path,data_only=True);s=w[sheet];rows=[];rawrows={}
 for rn,row in enumerate(s.values,1):
  if rn<=3:continue
  if row[3] is None:
   assert row[2] in (None,'mean','SEM'),(rn,row)
   continue
  values=list(row[1:14]);values[2]=str(values[2]);rawrows[values[2]]=values[:]
  for c in range(5,13):
   v=values[c]
   if isinstance(v,str):
    assert v.startswith('*') or v in ['n.m.','n.m','n.m. ','n.m.\n'],v
    report['flagged_cells'].append({'study':study,'IRN':values[2],'source_sheet':sheet,'source_cell':f'{openpyxl.utils.get_column_letter(c+2)}{rn}','source_token':v,'cleaned':None})
    values[c]=None
   else:assert v is None or isinstance(v,(float,int)),v
  rows.append(values)
 assert len(rows)==60 and len(rawrows)==60
 rows_by_study[study]=rawrows
 columns=[None]*5+['CX FA','CX FA','HC FA','HC FA','CX DEA','CX DEA','HC DEA','HC DEA']
 sub=['Cohort','Group','IRN','Sex','Genotype']+['Aβ40','Aβ42']*4
 (stage/(study+'.json')).write_text(json.dumps([columns,sub]+rows,indent=2)+'\n')
 report['sources'][study]={'path':str(path),'sheet':sheet,'rows':60,'numeric_values':sum(isinstance(x,(int,float)) for r in rows for x in r[5:])}
# Parse only requested PDF sections; metadata+four value tokens on each animal row.
txt=textpath.read_text();a=txt.index('9.3.1.1 Aβ');b=txt.index('9.3.1.2 Aβ',a);end=txt.index('9.3.1.3 Inflammation',b)
parsed=[];compared=0
for fraction,section,offset in [('DEA',txt[a:b],9),('FA',txt[b:end],5)]:
 seen=set()
 for line in section.splitlines():
  m=re.fullmatch(r'\s*([IVX]+)\s+([A-G])\s+(\d+(?:\s+A)?)\s+([fm])\s+(tg/wt)\s+(\*?[\d,]+)\s+(\*?[\d,]+)\s+(\*?[\d,]+)\s+(\*?[\d,]+)\s*',line)
  if not m:continue
  cohort,group,irn,sex,gt,*tokens=m.groups();irn=' '.join(irn.split());assert irn not in seen;seen.add(irn)
  row=rows_by_study['CO28152'][irn];assert row[:5]==[cohort,group,irn,sex,gt]
  for j,t in enumerate(tokens):
   value=row[offset+j];flag=t.startswith('*');assert flag==(isinstance(value,str) and value.startswith('*')),(irn,t,value)
   actual=float(str(value).lstrip('*').replace(',','.'));expected=float(t.lstrip('*').replace(',','.'))
   precision=1 if ',' in t else 0;assert abs(actual-expected)<=0.50001*10**(-precision),(irn,t,value)
   compared+=1
  parsed.append({'fraction':fraction,'Cohort':cohort,'Group':group,'IRN':irn,'Sex':sex,'Genotype':gt,**dict(zip(['CX Aβ40','CX Aβ42','HC Aβ40','HC Aβ42'],tokens))})
 assert len(seen)==42,(fraction,len(seen))
 report['pdf_comparison'][fraction]={'animals':42,'numeric_cells':168,'metadata_matches':True,'outlier_flags_match':True,'precision':'PDF printed precision: 0.1 for CX DEA Aβ40; integer otherwise'}
with (stage/'pdf-extracted-values.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(parsed[0]));w.writeheader();w.writerows(parsed)
report['pdf_comparison']['total_matched_numeric_cells']=compared
(stage/'extraction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({s:{k:v for k,v in d.items() if k!='path'} for s,d in report['sources'].items()},indent=2));print('PDF values matched',compared,'flagged cells',len(report['flagged_cells']))
