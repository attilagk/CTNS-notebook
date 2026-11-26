#!/usr/bin/env python3
"""Read-only source scan; write a separate figure/provenance inventory. Baseline untouched."""
import ast,base64,csv,hashlib,json,re,subprocess
from collections import defaultdict,Counter
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SCOPE=re.findall(r'^## (\d[^\n]+)',(ROOT/'recovery-audit/target-artifacts/target-artifacts.md').read_text(), re.M)
SCOPE=list(dict.fromkeys(SCOPE))+['2023-09-26-cell-bayes-assays']
RUNMAP={'latency-':'2023-11-07-5xfad-maze-latency','test-probe-':'2023-11-07-5xfad-maze-test-probe','co28154-':'2024-06-04-5xfad-behavior-CO28154','mwm-test-probe-':'2024-06-26-mwm-test-probe-CO28154','biochemistry-':'2024-08-08-biochemistry','cytokines-':'2024-11-11-cytokines','nfl-':'2024-05-20-nfl-modeling','maze-':'2023-10-13-5xfad-maze'}
def owner(p):
 for n in SCOPE:
  if n in str(p):return n
 for part in p.parts:
  for pre,n in RUNMAP.items():
   if part.startswith(pre):return n
 if p.name.startswith('nfl-'):return '2024-05-20-nfl-modeling'
 return None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=defaultdict(set);sites=[];embedded=[];disabled=set()
for n in SCOPE:
 p=ROOT/n/(n+'.ipynb');nb=json.loads(p.read_text())
 for i,c in enumerate(nb['cells'],1):
  s=''.join(c.get('source',[]))
  for match in re.finditer(r'(?:attila_utils\.)?savefig\(',s):
   pass
  try: tree=ast.parse(s)
  except SyntaxError: tree=None
  if tree:
   inactive={id(child) for node in ast.walk(tree) if isinstance(node,ast.If) and isinstance(node.test,ast.Constant) and node.test.value is False for stmt in node.body for child in ast.walk(stmt)}
   for call in ast.walk(tree):
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='savefig':continue
    helper=isinstance(call.func.value,ast.Name) and call.func.value.id=='attila_utils'
    arg=call.args[1] if helper and len(call.args)>1 else call.args[0] if call.args else None
    expr=ast.unparse(arg) if arg else ''
    literal=arg.value if isinstance(arg,ast.Constant) and isinstance(arg.value,str) else None
    sites.append(dict(notebook=n,cell=i,expression=expr,literal=literal or '',resolution='disabled by if False' if id(call) in inactive else 'literal' if literal else 'dynamic - runtime evidence required',source=s))
    if literal:
     for fn in ([literal+'.png',literal+'.pdf'] if helper else [literal]):
      expected[(n,Path(fn).name)].add(f'export cell {i}')
      if id(call) in inactive:disabled.add((n,Path(fn).name))
  for j,o in enumerate(c.get('outputs',[])):
   text=''.join(o.get('text',[]))+'\n'+''.join(o.get('data',{}).get('text/plain',[]))
   for fn in re.findall(r"(?:named-figure/|figures/)([^\s'\"\],]+\.(?:png|pdf|svg))",text):expected[(n,Path(fn).name)].add(f'saved output cell {i} (historical output alone does not prove rerun)')
   for mime,ext in [('image/png','png'),('image/svg+xml','svg')]:
    data=o.get('data',{}).get(mime)
    if data:
     embedded.append(dict(notebook=n,cell=i,output=j,mime=mime,sha256=hashlib.sha256(base64.b64decode(''.join(data)) if ext=='png' else ''.join(data).encode()).hexdigest(),origin='canonical notebook embedded display; separately recoverable, not proof of export'))
# Artifact receipts bind hashes to files; keep all records, including absent worktree paths.
receipts=defaultdict(list)
hash_evidence=defaultdict(list)
for p in (ROOT/'recovery-audit').rglob('*.json'):
 if any(x in p.parts for x in ['baseline','snapshots','figure-reconciliation','node_modules','work','workspace','promotion-workspace']):continue
 try:d=json.loads(p.read_text())
 except Exception:continue
 n=owner(p)
 def walk(x,key=''):
  if key in ['references','reference','targets','target'] or key.startswith(('reference_','target_')):return
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) and not any(a in k for a in ['target','reference','input','source','helper']):
     hash_evidence[v].append({'receipt':str(p),'field':k,'context':key})
   for k,v in x.items():
    if isinstance(v,str) and re.fullmatch('[0-9a-f]{64}',v) and re.search(r'\.(png|pdf|svg)$',k):
     nn=owner(Path(k)) or n
     if nn in SCOPE:receipts[(nn,Path(k).name)].append(dict(receipt=str(p),path=k,sha256=v));expected[(nn,Path(k).name)].add('checksummed run artifact receipt')
    walk(v,k)
  elif isinstance(x,list):
   for v in x:walk(v,key)
 walk(d)
# Search current project, preserved CTNS backup, and retained figure storage; don't follow symlinks.
roots=[ROOT,ROOT.parent/'results',ROOT.parent/'notebooks~',Path('/home/attila/CTNS~'),Path('/home/attila/figures')]
found=defaultdict(list)
def corroborate(p):
 try:
  parts=p.parts;idx=parts.index('reproductions');run=Path(*parts[:idx+2]);mp=run/'manifest.json'
  if not mp.exists():return []
  d=json.loads(mp.read_text());snapshot=run/'source.ipynb'
  if not snapshot.exists() or sha(snapshot)!=d.get('source_sha256'):return []
  execution=d.get('execution',d.get('execution_status'))
  success=execution in ['passed','completed'] if isinstance(execution,str) else isinstance(execution,dict) and execution.get('status')=='completed'
  if not success:return []
  executed=list(run.glob('*executed.ipynb'))
  if not executed:return []
  # The run's successful manifest, matching input snapshot, executed notebook,
  # and isolated output location corroborate origin. This is weaker than an
  # original per-file checksum; preserve that distinction in the inventory.
  return [{'receipt':str(mp),'source_snapshot':str(snapshot),'executed_notebooks':[str(x) for x in executed],'basis':'successful isolated run + source snapshot hash + executed notebook; no historical per-file checksum'}]
 except (ValueError,OSError,KeyError):return []
for root in roots:
 if not root.exists():continue
 for p in root.rglob('*'):
  if not p.is_file() or p.name.startswith('._') or p.suffix.lower() not in ['.png','.pdf','.svg']:continue
  if any(x in p.parts for x in ['.git','node_modules','frames','figure-reconciliation']):continue
  n=owner(p)
  if n not in SCOPE:continue
  # Generic figures directory needs its run owner; above mapping covers it.
  h=sha(p);valid=False;detail=''
  try:
   if p.suffix=='.png':
    with Image.open(p) as im:im.load();detail=f'{im.width}x{im.height}'
   elif p.suffix=='.pdf':
    r=subprocess.run(['pdfinfo',str(p)],capture_output=True,text=True);assert r.returncode==0,r.stderr;detail=next((l for l in r.stdout.splitlines() if l.startswith('Pages:')),'PDF parsed')
   else:import xml.etree.ElementTree as ET;ET.parse(p);detail='SVG parsed'
   valid=True
  except Exception as e:detail=str(e)
  matches=[r for r in receipts[(n,p.name)] if r['sha256']==h]+hash_evidence[h]
  rerun='reproductions' in p.parts and not any(x in p.parts for x in ['references','reference'])
  corroboration=corroborate(p) if rerun else []
  category='verified_rerun_hash_match' if matches and valid else 'documented_rerun_corroborated' if corroboration and valid else 'surviving_run_copy_unbound_hash' if rerun and valid else 'older_backup_or_unverified_origin'
  rec=dict(notebook=n,filename=p.name,path=str(p),sha256=h,bytes=p.stat().st_size,mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),opens=valid,validation=detail,category=category,receipts=matches+corroboration)
  found[(n,p.name)].append(rec)
  expected[(n,p.name)].add('surviving file; scope relevance requires export/receipt check')
  if corroboration and any(x in p.parts for x in ['named-figure','figures']):expected[(n,p.name)].add('corroborated successful run export')
rows=[]
for (n,fn),e in sorted(expected.items()):
 copies=found[(n,fn)]; rr=receipts[(n,fn)]
 status='verified_rerun_survives' if any(c['category']=='verified_rerun_hash_match' for c in copies) else 'documented_rerun_survives_corroborated' if any(c['category']=='documented_rerun_corroborated' for c in copies) else 'run_copy_survives_provenance_needs_review' if any(c['category']=='surviving_run_copy_unbound_hash' for c in copies) else 'documented_export_missing' if rr else 'older_copy_only_no_verified_rerun' if copies else 'no_export_copy_or_rerun_receipt'
 confirmed=any(not x.startswith('surviving file;') for x in e)
 if (n,fn) in disabled and not rr:confirmed=False
 if n=='2023-09-26-cell-bayes-assays' and fn not in ['prior-posterior-curves-sigmoid-4.png','prior-posterior-curves-sigmoid-4.pdf']:confirmed=False
 rows.append(dict(notebook=n,expected_filename=fn,expected_by_export_or_receipt=confirmed,status=status,expectation_evidence='; '.join(sorted(e)),copies=copies,receipts=rr))
OUT.mkdir(exist_ok=True)
def csvwrite(name,data,fields):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in data:w.writerow({k:json.dumps(r[k]) if isinstance(r[k],(dict,list)) else r[k] for k in fields})
flat=[x for l in found.values() for x in l]
csvwrite('expected-figures.csv',[r for r in rows if r['expected_by_export_or_receipt']],['notebook','expected_filename','status','expectation_evidence','copies','receipts'])
csvwrite('legacy-or-extra-figures.csv',[r for r in rows if not r['expected_by_export_or_receipt']],['notebook','expected_filename','status','expectation_evidence','copies','receipts'])
csvwrite('found-copies.csv',flat,['notebook','filename','path','sha256','bytes','mtime_utc','opens','validation','category','receipts'])
csvwrite('export-sites.csv',sites,['notebook','cell','expression','literal','resolution','source'])
csvwrite('embedded-displays.csv',embedded,['notebook','cell','output','mime','sha256','origin'])
(OUT/'inventory.json').write_text(json.dumps(dict(scanned_at=datetime.now(timezone.utc).isoformat(),scope=SCOPE,roots=[str(p) for p in roots],figures=rows,export_sites=sites,embedded_displays=embedded),indent=2)+'\n')
confirmed=[r for r in rows if r['expected_by_export_or_receipt']]
summary=dict(expected_figure_files=len(confirmed),legacy_or_extra_files=len(rows)-len(confirmed),copies=len(flat),statuses=dict(Counter(r['status'] for r in confirmed)),dynamic_savefig_sites_requiring_runtime_names=sum(s['resolution'].startswith('dynamic') and s['notebook']!='2023-09-26-cell-bayes-assays' for s in sites),dynamic_savefig_sites_outside_requested_cell_figure=sum(s['resolution'].startswith('dynamic') and s['notebook']=='2023-09-26-cell-bayes-assays' for s in sites),per_notebook={n:dict(Counter(r['status'] for r in confirmed if r['notebook']==n)) for n in SCOPE})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
