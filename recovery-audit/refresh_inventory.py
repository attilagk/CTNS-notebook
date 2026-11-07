#!/usr/bin/env python3
"""Preserve original observations and publish independently timestamped refreshes."""
import csv,json,hashlib,subprocess,os,shutil,importlib.util
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 now=datetime.now(timezone.utc).isoformat();sid=now.replace(':','').replace('+','_');B=H/'baseline';C=H/'current';C.mkdir(exist_ok=True)
 if not B.exists():
  B.mkdir();sources={}
  names=subprocess.check_output(['git','ls-tree','-r','--name-only','17e477f','--','recovery-audit'],cwd=H.parent,text=True).splitlines()
  for name in names:
   if Path(name).suffix not in {'.csv','.json','.dot','.svg'}:continue
   data=subprocess.check_output(['git','show','17e477f:'+name],cwd=H.parent);p=B/Path(name).name;p.write_bytes(data);sources[p.name]=dict(sha256=digest(data),source_commit='17e477f')
  G=B/'graphs';G.mkdir()
  for p in H.glob('notebook-dependencies*'):
   if p.is_file():shutil.copy2(p,G/p.name)
  dump(B/'manual-observations.json',{p.name:read(p) for p in H.glob('*.csv') if p.name in ['missing-files.csv','input-inventory.csv']})
  dump(B/'manifest.json',dict(baseline_id='audit-2026-11-04',archived_at=now,files=sources,graph_observation_note='Preserved separately; graph filesystem observations need not match audit time.',graphs={p.name:digest(p.read_bytes()) for p in G.iterdir()}))
 manifest=json.loads((B/'manifest.json').read_text())
 for name,m in manifest['files'].items():assert digest((B/name).read_bytes())==m['sha256'],name
 spec=importlib.util.spec_from_file_location('audit',H/'audit_inputs.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.OUT=C
 scope=H.parent.parent/'results/study-dirs';a.STUDY=set(scope.read_text().split())
 for p in H.glob('dropbox-*.json'):shutil.copy2(p,C/p.name)
 a.main()
 initial=read(B/'input-inventory.csv');oldout=read(B/'output-producers.csv');cur=read(C/'input-inventory.csv');out=read(C/'output-producers.csv');states={r['resolved_path']:r['status'] for r in initial if r['resolved_path']}
 records={};edges=[];evidence=[];receipts={}
 for r in json.loads((H/'manuscript-restoration.json').read_text())['files']:
  if r.get('action') in ['restored','existing identical file retained']:receipts[r['destination']]=(r['sha256'],'restored from Dropbox','manuscript-restoration.json')
 r=json.loads((H/'co28154-cleaning-log.json').read_text());receipts[r['destination']]=(r['sha256'],'reconstructed from raw data','co28154-cleaning-log.json')
 def fid(path):
  try:key=str(Path(path).relative_to(H.parent.parent))
  except ValueError:key=path
  return 'f'+digest(key.encode())[:20]
 def file(path):
  if path in records:return records[path]
  p=Path(path);old=states.get(path,'unknown');remote='://' in path;pattern=any(c in path for c in '*?[')
  status='remote not checked' if remote else 'unresolved pattern' if pattern else 'present' if p.is_file() else 'directory' if p.is_dir() else 'missing'
  rec='recovered' if old=='missing at expected path' and status=='present' else 'still missing' if old=='missing at expected path' and status=='missing' else 'newly missing' if old=='present' and status=='missing' else 'already present' if old=='present' and status=='present' else 'unknown'
  method='';validation='not checked';ev=''
  if path in receipts and status=='present':
   sha,method,ev=receipts[path];validation='passed' if digest(p.read_bytes())==sha else 'changed since validation'
  elif p.is_symlink():method='symlink alias';ev='filesystem symlink'
  row=dict(file_id=fid(path),resolved_path=path,initial_status=old,initial_checked_at=json.loads((B/'summary.json').read_text())['audited_at_utc'] if old!='unknown' else '',initial_evidence='baseline/input-inventory.csv' if old!='unknown' else '',current_status=status,current_checked_at=now,recovery_status=rec,recovery_method=method,recovery_note='Availability and validation are independent.',evidence_id=ev,validation_status=validation,symlink_target=str(p.readlink()) if p.is_symlink() else '',referenced_now=False)
  records[path]=row;return row
 for rows,direction,version in [(initial,'consumes','baseline'),(oldout,'produces','baseline'),(cur,'consumes','current'),(out,'produces','current')]:
  for r in rows:
   path=r['resolved_path'];f=file(path) if path else None
   if f and version=='current':f['referenced_now']=True
   edges.append(dict(notebook=r['notebook'],file_id=f['file_id'] if f else '',direction=direction,version=version,study=r['manuscript'],cell=r['cell'],source=r['source'],expression=r['expression'],unresolved=not bool(f)))
 # Keep historical views and original fields while adding explicit state.
 for name in ['input-inventory.csv','missing-files.csv','output-producers.csv']:
  rows=read(H/name)
  for r in rows:
   if r['resolved_path']:r.update(file(r['resolved_path']))
   else:r.update(current_status='unresolved',initial_status='unknown',current_checked_at=now,recovery_status='unresolved')
   r['recovery_checked_at']=now
  write(H/name,rows)
 for f in records.values():
  f['study']=any(e['file_id']==f['file_id'] and e['study']=='yes' for e in edges)
  if f['symlink_target']:edges.append(dict(notebook='',file_id=f['file_id'],direction='alias',version='current',study=str(f['study']),target_file_id=fid(str(Path(f['resolved_path']).parent/f['symlink_target']))))
 mapping=json.loads((B/'notebook-dependencies.mapping.json').read_text())
 for node in mapping['nodes']:
  f=records.get(node.get('resolved_path'))
  if f:
   f['baseline_graph_node_id']=node['id']
   f['graph_observed_present']=node.get('present_on_disk')
   f['baseline_position']=json.dumps(node.get('position_inches',{}))
   f['recovery_candidates']=json.dumps(node.get('recovery_candidates',[]))
 for e in edges:
  e['dependency_id']='d'+digest(json.dumps([e.get(k,'') for k in ['notebook','file_id','direction','source','expression']],sort_keys=True).encode())[:20]
  notebook=H.parent/e.get('notebook','')
  if e['version']=='current' and notebook.is_file():
   try:e['cell_id']=json.loads(notebook.read_text())['cells'][int(e['cell'])-1].get('id','')
   except (KeyError,ValueError,IndexError):e['cell_id']=''
 log=json.loads((H/'co28154-cleaning-log.json').read_text())
 raw=log['source'];destination=log['destination']
 rawfile=file(raw)
 edges.append(dict(dependency_id='raw-co28154',file_id=fid(destination),target_file_id=rawfile['file_id'],direction='reconstructed_from',version='current',evidence='co28154-cleaning-log.json',study='yes'))
 rawfile['study']=True
 write(H/'files.csv',list(records.values()));write(H/'dependencies.csv',edges)
 coverage=read(H/'notebook-coverage.csv')
 for r in coverage:
  fs={e['file_id'] for e in edges if e.get('notebook')==r['notebook'] and e['version']=='current' and e['direction']=='consumes'};rr=[f for f in records.values() if f['file_id'] in fs]
  r.update(current_unique_input_files=len(fs),current_missing_files=sum(f['current_status']=='missing' for f in rr),recovered_paths=sum(f['recovery_status']=='recovered' for f in rr),recovery_checked_at=now)
 write(H/'notebook-coverage.csv',coverage)
 from collections import Counter
 summary=dict(baseline_id=manifest['baseline_id'],snapshot_id=sid,checked_at=now,baseline=json.loads((B/'summary.json').read_text()),current=json.loads((C/'summary.json').read_text()),unique_file_statuses=dict(Counter(f['current_status'] for f in records.values())),recovery_statuses=dict(Counter(f['recovery_status'] for f in records.values())),study_recovery_statuses=dict(Counter(f['recovery_status'] for f in records.values() if f['study'])),original_missing_paths=len(read(B/'missing-files.csv')))
 dump(H/'recovery-status-summary.json',summary);dump(H/'dependency-state.json',dict(summary=summary,files=list(records.values()),dependencies=edges,provenance={'CO28154':'co28154-cleaning-log.json','restorations':'manuscript-restoration.json','manual':'baseline/manual-observations.json'}))
 S=H/'snapshots'/sid;S.mkdir(parents=True)
 for name in ['files.csv','dependencies.csv','dependency-state.json','recovery-status-summary.json','input-inventory.csv','missing-files.csv','output-producers.csv','notebook-coverage.csv']:shutil.copy2(H/name,S/name)
 dump(S/'manifest.json',dict(snapshot_id=sid,files={p.name:digest(p.read_bytes()) for p in S.iterdir()}))
 journal=H/'recovery-events.jsonl';existing=set()
 if journal.exists():existing={json.loads(l)['event_id'] for l in journal.read_text().splitlines()}
 with journal.open('a') as h:
  for f in records.values():
   event={k:f[k] for k in ['file_id','current_status','recovery_status','validation_status','symlink_target','evidence_id']};eid=digest(json.dumps(event,sort_keys=True).encode())
   if eid not in existing:h.write(json.dumps(dict(event_id=eid,observed_at=now,**event))+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
