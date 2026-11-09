import json,re,csv,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
m=json.loads((R/'baseline/notebook-dependencies.mapping.json').read_text());dot=(O.parent/'tudca-hcq-notebook-dependencies-v3.dot').read_text();ids=set(re.findall(r'^\s*\w+ \[id="([^"]+)"',dot,re.M));nodes=[n for n in m['nodes'] if n['id'] in ids];edges=[e for e in m['edges'] if e['source'] in ids and e['target'] in ids];files={r['baseline_graph_node_id']:r for r in csv.DictReader(open(R/'files.csv')) if r['baseline_graph_node_id'] in ids};events=[];seen=set();sources=[]
def add(t,kind,changed,evidence,certainty):
 changed=list(changed)
 if changed:events.append(dict(time=t,kind=kind,nodes=changed,evidence=evidence,timing=certainty));seen.update(changed)
for n in nodes:
 if n['kind']=='file':
  r=files.get(n['id'],{});n['initial']='missing' if 'missing' in r.get('initial_status','') else ('present' if r.get('initial_status')=='present' else 'unknown')
 else:n['initial']='missing'
# Receipts of recovery observations, retaining the distinction from action dates.
lookup={r['file_id']:i for i,r in files.items()}
for line in (R/'recovery-events.jsonl').read_text().splitlines():
 e=json.loads(line);i=lookup.get(e['file_id'])
 if i and e['current_status']=='present' and next(n for n in nodes if n['id']==i)['initial']!='present':add(e['observed_at'],'data recovery' if next(n for n in nodes if n['id']==i)['row']=='top' else 'notebook reproduction',[i],'recovery-events.jsonl:'+e['event_id'],'first documented observation')
# Completed reproduction manifests supply execution events and exact output paths where recorded.
for p in sorted((R/'reproductions').glob('**/*manifest.json')):
 if 'work' in p.parts:continue
 try:d=json.loads(p.read_text())
 except:continue
 if d.get('execution',d.get('execution_status')) not in ('passed','completed'):continue
 name=d.get('notebook','');name=Path(name).stem
 if not name:
  run=p.relative_to(R/'reproductions').parts[0]
  aliases={'latency-':'2023-11-07-5xfad-maze-latency','test-probe-':'2023-11-07-5xfad-maze-test-probe','mwm-test-probe-':'2024-06-26-mwm-test-probe-CO28154','co28154-':'2024-06-04-5xfad-behavior-CO28154','nfl-':'2024-05-20-nfl-modeling','biochemistry-':'2024-08-08-biochemistry','cytokines-':'2024-11-11-cytokines','maze-':'2023-10-13-5xfad-maze'}
  for prefix,identity in aliases.items():
   if run.startswith(prefix):name=identity;break
 match=[n['id'] for n in nodes if n['kind']=='notebook' and n['notebook'].split('/')[0]==name]
 t=d.get('finished_at',d.get('started_at',''))
 if isinstance(t,(float,int)):
  import datetime;t=datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
 if match and t:add(str(t),'notebook reproduction',match,str(p.relative_to(R)),'recorded execution')
# Promotion reports establish completed execution where old manifests use other formats.
for n in nodes:
 if n['kind']!='notebook' or any(n['id'] in e['nodes'] for e in events):continue
 name=n['notebook'].split('/')[0];ps=list((R/'promotions').glob(name+'*.md'))
 if name=='2023-11-07-5xfad-maze-latency':ps=list((R/'promotions').glob('latency-promotion.md'))
 if ps:
  import subprocess
  t=subprocess.check_output(['git','log','-1','--format=%cI','--',str(ps[0])],cwd=R.parent,text=True).strip()
  if t:add(t,'notebook reproduction',[n['id']],str(ps[0].relative_to(R)),'report committed by this time')
from datetime import datetime,timezone
for e in events:
 e['time']=datetime.fromisoformat(e['time'].replace('Z','+00:00')).astimezone(timezone.utc).isoformat()
events.sort(key=lambda e:e['time']);seen=set();filtered=[]
for e in events:
 e['nodes']=[i for i in e['nodes'] if i not in seen]
 if e['nodes']:filtered.append(e);seen.update(e['nodes'])
events=filtered;group=[]
for e in events:
 if group and (group[-1]['time'],group[-1]['kind'])==(e['time'],e['kind']):group[-1]['nodes']+=e['nodes'];group[-1]['evidence']+='; '+e['evidence']
 else:group.append(e)
data=dict(nodes=nodes,edges=edges,events=group,duration=15,note='Green notebooks: execution completed; convergence and reference agreement not implied. Generated outputs stay unresolved unless exact-path generation is documented.')
(O/'data.json').write_text(json.dumps(data,indent=2));print(len(nodes),len(edges),len(group),data['duration'])

(O/'evidence.json').write_text(json.dumps({'sources':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'baseline/input-inventory.csv',R/'baseline/output-producers.csv',R/'baseline/notebook-coverage.csv',R/'baseline/notebook-dependencies.mapping.json',R/'files.csv',R/'recovery-events.jsonl']},'limitations':['Observation times are bounds, not exact recovery action times.','Outputs turn green only with recorded file availability; unresolved identities remain amber.','Notebook execution can retain scientific warnings.']},indent=2))
