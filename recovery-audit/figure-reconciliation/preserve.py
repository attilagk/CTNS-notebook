#!/usr/bin/env python3
"""Preserve distinct surviving exports, keeping references and evidence grades separate."""
import sys,json,csv,hashlib,shutil,subprocess
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
DEST=Path(sys.argv[1]);DEST.mkdir(parents=True,exist_ok=True)
data=json.loads((HERE/'inventory.json').read_text());rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f in data['figures']:
 if not f['expected_by_export_or_receipt']:continue
 rank={'verified_rerun_hash_match':0,'documented_rerun_corroborated':1,'surviving_run_copy_unbound_hash':2,'older_backup_or_unverified_origin':3}
 candidates=sorted([c for c in f['copies'] if c['opens']],key=lambda c:(rank[c['category']],-Path(c['path']).stat().st_mtime))
 seen=set()
 for c in candidates:
  if c['sha256'] in seen:continue
  seen.add(c['sha256'])
  grade=c['category'];kind='rerun-exports' if grade=='verified_rerun_hash_match' else 'documented-rerun-exports' if grade=='documented_rerun_corroborated' else 'run-copies-awaiting-verification' if grade=='surviving_run_copy_unbound_hash' else 'older-reference-copies'
  dst=DEST/kind/f['notebook']/c['sha256'][:16]/f['expected_filename'];dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha(dst)==c['sha256']
  else:shutil.copy2(c['path'],dst)
  assert sha(dst)==c['sha256']
  if dst.suffix=='.png':
   with Image.open(dst) as im:im.load()
  elif dst.suffix=='.pdf':subprocess.run(['pdfinfo',str(dst)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  selected=not any(r['notebook']==f['notebook'] and r['filename']==f['expected_filename'] and r['selected'] for r in rows) and grade in ['verified_rerun_hash_match','documented_rerun_corroborated']
  link=DEST/'selected'/f['notebook']/f['expected_filename']
  if selected:
   import os
   link.parent.mkdir(parents=True,exist_ok=True)
   target=os.path.relpath(dst,link.parent)
   if link.is_symlink():link.unlink()
   elif link.exists():raise RuntimeError('Refusing to overwrite '+str(link))
   link.symlink_to(target);assert sha(link)==c['sha256']
  rows.append(dict(notebook=f['notebook'],filename=f['expected_filename'],source=c['path'],durable_path=str(dst),selected=selected,selected_path=str(link) if selected else '',sha256=c['sha256'],grade=grade,opens=True,origin_evidence=c['receipts']))
(DEST/'preservation-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
(HERE/'durable-manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
with (HERE/'durable-files.csv').open('w') as out:
 fields=['notebook','filename','source','durable_path','selected','selected_path','sha256','grade','opens','origin_evidence'];w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
 for r in rows:w.writerow({**r,'origin_evidence':json.dumps(r['origin_evidence'])})
print('Preserved',len(rows),'distinct figure copies in',DEST)
