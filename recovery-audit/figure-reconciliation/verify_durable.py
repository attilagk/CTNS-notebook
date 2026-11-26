#!/usr/bin/env python3
"""Reopen selected exports from persistent storage; render every PDF; make review sheets."""
import json,hashlib,subprocess,sys
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
HERE=Path(__file__).resolve().parent;out=HERE/'visual-review';out.mkdir(exist_ok=True)
rows=json.loads((HERE/'durable-manifest.json').read_text());selected=[r for r in rows if r['selected']]
results=[];pngs=[]
for r in selected:
 p=Path(r['selected_path']);assert p.exists();assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 if p.suffix=='.png':
  with Image.open(p) as im:im.load();size=im.size
  pngs.append(r)
 elif p.suffix=='.pdf':
  prefix=out/('pdf-'+r['sha256'][:16]);q=subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','400','-png',str(p),str(prefix)],capture_output=True,text=True);assert q.returncode==0,q.stderr
  with Image.open(str(prefix)+'.png') as im:im.load();size=im.size
 else:continue
 results.append(dict(path=str(p),sha256=r['sha256'],opened=True,dimensions=size,pdf_rendered=p.suffix=='.pdf'))
for k in range(0,len(pngs),16):
 sheet=Image.new('RGB',(1600,1280),'white');draw=ImageDraw.Draw(sheet)
 for j,r in enumerate(pngs[k:k+16]):
  x=(j%4)*400;y=(j//4)*320
  with Image.open(r['selected_path']) as im:thumb=ImageOps.contain(im.convert('RGB'),(390,270))
  sheet.paste(thumb,(x+(400-thumb.width)//2,y));draw.text((x+4,y+272),r['notebook'][:47],fill='black');draw.text((x+4,y+289),r['filename'][:53],fill='black')
 sheet.save(out/f'contact-{k//16+1:02}.png')
report=dict(selected_files=len(selected),reopened_files=len(results),png_files=len(pngs),pdf_files=sum(r['pdf_rendered'] for r in results),failures=[],files=results)
(HERE/'durable-verification.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='files'})
