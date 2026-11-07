#!/usr/bin/env python3
"""Conservative static input audit. Never imports or executes notebook/helper code.

Run: python3 recovery-audit/audit_inputs.py
Cell numbers are one-based notebook cell positions; source lines are one-based.
Unknown expressions remain explicit. Both sides of runtime branches are inspected.
"""
import ast
import collections
import csv
import difflib
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STUDY = set('2023-10-13-5xfad-maze 2023-11-07-5xfad-maze-latency 2023-11-07-5xfad-maze-test-probe 2023-11-15-5xfad-nfl-gfap 2023-12-18-5xfad-nfl-gfap-frequentist 2024-01-05-5xfad-nfl-frequentist 2024-02-29-Y-maze 2024-06-04-5xfad-behavior-CO28154 2024-08-08-biochemistry'.split())
UNKNOWN = object()
READERS = {'open', 'read_csv', 'read_table', 'read_excel', 'ExcelFile', 'read_pickle', 'read_parquet', 'read_hdf', 'read_feather', 'read_json', 'read_html', 'read_fwf', 'read_sas', 'read_stata', 'load', 'loadtxt', 'genfromtxt', 'fromfile', 'from_netcdf', 'read_gpickle', 'read_edgelist', 'read_graphml', 'read_gml', 'parse', 'iterparse', 'read_sbml_model', 'load_model', 'load_json_model', 'read_matlab_model', 'read_text', 'read_bytes'}
WRITERS = {'to_csv', 'to_excel', 'to_pickle', 'to_parquet', 'to_hdf', 'to_feather', 'to_json', 'to_netcdf', 'savefig', 'dump', 'save', 'savetxt', 'write_text', 'write_bytes'}
PATH_RE = re.compile(r'(?:\.\.?/|~/|/home/|/Users/|/mnt/|/data/)|\.(?:csv|tsv|txt|xlsx?|xlsm|gz|zip|gmt|pkl|pickle|nc|netcdf|h5|hdf5?|xml|json|sif|rds|rda|mat|db|sqlite|bib|parquet)$', re.I)

def pathlike(v): return isinstance(v, str) and bool(PATH_RE.search(v))
def expr(n):
    try: return ast.unparse(n)[:1200]
    except Exception: return ''
def norm(s): return re.sub('[^a-z0-9]', '', s.lower())

class Audit:
    def __init__(self):
        self.rows=[]; self.writes=[]; self.coverage=[]; self.stack=[]; self.module_cache={}; self.nb=None; self.cell=0; self.origin=''; self.branch=''; self.seen=set(); self.visited_calls=set(); self.sites=[]
    def resolve(self, s):
        if re.match(r'^[a-zA-Z]+://',s): return s
        base=self.nb.parent.parent if self.nb.parent.name=='.ipynb_checkpoints' else self.nb.parent
        return os.path.abspath(os.path.expanduser(str(base/s)))
    def record(self, n, name, value, kind='read', note=''):
        vals=value if isinstance(value,(list,tuple)) else [value]
        if not vals: vals=[UNKNOWN]
        for v in vals:
            if v is not UNKNOWN and not isinstance(v,(str,Path)): v=UNKNOWN
            raw=str(v) if v is not UNKNOWN else ''
            resolved=self.resolve(raw) if raw else ''
            row=dict(notebook=str(self.nb.relative_to(ROOT)),manuscript='yes' if self.nb.parent.name in STUDY else 'no',cell=self.cell,source=self.origin,line=getattr(n,'lineno',0),loader=name,expression=expr(n),expected_path=raw,resolved_path=resolved,role='',status='',local_candidates='',dropbox_candidates='',candidate_evidence='',producer='',notes='; '.join(x for x in [self.branch,note] if x))
            key=tuple(str(row[k]) for k in ['notebook','cell','source','line','loader','expected_path','notes'])
            if (kind,key) in self.seen:continue
            self.seen.add((kind,key))
            (self.writes if kind=='write' else self.rows).append(row)
    def bind(self, target, val, env):
        if isinstance(target,ast.Name):env[target.id]=val
        elif isinstance(target,(ast.List,ast.Tuple)):
            for i,t in enumerate(target.elts):self.bind(t,val[i] if isinstance(val,(list,tuple)) and i<len(val) else UNKNOWN,env)
    def module(self, name):
        if name in self.module_cache:return self.module_cache[name]
        candidates=[self.nb.parent/(name.replace('.','/')+'.py'),ROOT/'src'/(name.replace('.','/')+'.py')]
        p=next((p for p in candidates if p.is_file()),None)
        if not p:return None
        env={};self.module_cache[name]=env
        old=self.origin;self.origin=str(p.relative_to(ROOT))
        try:self.block(ast.parse(p.read_text()).body,env,definitions_only=True)
        finally:self.origin=old
        return env
    def ev(self,n,e):
        if n is None:return UNKNOWN
        try:return self._ev(n,e)
        except (TypeError,ValueError,KeyError,IndexError,AttributeError,ZeroDivisionError,OverflowError):return UNKNOWN
    def _ev(self,n,e):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return e.get(n.id,UNKNOWN)
        if isinstance(n,(ast.List,ast.Tuple,ast.Set)):return [self.ev(x,e) for x in n.elts]
        if isinstance(n,ast.Dict):
            pairs=[(self.ev(k,e),self.ev(v,e)) for k,v in zip(n.keys,n.values)]
            return dict(pairs) if all(k is not UNKNOWN for k,v in pairs) else UNKNOWN
        if isinstance(n,ast.BinOp):
            a,b=self.ev(n.left,e),self.ev(n.right,e)
            if a is UNKNOWN or b is UNKNOWN:return UNKNOWN
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Div) and isinstance(a,(str,Path)) and isinstance(b,str):return str(Path(a)/b)
            if isinstance(n.op,ast.Mod):return a%b
        if isinstance(n,ast.JoinedStr):
            vals=[self.ev(x.value,e) if isinstance(x,ast.FormattedValue) else x.value for x in n.values]
            return ''.join(map(str,vals)) if all(v is not UNKNOWN for v in vals) else UNKNOWN
        if isinstance(n,ast.Subscript):
            v=self.ev(n.value,e);i=self.ev(n.slice,e)
            return v[i] if v is not UNKNOWN and i is not UNKNOWN else UNKNOWN
        if isinstance(n,ast.Attribute):
            v=self.ev(n.value,e)
            if isinstance(v,dict) and n.attr in v:return v[n.attr]
            return UNKNOWN
        if isinstance(n,(ast.ListComp,ast.SetComp,ast.GeneratorExp,ast.DictComp)):
            result=[]
            def walk(i,local):
                if len(result)>500:return
                if i==len(n.generators):
                    result.append((self.ev(n.key,local),self.ev(n.value,local)) if isinstance(n,ast.DictComp) else self.ev(n.elt,local));return
                g=n.generators[i];values=self.ev(g.iter,local)
                for v in values[:500] if isinstance(values,(list,tuple,range)) else [UNKNOWN]:
                    sub=dict(local);self.bind(g.target,v,sub)
                    for cond in g.ifs:self.ev(cond,sub)
                    walk(i+1,sub)
            walk(0,dict(e))
            return dict(result) if isinstance(n,ast.DictComp) and all(k is not UNKNOWN for k,v in result) else result
        if isinstance(n,ast.Call):return self.call(n,e)
        if isinstance(n,ast.Lambda):return ('lambda',n,dict(e),self.origin)
        # Traverse other expressions to retain nested loading operations.
        for child in ast.iter_child_nodes(n):
            if isinstance(child,ast.expr):self.ev(child,e)
        return UNKNOWN
    def call(self,n,e):
        self.visited_calls.add(n)
        name=expr(n.func);leaf=name.split('.')[-1];fn=self.ev(n.func,e)
        args=[]
        for a in n.args:
            if isinstance(a,ast.Starred):
                v=self.ev(a.value,e);args.extend(v if isinstance(v,(list,tuple)) else [UNKNOWN])
            else:args.append(self.ev(a,e))
        kw={k.arg:self.ev(k.value,e) for k in n.keywords if k.arg}
        first=args[0] if args else next((kw[k] for k in ['filepath_or_buffer','io','path','filename','file','fpath','path_or_buf'] if k in kw),UNKNOWN)
        if name in {'sns.load_dataset','seaborn.load_dataset'}:
            self.record(n,name,'https://github.com/mwaskom/seaborn-data/blob/master/'+str(first)+'.csv',note='Seaborn example dataset; fetched or cached by seaborn, not a local input path')
            return UNKNOWN
        if leaf in WRITERS:
            dest=args[1] if leaf in {'dump','save'} and len(args)>1 else first
            if isinstance(n.func,ast.Attribute) and leaf in {'write_text','write_bytes'}:dest=self.ev(n.func.value,e)
            self.record(n,name,dest,'write');return UNKNOWN
        if name=='zulu.parse':return UNKNOWN
        if leaf in READERS:
            mode=args[1] if leaf=='open' and len(args)>1 else kw.get('mode','r')
            if leaf in {'read_text','read_bytes'} and isinstance(n.func,ast.Attribute):first=self.ev(n.func.value,e)
            self.record(n,name,first,'write' if leaf=='open' and isinstance(mode,str) and any(x in mode for x in 'wax') else 'read',note=('mode='+str(mode)) if leaf=='open' else '')
            return UNKNOWN
        if isinstance(fn,tuple) and fn and fn[0] in {'fn','lambda'}:
            _,f,closure,origin=fn
            key=(origin,getattr(f,'lineno',0))
            if key in self.stack or len(self.stack)>12:return UNKNOWN
            local=dict(closure);params=list(f.args.posonlyargs)+list(f.args.args)
            defaults=[UNKNOWN]*(len(params)-len(f.args.defaults))+[self.ev(x,closure) for x in f.args.defaults]
            for i,p in enumerate(params):local[p.arg]=args[i] if i<len(args) else kw.get(p.arg,defaults[i])
            for p,d in zip(f.args.kwonlyargs,f.args.kw_defaults):local[p.arg]=kw.get(p.arg,self.ev(d,closure))
            old=self.origin;self.origin=origin;self.stack.append(key)
            try:return self.ev(f.body,local) if isinstance(f,ast.Lambda) else self.block(f.body,local)
            finally:self.stack.pop();self.origin=old
        if name in {'zip','list','tuple','enumerate','range','str','int','sorted'}:
            if any(a is UNKNOWN for a in args):return UNKNOWN
            if name=='zip':return list(zip(*args))[:500]
            if name=='enumerate':return list(enumerate(*args))[:500]
            if name=='range':return list(range(*args))[:500]
            if name in {'list','tuple','sorted'}:return list(args[0])[:500] if name!='sorted' else sorted(args[0])[:500]
            if name=='str':return str(args[0])
            if name=='int':return int(args[0])
        if name in {'os.path.join','posixpath.join'} and all(isinstance(a,str) for a in args):return os.path.join(*args)
        if name in {'Path','pathlib.Path'} and isinstance(first,str):return first
        if leaf in {'keys','values','items'} and isinstance(n.func,ast.Attribute):
            obj=self.ev(n.func.value,e)
            if isinstance(obj,dict):return list(getattr(obj,leaf)())
        if leaf in {'format','replace','join','lower','upper','split','rstrip','lstrip'} and isinstance(n.func,ast.Attribute):
            obj=self.ev(n.func.value,e)
            if isinstance(obj,str) and all(a is not UNKNOWN for a in args) and all(v is not UNKNOWN for v in kw.values()):return getattr(obj,leaf)(*args,**kw)
        if leaf in {'glob','iglob','listdir'}:
            self.record(n,name,first,note='Directory/pattern enumeration; downstream contents may be dynamic')
            if isinstance(first,str):
                return glob.glob(self.resolve(first))[:500] if leaf!='listdir' else os.listdir(self.resolve(first))[:500] if Path(self.resolve(first)).is_dir() else UNKNOWN
        if re.match(r'^(read_|load_|parse_|get_.*(?:data|network|genes|targets)|idatas?_from_|idatadf_from_)',leaf):
            self.record(n,name,first,note='External/unresolved helper; inspect implementation and remaining arguments')
        elif any(pathlike(a) for a in args) and not re.search(r'plot|save|write|export|mkdir|exists|isfile|isdir|join|append|insert|remove|display|Image|IFrame|chdir|download|urlretrieve',name,re.I):
            self.record(n,name,next(a for a in args if pathlike(a)),note='Potential file consumer inferred from path argument; verify manually')
        return UNKNOWN
    def block(self,body,e,definitions_only=False):
        result=UNKNOWN
        for n in body:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                e[n.name]=('fn',n,e,self.origin);continue
            if isinstance(n,ast.Import):
                for a in n.names:
                    m=self.module(a.name)
                    if m is not None:e[a.asname or a.name]=m
                continue
            if isinstance(n,ast.ImportFrom):
                m=self.module(n.module or '')
                if m:
                    for a in n.names:
                        if a.name=='*':e.update(m)
                        elif a.name in m:e[a.asname or a.name]=m[a.name]
                continue
            if definitions_only and not isinstance(n,(ast.Assign,ast.AnnAssign)):continue
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                val=self.ev(n.value,e)
                for t in n.targets if isinstance(n,ast.Assign) else [n.target]:self.bind(t,val,e)
            elif isinstance(n,ast.AugAssign):
                a=self.ev(n.target,e);b=self.ev(n.value,e)
                self.bind(n.target,a+b if isinstance(n.op,ast.Add) and isinstance(a,type(b)) and a is not UNKNOWN else UNKNOWN,e)
            elif isinstance(n,ast.Expr):self.ev(n.value,e)
            elif isinstance(n,ast.Return):result=self.ev(n.value,e)
            elif isinstance(n,(ast.For,ast.AsyncFor)):
                vals=self.ev(n.iter,e)
                for v in vals[:500] if isinstance(vals,(list,tuple)) else [UNKNOWN]:self.bind(n.target,v,e);self.block(n.body,e)
                self.block(n.orelse,e)
            elif isinstance(n,ast.If):
                self.ev(n.test,e);old=self.branch;self.branch='Conditional branch; runtime necessity unverified'
                a=dict(e);b=dict(e);self.block(n.body,a);self.block(n.orelse,b)
                for k in set(a)|set(b):
                    x,y=a.get(k,UNKNOWN),b.get(k,UNKNOWN)
                    try:equal=x==y
                    except Exception:equal=False
                    e[k]=x if equal else UNKNOWN
                self.branch=old
            elif isinstance(n,(ast.With,ast.AsyncWith)):
                for item in n.items:
                    self.ev(item.context_expr,e)
                    if item.optional_vars:self.bind(item.optional_vars,UNKNOWN,e)
                self.block(n.body,e)
            elif isinstance(n,ast.Try):
                for part in [n.body,*[h.body for h in n.handlers],n.orelse,n.finalbody]:self.block(part,e)
            elif isinstance(n,ast.While):
                self.ev(n.test,e);self.block(n.body,e);self.block(n.orelse,e)
        return result
    def shell(self,source):
        self.record(ast.Constant(value=source),'shell/magic',UNKNOWN,note='Manual review required: shell or non-Python cell; source retained in expression')
        for i,line in enumerate(source.splitlines(),1):
            if line.lstrip().startswith('#'):continue
            for m in re.finditer(r'(?:\.\./|\./|/home/|/Users/)[^\s\x27\x22|;<>]+',line):
                token=m.group(0).rstrip('),')
                node=ast.Constant(value=line);node.lineno=i
                if any(x in token for x in ['$', '{', '}', '*','?']):
                    self.record(node,'shell path expression',UNKNOWN,note='Unresolved shell path: '+token);continue
                prefix=line[:m.start()].rstrip()
                self.record(node,'shell path',token,'write' if prefix.endswith('>') else 'read',note='Shell path reference; command role requires review')
    def r_cell(self, source, env):
        # R is not executed: resolve literal scalar assignments and named/literal
        # loader arguments, leaving arbitrary R expressions explicit.
        def value(text):
            text=text.strip()
            if text in env:return env[text]
            try:
                v=ast.literal_eval(text)
                return v if isinstance(v,str) else UNKNOWN
            except (ValueError,SyntaxError):return UNKNOWN
        for lineno,line in enumerate(source.splitlines(),1):
            line=line.strip()
            if not line or line.startswith('#'):continue
            assign=re.match(r'^([A-Za-z_][\w.]*)\s*(?:<-|=)\s*(.*)$',line)
            if assign:env[assign[1]]=value(assign[2])
            for m in re.finditer(r'\b(read\.(?:csv2?|table|delim2?|xlsx)|readRDS|read_excel|read_tsv|read_csv|fread|load|scan|source|write\.(?:csv|table)|saveRDS|saveWorkbook)\s*\((.*)',line):
                name,rest=m.groups()
                quoted_or_symbol=r'(?:"[^"\n]*"|\'[^\'\n]*\'|[A-Za-z_][\w.]*)'
                named=re.search(r'\bfile\s*=\s*('+quoted_or_symbol+r')',rest)
                args=re.findall(quoted_or_symbol,rest)
                iswrite=name.startswith(('write','save'))
                token=named[1] if named else args[1] if iswrite and len(args)>1 else args[0] if args else ''
                v=value(token);node=ast.Constant(value=line);node.lineno=lineno
                self.record(node,'R '+name,v,'write' if iswrite else 'read',note='R static literal/variable resolution; dynamic R expressions require review')
                if not iswrite:self.sites.append(dict(notebook=str(self.nb.relative_to(ROOT)),cell=self.cell,line=lineno,call='R '+name,expression=line,coverage='analyzed' if v is not UNKNOWN else 'unresolved R expression'))
    def notebook(self,p):
        self.nb=p;self.module_cache={};self.stack=[];self.origin=str(p.relative_to(ROOT));env={};issues=[];code=0
        nb=json.loads(p.read_text()); pending=[]
        is_r=nb.get('metadata',{}).get('language_info',{}).get('name','').lower()=='r'
        for i,c in enumerate(nb.get('cells',[]),1):
            if c.get('cell_type')!='code':continue
            code+=1;self.cell=i;self.origin=str(p.relative_to(ROOT))+'#cell='+str(i);source=''.join(c.get('source',[]))
            if is_r:
                self.r_cell(source,env);continue
            if source.lstrip().startswith('%%') and not source.lstrip().startswith(('%%time\n','%%timeit\n','%%capture\n')):
                self.shell(source);issues.append(i);continue
            lines=[]
            for line in source.splitlines():
                stripped=line.lstrip()
                if stripped.startswith(('%','!')):
                    if not re.match(r'%(?:load_ext|reload_ext|autoreload|connect_info|matplotlib|time|timeit|capture)\b',stripped):self.shell(line);issues.append(i)
                    if stripped.startswith(('%time ','%timeit ')):
                        line=line[:len(line)-len(stripped)]+stripped.split(' ',1)[1]
                    else:line=''
                lines.append(line)
            try:tree=ast.parse('\n'.join(lines))
            except SyntaxError as err:
                self.shell(source)
                self.record(ast.Constant(value=source),'unparsed cell',UNKNOWN,note=str(err));issues.append(i);continue
            self.block(tree.body,env)
            for n in ast.walk(tree):
                if not isinstance(n,ast.Call):continue
                name=expr(n.func);leaf=name.split('.')[-1]
                if (leaf in READERS and name!='zulu.parse') or re.match(r'^(read_|load_|parse_|idatas?_from_|idatadf_from_)',leaf):
                    seen=n in self.visited_calls
                    site=dict(notebook=str(p.relative_to(ROOT)),cell=i,line=n.lineno,call=name,expression=expr(n),coverage='pending')
                    self.sites.append(site);pending.append((n,i,self.origin,site))
        for n,i,origin,site in pending:
            site['coverage']='analyzed' if n in self.visited_calls else 'unresolved deferred operation'
            if n not in self.visited_calls:
                self.cell=i;self.origin=origin
                self.record(n,expr(n.func),UNKNOWN,note='Deferred function/callback or unsupported construct; not statically reached; may be unused')
        self.coverage.append(dict(notebook=str(p.relative_to(ROOT)),manuscript='yes' if p.parent.name in STUDY else 'no',code_cells=code,manual_review_cells=','.join(map(str,sorted(set(issues))))))


def same_study(a,b):
    x,y=set(re.findall(r'CO[0-9]+',a,re.I)),set(re.findall(r'CO[0-9]+',b,re.I))
    return not x or not y or bool({v.lower() for v in x}&{v.lower() for v in y})

def main():
    audit=Audit()
    # Reproduction copies and their workspaces are evidence, not canonical notebooks.
    notebooks=sorted(p for p in ROOT.rglob('*.ipynb')
                     if 'recovery-audit' not in p.relative_to(ROOT).parts)
    for p in notebooks:audit.notebook(p)
    files=[];scan_errors=[]
    for base in [ROOT.parent/'resources',ROOT.parent/'results',ROOT]:
        def onerror(err):scan_errors.append(str(err))
        for current,dirs,names in os.walk(base,onerror=onerror):
            dirs[:]=[d for d in dirs if d not in {'.git','.ipynb_checkpoints','recovery-audit','__pycache__','.venv'}]
            for name in names:
                if not name.startswith('._'):files.append(str(Path(current)/name))
    byname=collections.defaultdict(list)
    for p in files:byname[Path(p).name.lower()].append(p)
    listing=json.loads((OUT/'dropbox-qps-listing.json').read_text())
    metadata=json.loads((OUT/'dropbox-candidate-metadata.json').read_text())
    for entry in listing['entries']:
        match=next((m for m in metadata if m['path']==entry['path']),None)
        if match:entry['path_display']=match['file'].get('path_display','')
    remote=[e for e in listing['entries'] if e and e.get('object_type')=='file']
    producers=collections.defaultdict(set)
    for r in audit.writes:
        if r['resolved_path']:producers[r['resolved_path']].add(f"{r['notebook']} cell {r['cell']} ({r['source']}:{r['line']})")
    for r in audit.rows:
        p=r['resolved_path'];name=Path(p).name if p else '';low=p.lower()
        r['role']='cached model fit / index' if any(x in low for x in ['idatas','fpaths.csv','.nc','.netcdf','bf-all']) else 'generated result' if '/results/' in p else 'source data / resource' if '/resources/' in p else 'input (role needs review)'
        if r['loader'] in {'shell/magic','unparsed cell','shell path expression'} or not p:r['status']='unresolved'
        elif re.match(r'^[a-zA-Z]+://',p):r['status']='remote URL (not checked)'
        elif Path(p).is_file():r['status']='present'
        elif Path(p).is_dir():r['status']='directory present; members need review'
        elif glob.has_magic(p):r['status']='pattern matched' if glob.glob(p) else 'pattern has no matches'
        else:r['status']='missing at expected path'
        if p and r['status'] in {'missing at expected path','pattern has no matches'}:
            exact=byname.get(name.lower(),[])
            near=[] if exact else difflib.get_close_matches(norm(name),{norm(Path(f).name) for f in files if Path(f).suffix.lower()==Path(p).suffix.lower()},n=3,cutoff=.82)
            local=exact or [f for f in files if norm(Path(f).name) in near and same_study(name,Path(f).name)][:5]
            r['local_candidates']=' | '.join(local)
            remote_exact=[e for e in remote if e['name'].lower()==name.lower()]
            remote_near=[] if remote_exact else [e for e in remote if same_study(name,e['name']) and Path(e['name']).suffix.lower()==Path(p).suffix.lower() and difflib.SequenceMatcher(None,norm(name),norm(e['name'])).ratio()>=.82][:5]
            r['dropbox_candidates']=' | '.join(e.get('path_display') or e['path'] for e in remote_exact+remote_near)
            r['candidate_evidence']='; '.join(filter(None,['Local: same basename; contents unverified' if exact else 'Local: similar filename only; may be a different version' if local else '', 'Dropbox: same basename; contents unverified' if remote_exact else 'Dropbox: similar filename only; may be a different version' if remote_near else '']))
            if '/resources/QPS/' not in p and '/resources/cell-based-assays/' not in p:r['notes']+='; Other Dropbox backups not searched; only supplied /QPS scope available'
        if p in producers:r['producer']=' | '.join(sorted(producers[p]))
    fields=list(audit.rows[0])
    def write_csv(name,rows,fields):
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    write_csv('input-inventory.csv',audit.rows,fields)
    write_csv('output-producers.csv',audit.writes,fields)
    write_csv('loading-sites.csv',audit.sites,list(audit.sites[0]))
    grouped=collections.defaultdict(list)
    for r in audit.rows:
        if r['status']=='missing at expected path':grouped[r['resolved_path']].append(r)
    missing_rows=[]
    for path,rows in sorted(grouped.items()):
        first=rows[0]
        missing_rows.append(dict(resolved_path=path,role=first['role'],manuscript='yes' if any(r['manuscript']=='yes' for r in rows) else 'no',notebooks=' | '.join(sorted({r['notebook'] for r in rows})),local_candidates=first['local_candidates'],dropbox_candidates=first['dropbox_candidates'],candidate_evidence=first['candidate_evidence'],producer=first['producer']))
    write_csv('missing-files.csv',missing_rows,list(missing_rows[0]))
    statuses=collections.Counter(r['status'] for r in audit.rows)
    for c in audit.coverage:
        rows=[r for r in audit.rows if r['notebook']==c['notebook']]
        c.update(input_records=len(rows),missing_paths=len({r['resolved_path'] for r in rows if r['status']=='missing at expected path'}),unresolved_records=sum(r['status']=='unresolved' for r in rows))
    write_csv('notebook-coverage.csv',audit.coverage,list(audit.coverage[0]))
    missing=sorted({r['resolved_path'] for r in audit.rows if r['status']=='missing at expected path'})
    summary=dict(audited_at_utc=datetime.now(timezone.utc).isoformat(),notebooks=len(notebooks),checkpoint_notebooks=sum('.ipynb_checkpoints' in p.parts for p in notebooks),loading_sites=len(audit.sites),input_records=len(audit.rows),statuses=dict(statuses),unique_missing_paths=len(missing),local_files_indexed=len(files),scan_errors=scan_errors,dropbox_listing_complete=not listing['has_more'],dropbox_files=len(remote))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=['# Notebook input recovery audit','', 'Static audit of all '+str(len(notebooks))+' notebooks. No notebooks were executed; no source data or Dropbox files were changed or downloaded. Includes two checkpoint notebooks. Cell numbers are one-based positions including Markdown cells; the cell column identifies the calling cell and source #cell identifies a notebook helper definition.','', '## Findings','',f"- {len(audit.rows)} input records; {len(missing)} distinct paths missing at the expected location.",f"- {statuses['present']} records point to existing files; {statuses['unresolved']} records require manual resolution.",f"- Local filename candidates searched across resources, results, and notebooks ({len(files)} files indexed; AppleDouble files excluded).",f"- Supplied Dropbox /QPS folder fully listed: {len(remote)} files. Candidate matching is by filename, not verified contents.",'', 'Counts describe static references, including conditional and exploratory code, rather than guaranteed execution failures. Multiple records may reference the same file.','', '## Manuscript inputs missing at expected paths','', '| Expected path | Local / Dropbox candidates | Upstream producer |','|---|---|---|']
    study_missing={r['resolved_path']:r for r in audit.rows if r['manuscript']=='yes' and r['status']=='missing at expected path'}
    for p,r in sorted(study_missing.items()):
        candidates='; '.join(filter(None,[r['local_candidates'],r['dropbox_candidates']])) or 'None found'
        report.append('| '+p.replace(str(ROOT.parent)+'/','')+' | '+candidates.replace('|',';')+' | '+(r['producer'] or 'Not identified').replace('|',';')+' |')
    report+=['','## Missing input summary by notebook','','| Notebook | Missing paths | Unresolved records |','|---|---:|---:|']
    for c in audit.coverage:report.append(f"| {'**' if c['manuscript']=='yes' else ''}{c['notebook']}{'**' if c['manuscript']=='yes' else ''} | {c['missing_paths']} | {c['unresolved_records']} |")
    report+=['','## How to use the inventory','','- `missing-files.csv`: one row per distinct missing expected path, with affected notebooks and recovery candidates.','- `input-inventory.csv`: one row per loading site and resolved path, or explicit unresolved expression. Filter `manuscript=yes` to prioritize the nine study directories.','- `output-producers.csv`: statically detected writes, matched to input paths where possible; this does not prove successful regeneration.','- `loading-sites.csv`: every syntactically recognized Python loading site, including deferred functions that could not be followed.\n- `notebook-coverage.csv`: every notebook, including those with no detected inputs, plus shell/unparsed cell review locations.','- `dropbox-candidate-metadata.json`: metadata confirmation for six same-name backup candidates; workbook contents remain unchecked.\n- `dropbox-qps-listing.json`: complete metadata snapshot of the supplied backup, with file IDs, paths, sizes, and modification times.','- `summary.json`: machine-readable counts and local traversal errors.','','## Limits and next steps','','This is conservative static analysis, not a successful rerun or proof of an exhaustive runtime dependency set. Paths assume the notebook directory as working directory; checkpoint snapshots use their original notebook directory. Libraries and executable availability are outside this data-file audit. Both sides of conditions are inspected; unknown branch-dependent values remain unresolved. Helper functions in this repository are followed where argument values can be resolved; external helpers are flagged rather than executed. R loaders are scanned for literal or scalar-variable paths; arbitrary R expressions remain unresolved. DataFrame-driven paths, callbacks, nested closures, environment variables, configuration files, manifests, archives, imports of external packages, and dynamically generated names may require manual inspection. Directory records do not establish that every required member is present. Shell cells are retained as unresolved review entries even when literal path references can be extracted. Relative output producer matching is exact-path only. No workbook sheets or columns have been validated. A same-name candidate may contain different data.','','Review unresolved entries and filename candidates before restoring inputs. Validate workbook sheets, columns, study identifiers, and versions against notebook expectations; recover generated outputs from their producers or locate additional backups. Backup coverage outside /QPS is unknown.']
    if scan_errors:report+=['','### Local scan errors','',*scan_errors]
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    import shutil
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT/'current')
    args=parser.parse_args()
    original=OUT
    OUT=args.output_dir.resolve()
    if OUT==original or (original/'baseline')==OUT or (original/'baseline') in OUT.parents:
        parser.error('Audit output must not overwrite legacy inventories or the baseline')
    OUT.mkdir(parents=True,exist_ok=True)
    for metadata in original.glob('dropbox-*.json'):
        shutil.copy2(metadata,OUT/metadata.name)
    scope=ROOT.parent/'results/study-dirs'
    if scope.exists():STUDY=set(scope.read_text().split())
    main()
