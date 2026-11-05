#!/usr/bin/env python3
"""Prepare and verify a CO28154 workbook reconstruction from agreed rules.

prepare: write a data payload for clean_co28154.mjs, without changing workbooks.
verify: compare the staged exported workbook against all retained source cells.
install: atomically copy the verified workbook to its missing notebook input path.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from compare_behavioral_workbooks import read_workbook, val, rows, colname

HERE=Path(__file__).resolve().parent
WORK=Path('/tmp/ctns-co28154-cleaning')
RAW=HERE.parent.parent/'resources/QPS/5xFAD-mouse/CO28154-rawdata-in-vivo.xlsx'
STAGED=WORK/'CO28154-behavioral-data-cleaned.xlsx'
DEST=RAW.parent/'CO28154-behavioral-data-cleaned.xlsx'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def jsonvalue(v):
    if isinstance(v,Decimal):return int(v) if v==v.to_integral_value() else float(v)
    return v

def typed(v):
    if isinstance(v,Decimal):return dict(type='number',value=str(v))
    if v is None:return dict(type='blank',value=None)
    return dict(type='text',value=v)

def prepare():
    sheets=read_workbook(RAW)
    assert [s['name'] for s in sheets]==['MWM day 1-4','Probe Trial day5','Y-Maze','body weight']
    payload=dict(source=str(RAW),source_sha256=digest(RAW),rules_reference='co28152-cleaning-reconciliation.md and behavioral-data-cleaning.md',sheets=[])
    for sheet in sheets:
        hdr=next(row for row in rows(sheet) if {'Cohort','Group','IRN'}<={val(sheet,row,c) for c in range(1,25)})
        lookup={str(val(sheet,hdr,c)).lower():c for c in range(1,25) if val(sheet,hdr,c) is not None}
        cov=[lookup[k] for k in ['cohort','group','sex','irn']]
        left=min(c for (r,c),cell in sheet['cells'].items() if cell['value'] is not None)
        top=min(r for (r,c),cell in sheet['cells'].items() if cell['value'] is not None)
        right=max(c for (r,c),cell in sheet['cells'].items() if cell['value'] is not None)
        data=[];summary=[]
        for row in rows(sheet):
            if row<=hdr:continue
            nonempty=any(val(sheet,row,c) is not None for c in range(left,right+1))
            if not nonempty:continue
            label=val(sheet,row,lookup['cohort'])
            if isinstance(label,str) and label.strip().casefold() in {'mean','sem'}:
                summary.append(row);continue
            assert val(sheet,row,lookup['sex']) in {'f','m'},f'Unexpected record at {sheet["name"]}:{row}'
            assert val(sheet,row,lookup['irn']) is not None
            data.append(row)
        mode=sheet['name']
        if mode=='MWM day 1-4':
            # The saved notebook output and mwm.read_data_train() index expect
            # Cohort/Group/Sex/IRN. Only the columns move; their values/types do not.
            order=cov+list(range(left+4,right+1))
            retained=[hdr-1,hdr]+data
            matrix=[[jsonvalue(val(sheet,r,c)) for c in order] for r in retained]
            matrix[0][:4]=['Covariates',None,None,None]
            merges=['A1:D1','E1:H1','I1:L1','M1:P1','Q1:T1','U1:X1']
            headerrows=2
        elif mode=='Probe Trial day5':
            order=cov+list(range(left+4,right+1))
            retained=[hdr]+data
            matrix=[[jsonvalue(val(sheet,r,c)) for c in order] for r in retained]
            merges=[];headerrows=1
        elif mode=='Y-Maze':
            order=list(range(left,right+1))
            header=[val(sheet,hdr,c) for c in order[:4]]
            parent=None
            for c in order[4:]:
                candidate=val(sheet,hdr-1,c)
                if candidate is not None:parent=candidate
                assert parent in {'Total entries','Correct entries %'}
                header.append(parent+', '+str(val(sheet,hdr,c)))
            matrix=[header]+[[jsonvalue(val(sheet,r,c)) for c in order] for r in data]
            retained=[hdr]+data
            merges=[];headerrows=1
        else:
            # Preserve the body-weight title and existing lowercase sex label.
            order=list(range(left,right+1));retained=list(range(top,hdr+1))+data
            matrix=[[jsonvalue(val(sheet,r,c)) for c in order] for r in retained]
            merges=[];headerrows=hdr-top+1
        checks=[]
        for dstrow,srcrow in enumerate(data,headerrows+1):
            for dstcol,srccol in enumerate(order,1):
                checks.append(dict(raw_cell=f'{colname(srccol)}{srcrow}',clean_cell=f'{colname(dstcol)}{dstrow}',expected=typed(val(sheet,srcrow,srccol))))
        assert not any(sheet['cells'].get((r,c),{}).get('formula') is not None for r in data for c in order)
        removed_formulas=sum(cell['formula'] is not None for (r,c),cell in sheet['cells'].items() if r in summary)
        item=dict(name=mode,values=matrix,range=f'A1:{colname(len(order))}{len(matrix)}',headerRange=f'A1:{colname(len(order))}{headerrows}',headerRows=headerrows,merges=merges,previewRange=f'A1:{"L" if len(order)>12 else colname(len(order))}{min(len(matrix),headerrows+5)}',animal_records=len(data),summary_rows_removed=summary,summary_labels=dict(Counter(str(val(sheet,r,lookup['cohort'])) for r in summary)),source_header_row=hdr,source_column_order=order,source_data_rows=data,leading_blank_rows_removed=top-1,leading_blank_columns_removed=left-1,source_formula_cells_removed=removed_formulas,record_checks=checks,group_counts=dict(Counter(str(val(sheet,r,lookup['group'])) for r in data)))
        payload['sheets'].append(item)
    assert [s['animal_records'] for s in payload['sheets']]==[162,162,163,196]
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK/'payload.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps([dict(sheet=s['name'],records=s['animal_records'],removed_summaries=len(s['summary_rows_removed']),range=s['range']) for s in payload['sheets']],indent=2))


def preview(raw=False):
    """Render the relevant actual cell ranges for content/layout review."""
    from openpyxl import load_workbook
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    book=load_workbook(RAW if raw else STAGED,data_only=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    bold=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',14)
    for ws in book.worksheets:
        ncols=min(ws.max_column,12);nrows=min(ws.max_row,9)
        widths=[max(65,min(230,int(ws.column_dimensions[__import__('openpyxl').utils.get_column_letter(c)].width*7+5))) for c in range(1,ncols+1)]
        offsets=[0]
        for width in widths:offsets.append(offsets[-1]+width)
        rh=42;img=Image.new('RGB',(sum(widths)+20,nrows*rh+54),'white');draw=ImageDraw.Draw(img)
        draw.text((10,8),ws.title,fill='black',font=bold)
        covered=set();merges={}
        for merge in ws.merged_cells.ranges:
            if merge.min_row<=nrows and merge.min_col<=ncols:
                merges[(merge.min_row,merge.min_col)]=(min(merge.max_row,nrows),min(merge.max_col,ncols))
                covered.update((r,c) for r in range(merge.min_row,min(merge.max_row,nrows)+1) for c in range(merge.min_col,min(merge.max_col,ncols)+1) if (r,c)!=(merge.min_row,merge.min_col))
        for r in range(1,nrows+1):
            for c in range(1,ncols+1):
                if (r,c) in covered:continue
                r2,c2=merges.get((r,c),(r,c))
                x,y=10+offsets[c-1],44+(r-1)*rh
                x2,y2=10+offsets[c2],44+r2*rh
                cell=ws.cell(r,c);header=bool(cell.font.bold)
                draw.rectangle([x,y,x2,y2],fill='#edf0f3' if header else 'white',outline='#c9c9c9')
                v=cell.value
                if v is None:continue
                text=f'{v:.6g}' if isinstance(v,float) else str(v)
                wrapped=textwrap.wrap(text,width=max(8,int((x2-x-8)/8)))[:2]
                draw.multiline_text((x+4,y+3),'\n'.join(wrapped),fill='black',font=bold if header else font,spacing=2)
        prefix='raw' if raw else 'clean'
        target=WORK/(prefix+'-'+ws.title.replace(' ','_').replace('/','_')+'.png')
        img.save(target)
    print('Rendered', 'raw' if raw else 'cleaned', 'sheet previews.')


def build():
    """Fallback authoring: Artifact Tool initialization failed in both contexts."""
    from copy import copy
    from openpyxl import Workbook,load_workbook
    from openpyxl.styles import Alignment
    from openpyxl.utils import get_column_letter
    payload=json.loads((WORK/'payload.json').read_text())
    assert digest(RAW)==payload['source_sha256']
    raw=load_workbook(RAW,data_only=False)
    book=Workbook();book.remove(book.active)
    for item in payload['sheets']:
        source=raw[item['name']];sheet=book.create_sheet(item['name'])
        hdr=item['source_header_row']
        if item['name']=='MWM day 1-4':source_rows=[hdr-1,hdr]+item['source_data_rows']
        elif item['name'] in ['Probe Trial day5','Y-Maze']:source_rows=[hdr]+item['source_data_rows']
        else:source_rows=list(range(item['leading_blank_rows_removed']+1,hdr+1))+item['source_data_rows']
        assert len(source_rows)==len(item['values'])
        for r,row in enumerate(item['values'],1):
            sr=source_rows[r-1]
            if source.row_dimensions[sr].height is not None:sheet.row_dimensions[r].height=source.row_dimensions[sr].height
            for c,v in enumerate(row,1):
                original=source.cell(sr,item['source_column_order'][c-1]);cell=sheet.cell(r,c,v)
                for attr in ['font','fill','border','alignment','protection']:
                    setattr(cell,attr,copy(getattr(original,attr)))
                cell.number_format=original.number_format
        for c,sc in enumerate(item['source_column_order'],1):
            sheet.column_dimensions[get_column_letter(c)].width=source.column_dimensions[get_column_letter(sc)].width
        if item['name']=='MWM day 1-4':
            for attr in ['font','fill','border','alignment','protection']:
                setattr(sheet['A1'],attr,copy(getattr(source['E1'],attr)))
            sheet['A1'].number_format=source['E1'].number_format
        if item['name']=='Y-Maze':
            sheet.row_dimensions[1].height=34
            for cell in sheet[1]:cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
            for c in range(5,9):sheet.column_dimensions[get_column_letter(c)].width=22
        if item['name']=='Probe Trial day5':
            sheet.row_dimensions[1].height=30
            for cell in sheet[1]:cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
            sheet.column_dimensions['E'].width=23
        for merge in item['merges']:sheet.merge_cells(merge)
        sheet.freeze_panes='A'+str(item['headerRows']+1)
    book.save(STAGED)
    # Binary-float writers can shorten XML number text. Preserve the exact
    # source decimal representation in every retained numeric cell, without
    # changing numeric types or applying any measurement transformation.
    ET.register_namespace('',NS['m'])
    fixed=WORK/'exact-numeric-values.xlsx';repaired=0
    with zipfile.ZipFile(STAGED) as zin,zipfile.ZipFile(fixed,'w',compression=zipfile.ZIP_DEFLATED) as zout:
        replacements={}
        for i,item in enumerate(payload['sheets'],1):
            path=f'xl/worksheets/sheet{i}.xml';xml=ET.fromstring(zin.read(path))
            cells={c.get('r'):c for c in xml.findall('m:sheetData/m:row/m:c',NS)}
            for check in item['record_checks']:
                if check['expected']['type']!='number':continue
                cell=cells[check['clean_cell']];cell.set('t','n')
                value=cell.find('m:v',NS)
                assert value is not None
                original_text=check['expected']['value']
                repaired+=value.text!=original_text
                value.text=original_text
            replacements[path]=ET.tostring(xml,encoding='utf-8')
        for info in zin.infolist():zout.writestr(info,replacements.get(info.filename,zin.read(info.filename)))
    fixed.replace(STAGED)
    (WORK/'build-meta.json').write_text(json.dumps(dict(authoring_engine='openpyxl fallback; Artifact Tool timed out during import/initialization inside and outside sandbox',original_numeric_xml_serializations_preserved=repaired),indent=2)+'\n')
    print('Built cleaned workbook; preserved source numeric XML in',repaired,'cells that the writer would otherwise serialize differently.')


def verify():
    payload=json.loads((WORK/'payload.json').read_text())
    assert digest(RAW)==payload['source_sha256'],'Raw workbook changed'
    actual=read_workbook(STAGED)
    assert [s['name'] for s in actual]==[s['name'] for s in payload['sheets']]
    log=dict(authoring=json.loads((WORK/'build-meta.json').read_text()),verified_at_utc=datetime.now(timezone.utc).isoformat(),source=str(RAW),source_sha256=digest(RAW),staged=str(STAGED),destination=str(DEST),sheets=[])
    all_changes=[]
    for expected,sheet in zip(payload['sheets'],actual):
        for r in range(1,expected['headerRows']+1):
            for c,v in enumerate(expected['values'][r-1],1):
                assert val(sheet,r,c)==v,(sheet['name'],'header',r,c)
        mismatches=[];numeric_xml_variations=[]
        for check in expected['record_checks']:
            import re
            m=re.fullmatch(r'([A-Z]+)(\d+)',check['clean_cell'])
            from compare_behavioral_workbooks import colnum
            v=val(sheet,int(m[2]),colnum(m[1]));e=check['expected']
            if e['type']=='number':
                ok=isinstance(v,Decimal) and v==Decimal(e['value'])
                if not ok and isinstance(v,Decimal) and float(v)==float(Decimal(e['value'])):
                    numeric_xml_variations.append(dict(cell=check['clean_cell'],source=e['value'],exported=str(v)))
            elif e['type']=='blank':ok=v is None
            else:ok=isinstance(v,str) and v==e['value']
            if not ok:mismatches.append(dict(cell=check['clean_cell'],source=e['value'],exported=str(v)))
        assert set(sheet['merges'])==set(expected['merges'])
        assert sheet['dimension']==expected['range'],(sheet['name'],sheet['dimension'],expected['range'])
        assert not any(c['formula'] is not None for c in sheet['cells'].values())
        log['sheets'].append(dict(name=sheet['name'],animal_records=expected['animal_records'],animal_cells_checked=len(expected['record_checks']),summary_rows_removed=len(expected['summary_rows_removed']),summary_labels=expected['summary_labels'],range=expected['range'],merged_ranges=expected['merges'],group_counts=expected['group_counts'],changed_cells=mismatches,numeric_xml_variations=numeric_xml_variations))
        all_changes.extend(mismatches)
    (WORK/'verification.json').write_text(json.dumps(log,indent=2)+'\n')
    print(json.dumps(dict(sheets=[dict(name=s['name'],records=s['animal_records'],changes=len(s['changed_cells']),same_float_serializations=len(s['numeric_xml_variations'])) for s in log['sheets']],total_changes=len(all_changes)),indent=2))
    if all_changes:raise SystemExit('Export changed exact stored values. Inspect verification.json before installation.')
    log['sha256']=digest(STAGED);log['verification_passed']=True
    (WORK/'verification.json').write_text(json.dumps(log,indent=2)+'\n')


def smoke():
    import ast
    import pandas as pd
    import numpy as np
    module=ast.parse((HERE.parent/'src/mwm.py').read_text())
    fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='read_data_train')
    env={'pd':pd,'np':np}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'src/mwm.py:read_data_train','exec'),env)
    notebook=json.loads((HERE.parent/'2024-06-04-5xfad-behavior-CO28154/2024-06-04-5xfad-behavior-CO28154.ipynb').read_text())
    treatments=None
    for cell in notebook['cells']:
        source=''.join(cell.get('source',[]))
        if cell.get('cell_type')=='code' and 'treatments_CO28154 =' in source:
            for node in ast.parse(source).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='treatments_CO28154' for t in node.targets):treatments=ast.literal_eval(node.value)
    assert treatments is not None
    train=env['read_data_train'](str(STAGED),treatments,sheet_name='MWM day 1-4')
    assert train.index.names==['cohort','group','sex','irn','day']
    assert (train.index.get_level_values('sex').to_numpy()==train['Sex'].to_numpy()).all()
    assert (train.index.get_level_values('irn').to_numpy()==train['IRN'].to_numpy()).all()
    assert len(train[['Cohort','Group','Sex','IRN']].drop_duplicates())==162
    probe=pd.read_excel(STAGED,sheet_name='Probe Trial day5',index_col='IRN')
    assert len(probe)==162 and probe.Group.map(treatments).notna().all()
    y=pd.read_excel(STAGED,sheet_name='Y-Maze')
    assert len(y)==163 and y.Group.map(treatments).notna().all()
    assert {'Total entries, week 12','Total entries, week 13','Correct entries %, week 12','Correct entries %, week 13'}<=set(y.columns)
    weights=pd.read_excel(STAGED,sheet_name='body weight',header=1)
    assert len(weights)==196
    log=json.loads((WORK/'verification.json').read_text())
    log['reader_checks']=dict(MWM_reader='existing src/mwm.py read_data_train function executed unchanged',MWM_training_animal_records=162,MWM_training_long_rows=len(train),index_sex_and_irn_match_columns=True,probe_rows=162,Y_maze_rows=163,body_weight_rows=196,full_notebooks_executed=False)
    (WORK/'verification.json').write_text(json.dumps(log,indent=2)+'\n')
    print(json.dumps(log['reader_checks'],indent=2))


def install():
    log=json.loads((WORK/'verification.json').read_text())
    assert log.get('verification_passed') and digest(STAGED)==log['sha256']
    assert digest(RAW)==log['source_sha256']
    if DEST.exists() or DEST.is_symlink():raise FileExistsError(f'Refusing to overwrite {DEST}')
    fd,tempname=tempfile.mkstemp(prefix='.co28154-clean-',dir=DEST.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(STAGED.read_bytes());f.flush();os.fsync(f.fileno())
        os.chmod(tempname,0o644)
        os.link(tempname,DEST)
    finally:os.unlink(tempname)
    assert digest(DEST)==log['sha256']
    log['installed_at_utc']=datetime.now(timezone.utc).isoformat();log.pop('staged',None)
    (HERE/'co28154-cleaning-log.json').write_text(json.dumps(log,indent=2)+'\n')
    report=['# CO28154 behavioral data reconstruction','','Applied the independently verified CO28152 cleaning rules, adapted to CO28154 headers and saved notebook output. The source workbook is unchanged. No animals were excluded, measurements converted, missing values imputed, or records reordered within a sheet.','','| Sheet | Animal records retained | Summary rows removed | Cleaned range |','|---|---:|---:|---|']
    for s in log['sheets']:report.append(f"| {s['name']} | {s['animal_records']} | {s['summary_rows_removed']} | {s['range']} |")
    report+=['','## Adaptations','','- CO28154 sheet names retained, including MWM day 1-4 and Probe Trial day5, as required by its notebooks.','- Training headers retain their metric/day hierarchy; Covariates is added over the first four columns.','- MWM covariates reordered from Cohort, Group, IRN, Sex to Cohort, Group, Sex, IRN to match the saved notebook output and the reader index assumptions. Values and cell types remain unchanged.','- Y-Maze metric/week headers flattened to the established single-row names.','- The additional body weight sheet is retained and cleaned using the shared summary/padding rules. Its meaningful body weight [g] title, original column order and lowercase sex header are retained. Its twelve lowercase mean and twelve SEM rows are removed.','- Different sheet populations are retained: 162 training, 162 probe, 163 Y-Maze and 196 body-weight records. No subject intersection or group filtering was imposed.','','## Verification','','Every retained animal cell was compared to its source cell at exact stored precision and type. The source hash and output hash were checked. Header ranges, merged headers and notebook sheet names were checked. Workbook previews were reviewed. Full notebook execution and reproduction of fitted statistical results are separate work.','','Output: `'+str(DEST)+'`','','Output SHA-256: `'+log['sha256']+'`','','Raw SHA-256: `'+log['source_sha256']+'`']
    (HERE/'co28154-cleaning.md').write_text('\n'.join(report)+'\n')
    print('Installed:',DEST)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','preview-raw','build','preview-clean','verify','smoke','install']);a=p.parse_args()
    {'prepare':prepare,'preview-raw':lambda:preview(True),'build':build,'preview-clean':lambda:preview(False),'verify':verify,'smoke':smoke,'install':install}[a.action]()
