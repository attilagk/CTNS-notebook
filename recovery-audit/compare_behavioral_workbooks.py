#!/usr/bin/env python3
"""Read-only raw/cleaned comparison; does not read the human rule document."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'resources/QPS/5xFAD-mouse'
RAW = DATA / 'CO28152 rawdata.xlsx'
CLEAN = DATA / 'CO28152-behavioral-data-cleaned.xlsx'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'


def colnum(label):
    n = 0
    for c in label:
        n = n * 26 + ord(c) - ord('A') + 1
    return n


def colname(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65+r) + s
    return s


def read_workbook(path):
    result = []
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.findall('.//m:t', NS)) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for s in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS):
            target = rels[s.get(REL)]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            xml = ET.fromstring(z.read(target))
            cells = {}
            for c in xml.findall('m:sheetData/m:row/m:c', NS):
                ref = c.get('r');m = re.fullmatch(r'([A-Z]+)(\d+)', ref)
                v = c.find('m:v', NS);f = c.find('m:f', NS)
                t = c.get('t', 'n')
                if t == 's':
                    value = strings[int(v.text)] if v is not None else None
                elif t == 'inlineStr':
                    value = ''.join(x.text or '' for x in c.findall('.//m:t', NS))
                elif v is None or v.text is None:
                    value = None
                elif t == 'n':
                    value = Decimal(v.text)
                else:
                    value = v.text
                cells[(int(m[2]), colnum(m[1]))] = dict(ref=ref, value=value, type=t, formula=f.text if f is not None else None, formula_attributes=f.attrib if f is not None else None, style=c.get('s'))
            result.append(dict(name=s.get('name'), dimension=xml.find('m:dimension', NS).get('ref'), cells=cells, merges=[m.get('ref') for m in xml.findall('m:mergeCells/m:mergeCell', NS)]))
    return result


def val(sheet, row, col):
    return sheet['cells'].get((row,col),{}).get('value')


def rows(sheet):
    return sorted({r for r,c in sheet['cells']})


def display(sheet, row):
    return ' | '.join(f'{colname(c)}{row}={v["value"]!r}' for (r,c),v in sorted(sheet['cells'].items()) if r == row and v['value'] is not None)


def preview():
    for p in [RAW,CLEAN]:
        print('\nWORKBOOK',p.name)
        for s in read_workbook(p):
            print('\nSHEET',s['name'],'dimension',s['dimension'],'formula cells',sum(c['formula'] is not None for c in s['cells'].values()))
            for r in rows(s)[:19]:
                text=display(s,r)
                if text:print(text[:2100])


def json_default(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(type(value).__name__)


def infer_rules():
    """Infer and verify table transformations before any human-rule comparison."""
    raw, clean = read_workbook(RAW), read_workbook(CLEAN)
    assert len(raw) == len(clean) == 3
    expected_names = [('raw data MWM day 1 - 4', 'rawdata MWM day 1-4'), ('raw data MWM day 5', 'raw data MWM day 5'), ('Y-Maze', 'Y-Maze')]
    evidence = dict(inferred_at_utc=datetime.now(timezone.utc).isoformat(), method='Independent comparison of both XLSX XML packages. Human cleaning notes were not read by this script or before writing this report. Numeric values compared using exact Decimal values; strings, types, blanks, row order, headers and merged ranges checked separately. No formulas recalculated. Styles, workbook themes, chart objects and ZIP packaging are not reproduced.', sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [RAW,CLEAN]], sheets=[])
    all_keys = []
    for i,(r,c) in enumerate(zip(raw,clean)):
        assert (r['name'], c['name']) == expected_names[i]
        base = 2 if i < 2 else 1
        raw_header_end = 3 if i == 0 else 2
        width = [24,9,8][i]
        raw_records = [row for row in rows(r) if val(r,row,base+1) in set('ABCDEFGHIJ') and val(r,row,base+2) in {'f','m'} and val(r,row,base+3) is not None]
        clean_records = [row for row in rows(c) if val(c,row,2) in set('ABCDEFGHIJ') and val(c,row,3) in {'f','m'} and val(c,row,4) is not None]
        def key(sheet,row,b):
            return tuple(val(sheet,row,col) for col in range(b,b+4))
        raw_keys = [key(r,row,base) for row in raw_records]
        clean_keys = [key(c,row,1) for row in clean_records]
        assert len(raw_records) == len(clean_records) == 145
        assert len(set(raw_keys)) == len(raw_keys)
        assert raw_keys == clean_keys
        all_keys.append(raw_keys)
        removed = [row for row in rows(r) if row > raw_header_end and row not in raw_records and any(val(r,row,col) is not None for col in range(base,base+width))]
        assert Counter(val(r,row,base) for row in removed) == {'Mean':10,'SEM':10}
        differences=[]; blanks=[]; types=Counter();numeric=0
        for rr,cr in zip(raw_records,clean_records):
            for j in range(width):
                a,b = val(r,rr,j+base),val(c,cr,j+1)
                if a != b or type(a) != type(b):
                    differences.append(dict(raw_cell=f'{colname(j+base)}{rr}',clean_cell=f'{colname(j+1)}{cr}',raw=a,clean=b))
                if j >= 4:
                    types[type(a).__name__] += 1
                    if isinstance(a,Decimal):numeric+=1
                    if a is None:blanks.append(dict(raw_cell=f'{colname(j+base)}{rr}',clean_cell=f'{colname(j+1)}{cr}',key=list(key(r,rr,base))))
        assert not differences
        # Rebuild the cleaned value grid independently from the raw workbook.
        # This operates in memory, retaining exact values and existing row order.
        predicted={}
        if i == 0:
            retained=[2,3]+raw_records
            for dstrow,srcrow in enumerate(retained,1):
                for col in range(1,width+1):predicted[(dstrow,col)]=val(r,srcrow,col+1)
            predicted[(1,1)]='Covariates'
            predicted_merges={'A1:D1','E1:H1','I1:L1','M1:P1','Q1:T1','U1:X1'}
        elif i == 1:
            retained=[2]+raw_records
            for dstrow,srcrow in enumerate(retained,1):
                for col in range(1,width+1):predicted[(dstrow,col)]=val(r,srcrow,col+1)
            predicted_merges=set()
        else:
            for col in range(1,5):predicted[(1,col)]=val(r,2,col)
            for col in range(5,9):
                parent=val(r,1,5 if col <= 6 else 7)
                predicted[(1,col)]=parent+', '+val(r,2,col)
            for dstrow,srcrow in enumerate(raw_records,2):
                for col in range(1,width+1):predicted[(dstrow,col)]=val(r,srcrow,col)
            predicted_merges=set()
        actual={(row,col):cell['value'] for (row,col),cell in c['cells'].items() if cell['value'] is not None}
        predicted_nonempty={k:v for k,v in predicted.items() if v is not None}
        assert predicted_nonempty == actual, 'Reconstructed table differs from cleaned values/headers'
        assert predicted_merges == set(c['merges'])
        formula_cells=[dict(cell=cell['ref'],formula=cell['formula'],attributes=cell['formula_attributes']) for cell in r['cells'].values() if cell['formula'] is not None]
        assert all(int(re.search(r'\d+',f['cell'])[0]) in removed for f in formula_cells)
        assert not any(cell['formula'] is not None for cell in c['cells'].values())
        evidence['sheets'].append(dict(raw_sheet=r['name'],clean_sheet=c['name'],raw_dimension=r['dimension'],clean_dimension=c['dimension'],animal_records=145,group_counts=dict(Counter(str(val(r,row,base+1)) for row in raw_records)),row_order_preserved=True,identifiers_preserved=True,source_rows_removed=removed,removed_row_labels=dict(Counter(val(r,row,base) for row in removed)),source_rows_removed_count=len(removed),raw_to_clean_row_mapping=[dict(raw_row=rr,clean_row=cr,IRN=val(r,rr,base+3)) for rr,cr in zip(raw_records,clean_records)],raw_summary_formula_cells_removed=len(formula_cells),retained_data_formulas=0,measurement_cells=145*(width-4),numeric_measurement_cells=numeric,blank_measurement_cells=len(blanks),blank_measurement_locations=blanks,covariate_cells_compared=145*4,animal_cells_compared=145*width,changed_animal_cells=differences,whole_value_grid_reconstructed=True,nonempty_cells_reconstructed=len(actual),merged_ranges_reconstructed=sorted(predicted_merges)))
    canonical_keys = [[tuple(str(v) for v in key) for key in keys] for keys in all_keys]
    assert set(canonical_keys[0]) == set(canonical_keys[1]) == set(canonical_keys[2])
    evidence['cross_sheet_key_comparison'] = 'Compared identifiers as strings only for cross-sheet matching: IRN 16006 is stored numerically in Y-Maze and as text in MWM; each cleaned sheet preserves its own raw types.'
    evidence['summary']=dict(unique_animals=145,sheets=3,animal_cells_compared=sum(x['animal_cells_compared'] for x in evidence['sheets']),numeric_measurement_cells=sum(x['numeric_measurement_cells'] for x in evidence['sheets']),blank_measurement_cells=sum(x['blank_measurement_cells'] for x in evidence['sheets']),changed_animal_cells=0,removed_summary_rows=sum(x['source_rows_removed_count'] for x in evidence['sheets']),raw_summary_formula_cells_removed=sum(x['raw_summary_formula_cells_removed'] for x in evidence['sheets']),MWM_sheets_same_row_order=all_keys[0]==all_keys[1],Y_maze_original_order_differs_from_MWM=canonical_keys[0]!=canonical_keys[2])
    (HERE/'co28152-cleaning-evidence.json').write_text(json.dumps(evidence,indent=2,default=json_default)+'\n')
    report = [
        '# Independent CO28152 behavioral cleaning rules', '',
        'This assessment was written before opening the human comparison in behavioral-data-cleaning.md. The comparison script reads only the raw and cleaned workbooks. Neither workbook was modified.', '',
        '## Result', '',
        'Cleaning changes the table layout and headers. All 145 animal records per sheet, their identifiers, and their within-sheet ordering are preserved. Across the three sheets, 5,945 animal-table cells match exactly, including 4,198 numeric measurement cells, 1,740 covariate cells, and seven blank measurement cells. No measurement conversion, rounding, correction, imputation, clipping, or animal exclusion was observed.', '',
        '| Raw sheet | Cleaned sheet | Animal rows | Removed summary rows | Measurement cells (numeric / blank) |',
        '|---|---|---:|---:|---:|',
    ]
    for x in evidence['sheets']:
        report.append(f"| {x['raw_sheet']} | {x['clean_sheet']} | 145 | 20 | {x['numeric_measurement_cells']} / {x['blank_measurement_cells']} |")
    report += [
        '', '## Shared rules', '',
        '1. Delete the ten `Mean` rows and ten `SEM` rows from each sheet. These are interleaved group summaries, not animal records. Recognize them by their row labels rather than fixed row numbers.',
        '2. Retain every animal row and its original position relative to the other animal rows. All sheets contain the same 145 composite `(Cohort, Group, Sex, IRN)` keys, but Y-Maze starts in a different order from the MWM sheets; do not impose a shared sort.',
        '3. Preserve the covariate values and cell types exactly, including Roman-numeral cohorts, letter groups, lowercase sex codes, and IRN suffixes such as `14627 A`. Do not strip the letter suffix or coerce all IRNs to integers.',
        'One pre-existing cross-sheet type difference is retained: IRN 16006 is numeric in Y-Maze and text in MWM. Cross-sheet subject matching accounts for this without editing either workbook.',
        '4. Preserve all measurements and blanks exactly. Distance remains in cm and velocity in cm/s; latency remains in seconds, including values such as 60.08. Percent-valued measurements retain their source scale. Do not replace missing Y-Maze values with zero.',
        '', '## Sheet-specific rules', '',
        '### MWM training, days 1–4', '',
        '- Rename `raw data MWM day 1 - 4` to **`rawdata MWM day 1-4`**.',
        '- Remove the blank leading row 1 and column A: the meaningful range starts at raw B2. After removing summaries, raw B2:Y168 becomes cleaned A1:X147.',
        '- Keep a two-row header. Add **Covariates** over the first four columns, merged as A1:D1. The second row remains Cohort, Group, Sex, IRN.',
        '- Keep the five metric blocks and their four Day 1–4 subcolumns in their original order: Latency (s), Distance (cm), Velocity (cm/s), Thigmotaxis %, Floating %.',
        '- Retain the metric header merges, shifted one column left and one row up: E1:H1, I1:L1, M1:P1, Q1:T1, U1:X1.',
        '', '### MWM probe, day 5', '',
        '- Keep the sheet name **raw data MWM day 5** unchanged, including its spaces.',
        '- Remove the blank leading row 1 and column A. Raw B2:J167 becomes cleaned A1:I146 after dropping the summaries.',
        '- Keep the existing single-row headers: Cohort, Group, Sex, IRN, Target Zone Crossings, SE (%), NE (%), NW (%), SW (%). Do not merge headers or rescale the quadrant percentages.',
        '', '### Y-Maze', '',
        '- Keep the sheet name **Y-Maze** unchanged. Its table already starts in column A; do not remove a leading column.',
        '- Collapse the two raw header rows into one. Retain Cohort, Group, Sex, IRN. Expand the merged metric labels and combine each with its week using comma-space:',
        '  - Total entries, week 12',
        '  - Total entries, week 13',
        '  - Correct entries %, week 12',
        '  - Correct entries %, week 13',
        '- After deleting the summaries and collapsing the header, raw A1:H167 becomes cleaned A1:H146. The cleaned sheet has no merged cells.',
        '', '## Verification and limits', '',
        'The inferred rules reconstruct every nonempty value and header in all three cleaned sheets, with matching merged ranges and dimensions. Animal cells were compared as exact stored Decimal values or strings, with type checks; this is stronger than comparing displayed rounded values. Numeric zero remains distinct from a blank.',
        'All 120 raw formula cells are in removed summary rows (20 in training, 100 in probe, zero in Y-Maze). Neither workbook contains formulas in retained animal cells. This does not require recomputing any animal measurement.',
        'This reconstructs the analytical tables and headers, not workbook formatting, styles, themes, package bytes, or editing history. The exact original sequence of manual Excel actions cannot be inferred uniquely.',
        'No new cleaned workbook has been generated. Before adapting these rules to CO28154, inspect its own sheet layout and header wording; do not copy CO28152 row numbers, week labels, animal counts, or sheet names blindly.',
        '', '## Source fingerprints', '',
    ]
    for source in evidence['sources']:report += ['- `'+source['path']+'`', '  SHA-256: `'+source['sha256']+'`']
    report += ['', 'Detailed row mappings, removed row numbers, and the seven retained blank locations are in co28152-cleaning-evidence.json.']
    (HERE/'co28152-cleaning-independent.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(evidence['summary'],indent=2))
    print('Independent report SHA-256:',hashlib.sha256((HERE/'co28152-cleaning-independent.md').read_bytes()).hexdigest())
    print('Blank Y-Maze locations:',json.dumps(evidence['sheets'][2]['blank_measurement_locations'],default=json_default))


if __name__ == '__main__':
    infer_rules()
