"""Recover the two-header cytokine CSV, checking PDF tables and notebook Out[3].

Run with base Python: reconstruct.py RESOURCE_DIR PDF_LAYOUT_TEXT STAGE_DIR
The staged CSV is installed separately, without overwriting a different file.
"""
import csv
import hashlib
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import statsmodels.formula.api as smf

RESOURCE, PDF_TEXT, STAGE = map(Path, sys.argv[1:])
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE.mkdir(parents=True, exist_ok=True)
NOTEBOOK = ROOT / '2024-11-11-cytokines/2024-11-11-cytokines.ipynb'
nb = json.loads(NOTEBOOK.read_text())
MARKERS = ['IFN-γ', 'IL-10', 'IL-12p70', 'IL-1β', 'IL-2', 'IL-4', 'IL-5', 'IL-6', 'KC/GRO', 'TNF-α']
raw = {}
rows = []
changes = []
report = {'sources': {}, 'pdf_comparison': {}, 'notebook_comparison': {}}
for study, name, count in [('CO28154', 'CO28154 rawdata biochemistry.xlsx', 60),
                           ('CO28152', 'CO28152_IV_data all.xlsx', 59)]:
    path = RESOURCE / name
    sheet = openpyxl.load_workbook(path, data_only=True)['Inflammation']
    assert list(next(sheet.iter_rows(min_row=3, max_row=3, values_only=True))[6:26]) == MARKERS * 2
    raw[study] = {}
    for rn, r in enumerate(sheet.values, 1):
        if rn <= 3:
            continue
        if r[3] is None:
            assert r[2] in (None, 'mean', 'SEM'), (rn, r)
            continue
        meta = list(r[1:6])
        meta[2] = str(meta[2])
        assert meta[2] not in raw[study]
        raw[study][meta[2]] = meta + list(r[6:26])
        clean = []
        for j, v in enumerate(r[6:26]):
            if isinstance(v, str):
                assert v.startswith('*') or v.strip() == 'n.m.', (rn, j, v)
                changes.append(dict(study=study, IRN=meta[2], region=['Cortex', 'Hippocampus'][j//10],
                                    cytokine=MARKERS[j%10], source_cell=f'{openpyxl.utils.get_column_letter(j+7)}{rn}',
                                    source_token=v, reason='starred source value' if v.startswith('*') else 'not measured'))
                v = None
            else:
                assert v is None or isinstance(v, (int, float)), v
            # Explicitly visible in saved Out[3]; the raw workbook/PDF do not flag it.
            if (study, meta[2], j) == ('CO28152', '16192', 0):
                changes.append(dict(study=study, IRN=meta[2], region='Cortex', cytokine='IFN-γ',
                                    source_cell=f'G{rn}', source_token=v,
                                    reason='blank in canonical notebook saved Out[3], despite unflagged raw value'))
                v = None
            clean.append(v)
        rows.append([study] + meta + clean)
    assert len(raw[study]) == count
    report['sources'][study] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     sheet='Inflammation', animals=count)

# Independently extract every animal record from both adjacent PDF sections.
text = PDF_TEXT.read_text()
a = text.index('9.3.1.3 Inflammation marker Cortex')
b = text.index('9.3.1.4 Inflammation marker Hippocampus', a)
c = text.index('9.3.1.5 NF-L', b)
pdf_rows = []
for region, section, offset in [('Cortex', text[a:b], 5), ('Hippocampus', text[b:c], 15)]:
    seen = set()
    flags = missing = 0
    for line in section.splitlines():
        m = re.fullmatch(r'\s*([IVX]+)\s+([A-J])\s+(\d+(?:\s+A)?)\s+([fm])\s+(tg/wt|wt/wt)\s+(.+?)\s*', line)
        if not m:
            continue
        cohort, group, irn, sex, genotype, data = m.groups()
        irn = ' '.join(irn.split())
        tokens = data.split()
        assert len(tokens) == 10, (irn, tokens)
        assert irn not in seen
        seen.add(irn)
        source = raw['CO28152'][irn]
        assert source[:5] == [cohort, group, irn, sex, genotype]
        for j, token in enumerate(tokens):
            value = source[offset+j]
            flagged = token.startswith('*')
            assert flagged == (isinstance(value, str) and value.startswith('*')), (irn, token, value)
            if token == 'n.m.':
                assert value == 'n.m.'
                missing += 1
            else:
                number = float(str(value).lstrip('*').replace(',', '.'))
                printed = float(token.lstrip('*').replace(',', '.'))
                digits = len(token.split(',')[1]) if ',' in token else 0
                assert abs(number-printed) <= .50001 * 10**(-digits), (region, irn, token, value)
            flags += flagged
        pdf_rows.append(dict(Region=region, Cohort=cohort, Group=group, IRN=irn, Sex=sex,
                             Genotype=genotype, **dict(zip(MARKERS, tokens))))
    assert seen == set(raw['CO28152']), (region, len(seen))
    report['pdf_comparison'][region] = dict(animals=len(seen), measurement_tokens=len(seen)*10,
                                           starred_tokens=flags, not_measured_tokens=missing,
                                           metadata_and_values_and_flags_match=True)
with (HERE / 'pdf-extracted-values.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(pdf_rows[0])); w.writeheader(); w.writerows(pdf_rows)

path = STAGE / 'inflammation-cleaned.csv'
with path.open('w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    # Blank index headers preserve unnamed MultiIndex levels expected by the notebook.
    w.writerow(['']*6 + ['Cortex']*10 + ['Hippocampus']*10)
    w.writerow(['']*6 + MARKERS*2)
    w.writerows(rows)

# Round trip through the exact consumer schema; validate all source values, not just previews.
wide = pd.read_csv(path, header=[0,1], index_col=list(range(6)))
assert wide.shape == (119, 20) and wide.index.is_unique
assert [tuple(map(str, ix)) for ix in wide.index] == [tuple(r[:6]) for r in rows]
assert np.allclose(wide.to_numpy(dtype=float), np.array([r[6:] for r in rows], dtype=float),
                   rtol=1e-14, atol=0, equal_nan=True)
ns = {'pd': pd, 'np': np, 'smf': smf, 're': re}
exec(''.join(nb['cells'][1]['source']), ns)
code = ''.join(nb['cells'][2]['source']).replace("'../../resources/QPS/5xFAD-mouse/inflammation-cleaned.csv'", repr(str(path)))
exec(code, ns)
assert ns['data_CO28152'].shape == (118, 18)
assert ns['data_CO28154'].shape == (120, 18)
out3 = next(o for o in nb['cells'][2]['outputs'] if o.get('execution_count') == 3)
markup = ''.join(out3['data']['text/html'])
visible_rows = numeric_cells = missing_cells = 0
rounding_boundary_differences = []
for tr in re.findall(r'<tr>(.*?)</tr>', markup, re.S):
    t = [html.unescape(re.sub('<.*?>', '', s)).strip() for s in re.findall(r'<td>(.*?)</td>', tr, re.S)]
    if not t or t[0] == '...':
        continue
    assert len(t) == 18
    key = tuple(t[2:8])
    actual = ns['data_CO28152'].loc[key]
    assert [str(actual[col]) for col in ['Treatment', 'Study', 'Cohort', 'Group', 'IRN', 'Sex', 'Genotype', 'Region']] == t[:8]
    for marker, token in zip(MARKERS, t[8:]):
        value = actual[marker]
        if token == 'NaN':
            assert pd.isna(value), (key, marker, value)
            missing_cells += 1
        else:
            # Original CSV was lost: retain source precision rather than rewriting source values.
            error = abs(float(token)-value)
            assert error <= 1e-6, (key, marker, token, value)
            if error > .50001e-6:
                rounding_boundary_differences.append(dict(IRN=t[4], region=t[7], cytokine=marker,
                                                          saved_display=token, recovered_value=value, absolute_difference=error))
            numeric_cells += 1
    visible_rows += 1
report['notebook_comparison'] = dict(visible_rows=visible_rows, numeric_cells=numeric_cells,
    missing_cells=missing_cells, absolute_tolerance=1e-6,
    rounding_boundary_differences=rounding_boundary_differences, original_shape_matched=True,
    limitation='Out[3] is truncated; hidden rows cannot be compared to the lost cleaned CSV.')
report.update(status='passed', wide_shape=list(wide.shape), long_shapes={s: list(ns['data_'+s].shape) for s in raw},
              cleaning_changes=changes, numeric_cells=int(wide.notna().sum().sum()),
              missing_cells=int(wide.isna().sum().sum()), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
              scope='Source/PDF reconciliation, CSV round trip, and notebook input cell only; not full notebook reproduction.')
(HERE / 'validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['cleaning_changes', 'sources']}, indent=2))
