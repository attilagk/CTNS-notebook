"""Compare visible saved OLS table cells using the staged CSV; write no notebook results."""
from pathlib import Path
import html
import json
import re
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

here = Path(__file__).resolve().parent
nb = json.loads((here.parents[1] / '2024-11-11-cytokines/2024-11-11-cytokines.ipynb').read_text())
ns = dict(pd=pd, np=np, smf=smf, re=re)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    for cell in [1, 2, 3, 6, 7, 12]:
        code = ''.join(nb['cells'][cell]['source'])
        if cell == 2:
            code = code.replace("'../../resources/QPS/5xFAD-mouse/inflammation-cleaned.csv'", repr(str(Path(sys.argv[1]).resolve())))
        if cell == 12:
            code = code.split('with pd.ExcelWriter')[0]
        exec(code, ns)
report = {}
for cell, key in [(13, 'TUDCA and HCQ'), (14, 'TUDCA or HCQ')]:
    data = ns['estimatesd'][key]
    output = next(o for o in nb['cells'][cell]['outputs'] if 'text/html' in o.get('data', {}))
    markup = ''.join(output['data']['text/html'])
    failures = []
    checked = 0
    for i, tr in enumerate(re.findall(r'<tr>(.*?)</tr>', markup.split('<tbody>')[1], re.S)):
        tokens = [html.unescape(re.sub('<.*?>', '', s)).strip() for s in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
        assert len(tokens) == 21 and tokens[10] == '...'
        for j, token in enumerate(tokens):
            if token == '...':
                continue
            col = j if j < 10 else data.shape[1] - (21-j)
            value = data.iloc[i, col]
            mantissa, *exp = token.lower().split('e')
            dec = len(mantissa.split('.')[1]) if '.' in mantissa else 0
            tol = .50001 * 10**(-dec + (int(exp[0]) if exp else 0))
            if not abs(float(token)-value) <= tol:
                failures.append(dict(cytokine=data.index[i], column=list(data.columns[col]), saved=token,
                                     recovered=value, absolute_difference=abs(float(token)-value)))
            checked += 1
    report[key] = dict(visible_numeric_cells=checked, failed_at_saved_display_precision=len(failures), failures=failures)
report['warnings'] = sorted(set(str(w.message) for w in caught))
report['scope'] = 'Selected original OLS/estimate cells only, excluding result exports and plots; not full reproduction.'
(here / 'ols-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: {a: b for a, b in v.items() if a != 'failures'} if isinstance(v, dict) else v for k, v in report.items()}, indent=2))
