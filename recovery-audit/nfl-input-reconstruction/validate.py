#!/usr/bin/env python3
"""Check reconstructed inputs through the notebook's real reader and saved tables."""
import json
import math
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
import nfl


class Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.body = [], None, None, False

    def handle_starttag(self, tag, attrs):
        if tag == 'tbody': self.body = True
        if tag == 'tr' and self.body: self.row = []
        if tag in ('th', 'td') and self.row is not None: self.cell = ''

    def handle_data(self, data):
        if self.cell is not None: self.cell += data

    def handle_endtag(self, tag):
        if tag in ('th', 'td') and self.cell is not None:
            self.row.append(self.cell.strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def main():
    stage, resources, output = map(Path, sys.argv[1:])
    notebook = json.loads((REPO / '2024-05-20-nfl-modeling/2024-05-20-nfl-modeling.ipynb').read_text())
    report = {}
    for study, cell in [('CO28152', 1), ('CO28154', 2)]:
        namespace = {'pd': pd}
        code = ''.join(notebook['cells'][cell]['source'])
        exec(code.split('fpath =')[0], namespace)
        treatments = namespace['treatments_' + study]
        workbook = resources / 'CO28152_NFL-GFAP_datasum.xlsx' if study == 'CO28152' else stage / 'CO28154-biochemistry-cleaned.xlsx'
        animal_csv = stage / f'{study}-animal-list.csv'
        animals = pd.read_csv(animal_csv, dtype={'IRN': str}).set_index('IRN')
        raw = pd.read_excel(workbook, sheet_name='NFL in vivo clean', dtype={'IRN': str})
        assert raw.IRN.is_unique and animals.index.is_unique
        assert set(raw.IRN) <= set(animals.index), 'Inner join would silently lose animals'
        metadata_differences = []
        for column in ['Cohort', 'Group', 'Sex', 'Genotype']:
            indexed = raw.set_index('IRN')
            for irn in raw.IRN:
                if indexed.loc[irn, column] != animals.loc[irn, column]:
                    metadata_differences.append({'IRN': irn, 'column': column, 'nfl_value': indexed.loc[irn, column], 'animal_list_value': animals.loc[irn, column]})
        data = nfl.read_nfl_data(workbook, 'NFL in vivo clean', treatments, study, animal_csv)
        table = Table()
        table.feed(''.join(notebook['cells'][cell]['outputs'][0]['data']['text/html']))
        assert list(data.index) == [r[0] for r in table.rows]
        checked = 0
        for row in table.rows:
            for column, expected in zip(data.columns, row[1:]):
                actual = data.loc[row[0], column]
                if expected == 'NaN':
                    assert pd.isna(actual), (study, row[0], column, actual, expected)
                elif isinstance(actual, (int, float)):
                    decimals = len(expected.split('.')[1]) if '.' in expected else 0
                    assert math.isclose(actual, float(expected), rel_tol=0, abs_tol=0.500001 * 10**(-decimals)), (study, row[0], column, actual, expected)
                else:
                    assert str(actual).strip() == expected, (study, row[0], column, actual, expected)
                checked += 1
        report[study] = {'animal_list_rows': len(animals), 'joined_nfl_rows': len(data), 'saved_table_cells_matched': checked, 'join_losses': 0, 'source_metadata_disagreements_preserved': metadata_differences}
    # Independent round-trip check: every exported NFL value equals staged extraction.
    values = json.loads((stage / 'nfl-values.json').read_text())
    actual = pd.read_excel(stage / 'CO28154-biochemistry-cleaned.xlsx', dtype={'IRN': str})
    pd.testing.assert_frame_equal(actual, pd.DataFrame(values[1:], columns=values[0]), check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-12)
    report['status'] = 'passed: source extraction, XLSX round trip, metadata joins and both historical dataframes'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
