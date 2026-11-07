#!/usr/bin/env python3
"""Extract notebook-compatible inputs; never overwrite a different existing file.

Usage: python reconstruct.py SOURCE_DIRECTORY STAGING_DIRECTORY
Sources: Animal Lists NIA in vivo projects.xlsx; CO28154 rawdata biochemistry.xlsx.
The XLSX is authored separately by build-workbook.mjs from nfl-values.json.
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import openpyxl


def write_new(path, content):
    if path.exists():
        assert path.read_bytes() == content, f"Refusing to overwrite {path}"
    else:
        path.write_bytes(content)


def main():
    source, stage = map(Path, sys.argv[1:])
    stage.mkdir(parents=True, exist_ok=True)
    animals = openpyxl.load_workbook(source / 'Animal Lists NIA in vivo projects.xlsx', data_only=True)
    report = {'animal_lists': {}, 'date_normalizations': [], 'flagged_measurements': []}
    for study in ('CO28152', 'CO28154'):
        sheet = animals[study]
        headers = [c.value for c in sheet[1]][1:] + ['Lifespan (days)']
        records = []
        for row in list(sheet.values)[1:]:
            if row[4] is None:
                assert not any(row), 'Unexpected nonempty row without IRN'
                continue
            values = list(row[1:])
            values[3] = str(values[3])  # IRN, including any letter suffix.
            birth, death = values[2], values[7]
            if death == '04:07.2023':
                report['date_normalizations'].append({'study': study, 'IRN': values[3], 'source': death, 'normalized': '2023-07-04'})
                death = datetime(2023, 7, 4)
            assert isinstance(birth, datetime) and isinstance(death, datetime)
            lifespan = (death - birth).days
            assert lifespan >= 0
            values[2], values[7] = birth.date().isoformat(), death.date().isoformat()
            records.append(values + [lifespan])
        assert len({r[3] for r in records}) == len(records)
        # CSV writer preserves identifiers, literal N/A, whitespace and history text.
        import io
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(headers)
        writer.writerows(records)
        write_new(stage / f'{study}-animal-list.csv', stream.getvalue().encode())
        report['animal_lists'][study] = {'rows': len(records), 'columns': headers}

    raw = openpyxl.load_workbook(source / 'CO28154 rawdata biochemistry.xlsx', data_only=True)
    headers = ['Cohort', 'Group', 'IRN', 'Sex', 'Genotype'] + [f'NF-L week {w}' for w in (0, 4, 8, 12)]
    records = []
    for row_number, row in enumerate(raw['NF-L invivo'].values, 1):
        if row_number <= 3:
            continue
        if row[3] is None:
            assert row[2] in (None, 'mean', 'SEM'), f'Unexpected summary row {row_number}'
            continue
        values = list(row[1:10])
        values[2] = str(values[2])
        for col in range(5, 9):
            value = values[col]
            if isinstance(value, str) and (value.startswith('*') or value == 'n.m.'):
                report['flagged_measurements'].append({'source_cell': f'{openpyxl.utils.get_column_letter(col+2)}{row_number}', 'IRN': values[2], 'column': headers[col], 'source_value': value, 'cleaned_value': None})
                values[col] = None
            else:
                assert isinstance(value, (int, float)), (row_number, col, value)
        records.append(values)
    assert len(records) == 60 and len({r[2] for r in records}) == 60
    report['nfl_rows'] = len(records)
    report['scope'] = 'NFL in vivo clean only, as consumed by the May 20 and January 5 notebooks; other biochemical assays are not reconstructed.'
    write_new(stage / 'nfl-values.json', (json.dumps([headers] + records, indent=2) + '\n').encode())
    write_new(stage / 'transformations.json', (json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
