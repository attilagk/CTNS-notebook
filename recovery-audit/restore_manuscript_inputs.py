#!/usr/bin/env python3
"""Stage/validate five manuscript backups, then install without overwriting.

Temporary Dropbox URLs belong in an untracked /tmp manifest. They are never
written to the permanent recovery log. This script does not execute notebooks.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import posixpath
import tempfile
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
DEST = HERE.parent.parent / 'resources/QPS/5xFAD-mouse'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
REQ = {
    'CO26990-MWM-mean-distance-to-platform.xlsx': {'cleaned': (1, ['IRN', 'Mean distance (cm)'])},
    'CO28152-MWM-mean-distance-to-platform.xlsx': {'cleaned': (1, ['IRN', 'Group', 'Distance to Platform (cm)'])},
    'CO28152_NFL-GFAP_datasum.xlsx': {
        'NFL in vivo clean': (1, ['Cohort', 'Group', 'IRN', 'Sex', 'Genotype', 'NF-L week 0']),
        'NFL terminal clean': (1, ['Cohort', 'Group', 'IRN', 'Sex', 'Genotype', 'NF-L terminal plasma', 'NF-L terminal CSF']),
        'GFAP plasma clean': (1, ['Group', 'IRN', 'Sex', 'Genotype']),
    },
    'CO26990-behavioral-data-cleaned.xlsx': {
        'rawdata MWM day 1-4': (2, ['Covariates', 'Cohort', 'Group', 'Sex', 'IRN', 'Latency (s)', 'Velocity (m/s)', 'Thigmotaxis %', 'Floating %', 'Distance (m)']),
        'Y-Maze': (1, ['Group', 'Correct Entries %']),
    },
    'CO28152-behavioral-data-cleaned.xlsx': {
        'rawdata MWM day 1-4': (2, ['Covariates', 'Cohort', 'Group', 'Sex', 'IRN', 'Latency (s)', 'Velocity (cm/s)', 'Thigmotaxis %', 'Floating %', 'Distance (cm)']),
        'Y-Maze': (1, ['Group', 'Correct entries %, week 12']),
    },
}


def hashes(data):
    blocks = [hashlib.sha256(data[i:i + 4*1024*1024]).digest() for i in range(0, len(data), 4*1024*1024)]
    return {'sha256': hashlib.sha256(data).hexdigest(), 'dropbox_content_hash': hashlib.sha256(b''.join(blocks)).hexdigest()}


def workbook(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None, 'ZIP integrity failure'
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//m:t', NS)) for item in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        rels = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets = {}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet', NS):
            target = rels[sheet.attrib[REL]]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            data = ET.fromstring(z.read(target))
            rows = []
            for row in data.findall('m:sheetData/m:row', NS):
                vals = []
                for cell in row.findall('m:c', NS):
                    value = cell.find('m:v', NS)
                    if cell.get('t') == 's':
                        text = strings[int(value.text)] if value is not None else ''
                    elif cell.get('t') == 'inlineStr':
                        text = ''.join(t.text or '' for t in cell.findall('.//m:t', NS))
                    else:
                        text = value.text if value is not None else ''
                    vals.append(text)
                rows.append({'row': int(row.get('r')), 'values': vals})
            sheets[sheet.attrib['name']] = dict(nonempty_rows=sum(any(v not in {'', None} for v in r['values']) for r in rows), header_rows=[r for r in rows if r['row'] <= 2])
        return sheets


def validate(path, metadata):
    data = path.read_bytes()
    assert len(data) == metadata['size'], 'Byte count differs from Dropbox metadata'
    digests = hashes(data)
    if metadata.get('content_hash'):
        assert digests['dropbox_content_hash'] == metadata['content_hash'], 'Dropbox content hash mismatch'
    sheets = workbook(path)
    checks = []
    for name, (nrows, headers) in REQ[path.name].items():
        if name not in sheets:
            checks.append(dict(sheet=name, passed=False, missing_sheet=True));continue
        values = {v for row in sheets[name]['header_rows'] if row['row'] <= nrows for v in row['values']}
        missing = [h for h in headers if h not in values]
        checks.append(dict(sheet=name, passed=not missing, missing_headers=missing, nonempty_rows=sheets[name]['nonempty_rows']))
    warnings = []
    if path.name == 'CO28152-behavioral-data-cleaned.xlsx':
        warnings.append('The backup uses Distance (cm) and Velocity (cm/s). src/mwm.py and the older notebook rename list use Distance (m) and Velocity (m/s). Covariates, daily latency, thigmotaxis, floating, and Y-Maze headers are present. No values, units, or headers were changed. Review this compatibility difference before interpreting distance/velocity results.')
    return dict(size=len(data), **digests, sheets=sheets, checks=checks, warnings=warnings, validation_passed=all(c['passed'] for c in checks))


def stage(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    entries = manifest['entries']
    assert {e['name'] for e in entries} == set(REQ), 'Unexpected file set'
    staging = Path(tempfile.mkdtemp(prefix='ctns-manuscript-restore-', dir='/tmp'))
    log = dict(staged_at_utc=datetime.now(timezone.utc).isoformat(), staging=str(staging), files=[])
    for entry in entries:
        name = entry['name']
        path = staging / name
        # Single-use URL: exactly one GET, without HEAD or preview requests.
        with urllib.request.urlopen(entry['download_url'], timeout=60) as response:
            data = response.read(10 * 1024 * 1024 + 1)
        assert len(data) <= 10 * 1024 * 1024, 'Unexpectedly large workbook'
        path.write_bytes(data)
        record = dict(name=name, dropbox_path=entry.get('path_display') or entry['path'], dropbox_id=entry.get('id'), destination=str(DEST / name), **validate(path, entry))
        log['files'].append(record)
    summary = staging / 'stage-summary.json'
    summary.write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps(dict(staging_summary=str(summary), files=[dict(name=f['name'], validation_passed=f['validation_passed'], checks=f['checks']) for f in log['files']]), indent=2))
    if not all(f['validation_passed'] for f in log['files']):
        raise SystemExit('Workbook expectations differ; nothing installed. Inspect staged checks first.')


def revalidate(summary_path):
    path = Path(summary_path)
    log = json.loads(path.read_text())
    for record in log['files']:
        metadata = dict(size=record['size'], content_hash=record['dropbox_content_hash'])
        record.update(validate(Path(log['staging']) / record['name'], metadata))
    path.write_text(json.dumps(log, indent=2) + '\n')
    print(json.dumps([dict(name=r['name'], validation_passed=r['validation_passed'], warnings=r['warnings']) for r in log['files']], indent=2))
    assert all(r['validation_passed'] for r in log['files'])


def install(summary_path):
    log = json.loads(Path(summary_path).read_text())
    assert {f['name'] for f in log['files']} == set(REQ)
    assert all(f['validation_passed'] for f in log['files'])
    staging = Path(log['staging'])
    for record in log['files']:
        path = staging / record['name']
        assert hashes(path.read_bytes())['sha256'] == record['sha256'], 'Staged file changed'
        destination = DEST / record['name']
        assert str(destination) == record['destination']
        if destination.exists() or destination.is_symlink():
            record['action'] = 'existing identical file retained' if destination.is_file() and hashes(destination.read_bytes())['sha256'] == record['sha256'] else 'conflict; existing path retained'
            continue
        # Atomic link into the destination, with no-overwrite semantics.
        fd, tempname = tempfile.mkstemp(prefix='.ctns-restore-', dir=DEST)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
            os.chmod(tempname, 0o644)
            try:
                os.link(tempname, destination)
                record['action'] = 'restored'
            except FileExistsError:
                record['action'] = 'conflict; destination appeared during restoration'
        finally:
            os.unlink(tempname)
        if record['action'] == 'restored':
            assert hashes(destination.read_bytes())['sha256'] == record['sha256']
    log['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    # Staging location and download URLs are not needed in the durable log.
    log.pop('staging', None)
    (HERE / 'manuscript-restoration.json').write_text(json.dumps(log, indent=2) + '\n')
    report = ['# Manuscript input restoration', '', 'Five exact-name Dropbox backups were downloaded and checked against Dropbox content hashes and notebook sheet/header expectations. No notebooks were executed; data values were not changed. Passing these checks does not yet establish reproduced scientific results.', '', '| File | Action | Required sheets/headers | SHA-256 |', '|---|---|---|---|']
    for f in log['files']:
        report.append(f"| {f['name']} | {f['action']} | passed | `{f['sha256']}` |")
    report += ['', '## Compatibility notes', '']
    for record in log['files']:
        for warning in record.get('warnings', []):report.append('- **' + record['name'] + '**: ' + warning)
    report += ['', 'Destination: `' + str(DEST) + '`', '', '## Remaining missing source paths in the nine manuscript directories', '']
    import csv
    with (HERE / 'missing-files.csv').open() as f:
        for row in csv.DictReader(f):
            p = Path(row['resolved_path'])
            if row['manuscript'] == 'yes' and '/resources/' in str(p) and not p.is_file():
                report.append('- `' + str(p) + '`')
    report += ['', 'CO28154 raw data already exist locally; reconstruction of its cleaned workbook is deferred until the independent raw/cleaned comparison. The amiloride workbook alias, biochemistry inputs, exploratory sample data, and missing generated results require further work. Existing audit and graph files remain historical snapshots. The professional-vault report was not modified.']
    (HERE / 'manuscript-restoration.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(files=[{'name': f['name'], 'action': f['action']} for f in log['files']], log=str(HERE / 'manuscript-restoration.json')), indent=2))
    if any('conflict' in f['action'] for f in log['files']):
        raise SystemExit('Some destinations were retained because of conflicts; see log.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'revalidate', 'install'])
    parser.add_argument('manifest_or_summary')
    args = parser.parse_args()
    {'stage': stage, 'revalidate': revalidate, 'install': install}[args.action](args.manifest_or_summary)
