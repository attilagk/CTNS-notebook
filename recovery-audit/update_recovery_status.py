#!/usr/bin/env python3
"""Refresh recovery columns in the original CSVs, preserving audit observations.

Run: python3 recovery-audit/update_recovery_status.py
Does not rerun notebooks, rewrite the original audit report, or modify data files.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

HERE=Path(__file__).resolve().parent
EXTRA=['recovery_status','recovery_method','recovery_note','recovery_checked_at']


def read(name):
    with (HERE/name).open(newline='') as f:
        reader=csv.DictReader(f)
        return list(reader.fieldnames),list(reader)


def save(name,fields,rows):
    fd,tmp=tempfile.mkstemp(prefix='.'+name+'-',dir=HERE)
    try:
        with os.fdopen(fd,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        os.chmod(tmp,(HERE/name).stat().st_mode & 0o777)
        os.replace(tmp,HERE/name)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main():
    checked=datetime.now(timezone.utc).isoformat()
    tables={name:read(name) for name in ['input-inventory.csv','missing-files.csv','output-producers.csv','notebook-coverage.csv']}
    original_missing={r['resolved_path'] for r in tables['missing-files.csv'][1]}
    receipts={}
    restore=HERE/'manuscript-restoration.json'
    if restore.exists():
        for record in json.loads(restore.read_text())['files']:
            if record.get('action') in {'restored','existing identical file retained'}:
                receipts[record['destination']]=dict(method='restored from Dropbox',hash=record['sha256'],log=restore.name)
    reconstructed=HERE/'co28154-cleaning-log.json'
    if reconstructed.exists():
        record=json.loads(reconstructed.read_text())
        receipts[record['destination']]=dict(method='reconstructed from raw data',hash=record['sha256'],log=reconstructed.name)
    cache={}
    def status(path):
        if path in cache:return cache[path]
        result=dict(recovery_status='',recovery_method='',recovery_note='',recovery_checked_at=checked)
        receipt=receipts.get(path)
        if not path:
            result.update(recovery_status='unresolved',recovery_note='No concrete file path was resolved by the audit.')
        elif re.match(r'^[A-Za-z]+://',path):
            result.update(recovery_status='remote not checked',recovery_note='Remote resource availability was not checked.')
        elif glob.has_magic(path):
            matches=glob.glob(path)
            result.update(recovery_status='pattern matches' if matches else 'still missing',recovery_note=f'Pattern matches {len(matches)} local paths; individual members require review.')
        elif Path(path).is_file():
            if receipt:
                matches=hashlib.sha256(Path(path).read_bytes()).hexdigest()==receipt['hash']
                result.update(recovery_status='recovered',recovery_method=receipt['method'],recovery_note=f"Logged in {receipt['log']}; "+('current SHA-256 matches recovery log.' if matches else 'current contents differ from recovery log; review changes.'))
            elif path in original_missing:
                result.update(recovery_status='recovered',recovery_method='file appeared since original audit',recovery_note='Present at the exact expected path; origin and contents not verified.')
            else:
                result.update(recovery_status='present',recovery_note='Present at the expected path; not recorded as a recovered missing file.')
        elif Path(path).is_dir():
            result.update(recovery_status='directory present',recovery_note='Directory exists; required members are not established by this check.')
        else:
            result.update(recovery_status='still missing',recovery_note='File is absent at the exact expected path. A candidate or raw precursor does not count as recovered.')
            if receipt:
                result['recovery_method']=receipt['method']
                result['recovery_note']='Previously recovered according to '+receipt['log']+', but the expected path is currently missing.'
        cache[path]=result
        return result
    for name in ['input-inventory.csv','missing-files.csv','output-producers.csv']:
        fields,rows=tables[name]
        for field in EXTRA:
            if field not in fields:fields.append(field)
        for row in rows:row.update(status(row['resolved_path']))
        save(name,fields,rows)
    by_notebook=defaultdict(list)
    for row in tables['input-inventory.csv'][1]:by_notebook[row['notebook']].append(row)
    fields,coverage=tables['notebook-coverage.csv']
    count_fields=['recovered_paths','still_missing_paths','still_missing_source_paths','still_missing_generated_paths','current_unresolved_records','recovery_checked_at']
    for field in count_fields:
        if field not in fields:fields.append(field)
    for row in coverage:
        inputs=by_notebook[row['notebook']]
        recovered={r['resolved_path'] for r in inputs if r['recovery_status']=='recovered'}
        missing={r['resolved_path'] for r in inputs if r['recovery_status']=='still missing'}
        source={r['resolved_path'] for r in inputs if r['recovery_status']=='still missing' and '/resources/' in r['resolved_path']}
        generated={r['resolved_path'] for r in inputs if r['recovery_status']=='still missing' and '/results/' in r['resolved_path']}
        row.update(recovered_paths=len(recovered),still_missing_paths=len(missing),still_missing_source_paths=len(source),still_missing_generated_paths=len(generated),current_unresolved_records=sum(r['recovery_status']=='unresolved' for r in inputs),recovery_checked_at=checked)
    save('notebook-coverage.csv',fields,coverage)
    missing_rows=tables['missing-files.csv'][1]
    counts=Counter(r['recovery_status'] for r in missing_rows)
    study_rows=[r for r in missing_rows if r['manuscript']=='yes' and '/resources/' in r['resolved_path']]
    summary=dict(recovery_checked_at=checked,original_missing_paths=len(missing_rows),original_missing_paths_by_recovery_status=dict(counts),manuscript_source_paths_by_recovery_status=dict(Counter(r['recovery_status'] for r in study_rows)),manuscript_source_paths_still_missing=[r['resolved_path'] for r in study_rows if r['recovery_status']=='still missing'],files_updated=list(tables),unchanged_loading_sites='loading-sites.csv describes code coverage rather than file availability; use input-inventory.csv for recovery filters.')
    (HERE/'recovery-status-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    guide='''# Filtering the recovery inventories

The CSVs retain their original audit observations and now have current recovery fields. `missing-files.csv` is the original list of missing paths, including paths since recovered; no historical rows have been dropped.

## Main filters

- `recovery_status = still missing`: file remains absent at the expected notebook path.
- `recovery_status = recovered`: previously missing file now exists, either restored from Dropbox or reconstructed from raw data.
- `recovery_status = present`: existing file not recorded as a recovered missing input.
- `recovery_status = unresolved`: audit could not establish a concrete path.

Other statuses are `remote not checked`, `directory present`, and `pattern matches`; these should not be interpreted as verified data files.

Start with **missing-files.csv** for one row per originally missing path. Add `manuscript = yes` for the selected study. Add `role = source data / resource` to focus on source inputs rather than generated results or caches.

Use **input-inventory.csv** to see which notebook and cell uses a file; shared inputs appear more than once. Its original `status` column remains the initial audit result. Filter the new `recovery_status` column for current availability.

Use **output-producers.csv** to check current availability of generated files. A missing generated file may be reproducible from its notebook; its absence does not imply that a source backup is required.

Use **notebook-coverage.csv** for the added `recovered_paths`, `still_missing_paths`, `still_missing_source_paths`, `still_missing_generated_paths`, and `current_unresolved_records` counts. The original `missing_paths` and `unresolved_records` remain historical.

`recovery_method` distinguishes `restored from Dropbox` and `reconstructed from raw data`. `recovery_note` states whether a recovered file matches the logged SHA-256. `recovery_checked_at` is the UTC check time.

`loading-sites.csv` is unchanged: it records code operations, not file paths or file availability. A code site's `coverage` is not a recovery status.

Refresh only these recovery columns with:

```sh
python3 recovery-audit/update_recovery_status.py
```

This does not execute notebooks, change source data, or update the report in professional-vault. Graphs and original audit counts remain snapshots until separately regenerated.
'''
    (HERE/'CSV-filter-guide.md').write_text(guide)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
