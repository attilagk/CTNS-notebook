# Filtering the recovery inventories

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
