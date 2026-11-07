# Baseline and recovery progress

Run `python3 recovery-audit/refresh_inventory.py` (the old update_recovery_status.py command delegates to it). This performs static analysis and local availability/hash checks, not notebook execution or a fresh Dropbox search.

## Historical evidence

`baseline/` is anchored to commit 17e477f and the November 4 audit. Its manifest checks hashes before refresh. The original 241 missing paths remain in missing-files.csv; original input status and coverage counts remain historical. summary.json stays unchanged. Output availability absent from historical observations is unknown. Archived graph filesystem observations are separate from the audit baseline.

## Current views

- Original top-level CSVs retain their original rows and fields, plus explicit initial_status/current_status, recovery status, validation, timestamps, IDs and evidence links.
- `current/` contains a fresh static audit, including newly discovered and removed references. Its status columns describe this scan, not the initial audit.
- `files.csv` is the unique file registry, including baseline and current identities. Filter study=True for the selected studies. A missing initial observation means unknown, never presumed missing. Recovery and validation are independent.
- `dependencies.csv` records baseline/current consumes and produces links, unresolved references, alias relationships and reconstruction provenance. Current cell IDs are included when available. Dynamic or ambiguous identities still require review; unresolved records have no confirmed file_id.
- `recovery-status-summary.json` separates baseline counts from current counts and unique-file recovery totals. Counts of files differ from counts of loading references.
- `dependency-state.json` is the graph interchange artifact. It includes original graph IDs and positions where matched, filenames, availability and provenance. Use the original graph topology and positions for fair before/after overlays; do not conflate dependency changes with recovery.
- `snapshots/` stores timestamped views and hash manifests. `recovery-events.jsonl` records distinct observed status combinations without duplicating identical events. It is an observation journal, not an exact chronology of physical restoration actions; restoration receipts provide those dates.

The archived manual observations preserve human CSV changes. Files present without recovery receipts remain unvalidated, and hashes differing from logged receipts are marked changed since validation. The report symlink in professional-vault and existing graphs are not rewritten.

Study scope uses the nine distinct directories listed in results/study-dirs, with all their references, including exploratory notebook sections. Fine-grained scientific relevance remains to be curated before drawing the detailed canvas. Dropbox candidates are historical candidates, not proof of recovery. No tldraw canvas is created by the refresh.
