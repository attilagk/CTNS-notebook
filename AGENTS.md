# CTNS notebook recovery and reproduction

## Project layout and sources of truth

- This Git repository contains notebooks and shared Python helpers in `src/`. Scientific inputs and generated results normally live in the sibling `../resources/` and `../results/` directories. Resolve paths from the notebook's expected working directory.
- Read the exact notebook section in `recovery-audit/target-artifacts/target-artifacts.md` for its target tables/figures **and its `conda env:` declaration**. Do not assume every notebook uses `bayes`; several use `base`. If the declaration is missing or contradictory, resolve it before execution. Preserve the user's reference values, images, and environment declarations.
- Read the selected reproduction's manifest, report, validation status, and diagnostics for current evidence. Workflow instructions are not a record of which notebooks currently pass.
- Use `$ctns-reproduce-notebook` for execution/validation and `$ctns-promote-notebook` for replacing a canonical notebook with a validated reproduction. Installed skill entrypoints are `.agents/skills/ctns-reproduce-notebook/SKILL.md` and `.agents/skills/ctns-promote-notebook/SKILL.md`.

## Scientific and recovery invariants

- Distinguish execution completion, agreement with reference artifacts, and inference quality. A successful Run All alone does not establish reproduction. Report unresolved discrepancies and warnings explicitly; never change scientific choices simply to match a reference.
- Preserve input values, model formulas, priors, censoring, animal selection, seeds, and sampling settings unless the requested correction requires a change. Explain and record any such change; ask about ambiguous scientific intent.
- CO26990 behavioral distances/velocities are meter based in the cleaned source workbook; CO28152 and CO28154 are centimeter based. The shared reader in `src/mwm.py` normalizes these to centimeters and centimeters per second. Avoid double conversion and retain correct exported figure labels.
- Treat `recovery-audit/baseline/` as immutable before-recovery evidence. Preserve historical columns and manual observations when refreshing inventory; use `refresh_inventory.py` and the CSV filter guide rather than replacing history with present-day existence checks. Keep reproduction copies and temporary workspaces out of canonical notebook counts.
- `recovery-audit/CTNS-notebooks-recovery-audit-REPORT.md` is a symlink to the professional vault. Inspect its target before writing; do not replace the symlink or recreate the old `REPORT.md` accidentally.
- Keep raw inputs and expensive posterior draws intact. Cache validity needs provenance and identity checks, not just file existence. Reuse `src/mcmc_cache.py` where appropriate; inspect its supported model identity before extending it.

## Working conventions

- Inspect Git status before edits and preserve unrelated user changes. Use a dedicated branch for a reproduction/promotion when requested or when preparing a reviewable change; do not overwrite another branch's work.
- Execute in a fresh kernel using the declared environment's Python, with explicit import paths for this repository's `src/` and `/home/attila/src` when needed. A kernelspec called `python3` is not evidence of the correct interpreter.
- Isolate exploratory execution outputs in a timestamped reproduction directory. Record the source/input/helper hashes, Git revision and dirty state, interpreter/package versions, execution log, artifact comparisons, and inference diagnostics. Reuse existing runners as examples after checking their notebook-specific assumptions.
- Keep large NetCDF caches, transient executed copies, and duplicate workspaces out of Git. Commit compact provenance, comparison reports, necessary code, and selected target artifacts using narrow ignore exceptions. Do not change broad ignore rules just to include one figure.
- Promotion is a separate task. Replace the canonical notebook only after validation; use Git history instead of an archive unless requested. Persistent cache files outside Git require separate backup and must not be described as backed up merely because the notebook was committed.
- Notify the user before modifying a notebook they may have open: Jupyter **Revert** loads disk changes; **Overwrite** writes the browser's copy over disk changes. Never discard unsaved work.
- Commit, push, PR comments, merge, and pull follow the user's authorization for the current task. A skill does not itself authorize these actions. No review response is not an approval; bind a successful review to the latest PR head.
