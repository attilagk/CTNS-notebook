# 2024-01-05-5xfad-nfl-frequentist: execution-only rerun and promotion

Both isolated reproduction and promoted-source fresh-kernel verification passed in `/home/attila/miniconda3/bin/python` (base), with 20 executable cells each. Reference artifacts were not compared, as explicitly requested. No scientific reproduction claim is made. No MCMC or persistent posterior cache applies.

The only notebook source change removes `%connect_info`; all scientific cells, selections, formulas, and units are preserved. Executed outputs replace old outputs. Retained warnings: two existing regex strings use non-raw `\W` escape syntax, generating Python SyntaxWarnings. No execution errors occurred.

Each execution used a separate kernel and the same isolated mirrored work directory. Inputs were read from existing resources; all result CSVs and figures remain under the ignored `work/` directory. Canonical results and raw inputs were not overwritten. The promoted notebook has no bootstrap cells or temporary run paths in source and runs using ordinary repository paths. Manifests record source/input/helper identities, package versions, interpreter and Git state. Verification source hash matches promoted notebook before execution. Original source is retained by Git history.

Evidence: `recovery-audit/reproductions/rerun-2024-01-05-5xfad-nfl-frequentist-20261108/`. Worktree: `/tmp/ctns-rerun-20261108/nfl-jan`. Promotion branch: `promote-2024-01-05-5xfad-nfl-frequentist`. Local commit only; no push, PR, or merge.
