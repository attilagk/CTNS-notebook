# 2024-02-29-Y-maze: execution-only rerun and promotion

Both isolated reproduction and promoted-source fresh-kernel verification passed in `/home/attila/miniconda3/bin/python` (base), with 11 executable cells each. Reference artifacts were not compared, as explicitly requested; existing Y-maze target definitions remain unchanged. No scientific reproduction claim is made. No MCMC or posterior cache applies.

The only source change removes `%connect_info`; scientific cells, selections, computations, and units are preserved. Executed outputs replace old outputs. Five existing Seaborn FutureWarnings about automatic grayscale interpretation were retained. No execution errors occurred.

Executions used distinct fresh kernels and the same isolated mirrored work directory. Inputs were read from existing resources, with figures retained under ignored `work/`; canonical results and raw inputs were not overwritten. No temporary run paths or bootstrap instrumentation appear in promoted notebook source. Manifests retain source/input/helper hashes, interpreter/package versions, and Git state. Original source is retained in Git history.

Evidence: `recovery-audit/reproductions/rerun-2024-02-29-Y-maze-20261108/`. Worktree: `/tmp/ctns-rerun-20261108/ymaze`. Promotion branch: `promote-2024-02-29-Y-maze`. Local commit only; no push, PR, or merge.

PR review follow-up: repeated reproduction and promotion verification in fresh base kernels; both passed. Each manifest now records the actual kernel-imported attila_utils module path and SHA-256. The provenance probe is removed before saving executed notebooks. No scientific source changes.
