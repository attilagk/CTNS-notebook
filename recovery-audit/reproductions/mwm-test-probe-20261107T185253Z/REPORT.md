# Reproduction: 2024-06-26-mwm-test-probe-CO28154

- Execution completed in a fresh kernel using the target section's `base` conda environment. All notebook cells completed.
- Inputs: the cleaned CO26990, CO28152, and CO28154 behavioral workbooks. Their hashes are recorded in `manifest.json`.
- Artifact 1: the `fractions` table was regenerated from the three studies, for both sexes and male-only subsets. The displayed values align with the target artifact values, including the CO26990, CO28152, and CO28154 entries.
- Artifacts 2 and 3: `plotall-CO26990-CO28152.png` and `pie-chart-wt-tg-uniform.png` were generated. Other notebook figures were also saved in the isolated `workspace/notebooks/2024-06-26-mwm-test-probe-CO28154/named-figure/` directory.
- This analysis uses Dirichlet maximum-likelihood fits, not MCMC. The `xuod/dirichlet` 0.7 routine emitted its numerical warnings during fitting; these are preserved in the executed notebook. Matplotlib emitted a legend color override warning.
- Execution used a temporary import-path bootstrap and isolated relative paths. No scientific code or canonical notebook was changed. `%connect_info` output was removed from the saved execution copy to avoid retaining a transient connection key.

The current package inventory is included; its capture time is after the run, and the original run did not save a package lock. The external `attila_utils` path and hash are recorded. The initial branch, commit, and dirty paths were reconstructed from the session status at run start. See `manifest.json`, `conda-packages.json`, `source.ipynb`, `executed.ipynb`, and the isolated figures for provenance and output.
