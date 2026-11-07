# Reproduction: 2023-11-07-5xfad-maze-test-probe

- Execution: completed in a fresh kernel using the notebook target section's `base` environment.
- Reference comparison: the two Dirichlet uniformity tests match the target text exactly at its displayed precision.
  - WT: likelihood-ratio statistic 14.3535; p = 0.00246142; null and alternative MLE parameters match.
  - 5xFAD + Amiloride: likelihood-ratio statistic 3.68336; p = 0.297748; null and alternative MLE parameters match.
- Figures: three group Dirichlet plots and four other requested notebook plots were generated under `workspace/notebooks/2023-11-07-5xfad-maze-test-probe/named-figure/`.
- Inference quality: not an MCMC notebook. The Dirichlet fit emitted divide-by-zero / invalid-value warnings in the package's Newton solver, as did the saved historical output. The reported target statistics matched despite these warnings; they remain documented.
- Dependency resolution: installed `xuod/dirichlet` commit `78ce7c7cf45af75d473ae80d55cc2b6f4caadf78` (version 0.7). It exposes the `dirichlet.plot` API expected by the notebook. The earlier `ericsuh/dirichlet` install was removed before installing this package.
- Scope: canonical notebook and scientific code were not modified.

See `manifest.json`, `executed.ipynb`, and the generated figures for provenance and outputs. The first two failed attempts are preserved in their own timestamped directories.
