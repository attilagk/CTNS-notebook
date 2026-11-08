# Promotion: 2024-11-11-cytokines

Promoted the fresh-kernel reproduction to the canonical
`2024-11-11-cytokines/2024-11-11-cytokines.ipynb` on branch
`promote-2024-11-11-cytokines`. The analysis cells, treatment selections,
formulas, and OLS fitting behavior are retained. Removed the trailing
`%connect_info` cell because it exposes live kernel connection details and
is unrelated to the analysis.

## Verification

- Used the notebook-declared `base` conda environment, Python 3.13.12.
- Executed all 17 executable cells in a fresh kernel from the notebook's
  expected working directory layout. No cell errors.
- The notebook produced the parameter workbook and three PNG/PDF figure pairs
  in an isolated validation workspace; the target figure visually matches the
  saved reference plot. The validation workspace is under
  `recovery-audit/reproductions/cytokines-20261107T2345/work/`.
- Compared the reconstructed input's OLS outputs with the original saved table:
  398 of 400 visible values match at displayed precision. The two differences
  are at the last printed digit (CO28154 IL-10 intercept p-value differs by
  5.25e-28; IL-2 Saline WT coefficient differs by 5.00e-7). No scientific
  choices were changed to force exact printed rounding.
- The existing analyses emit a Pandas categorical-category deprecation warning
  and statsmodels rank-deficiency warnings. These arise from the retained
  treatment category/model structure; they were not suppressed or altered.
- No fit cache applies: this notebook runs OLS, not MCMC.

The parameter workbook and rendered figures from validation remain in the
isolated workspace, not in the Git commit. The promoted notebook itself retains
its executed cell outputs, including the target estimate table and plot.

Detailed hashes, package versions, exact validation paths, and execution count
are recorded in `2024-11-11-cytokines-verification.json`.
