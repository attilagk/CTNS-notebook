# Promotion: 2024-08-08-biochemistry

## Result

The notebook completed in a fresh `base` environment kernel after two compatibility changes. Both requested targets pass: all 66 cells in `results_params[('CO28154', 'TUDCA and HCQ')]` agree with the target table at its displayed rounding, and the TUDCA + HCQ boxplot matches the reference visually. The promoted notebook was run again from its canonical path in a fresh kernel; all 26 canonical cells completed, all seven inline plot outputs were retained, and the target checks passed again.

The target OLS design is full rank for both sexes (`n=94`, 11/11 design columns) and males only (`n=71`, 11/11 columns). No Bayesian sampling or persistent fit cache applies to this notebook.

## Reproduction record

- Run: `recovery-audit/reproductions/biochemistry-20261107T210813Z/`
- Reproduction branch started at `9406b22` (`recover-abeta-inputs`); promotion branch: `promote-2024-08-08-biochemistry`.
- Environment: conda `base`, `/home/attila/miniconda3/bin/python` 3.13.12; pandas 3.1.1, NumPy 2.4.6, SciPy 1.18.1, Matplotlib 3.10.9, seaborn 0.13.2, statsmodels 0.15.0, statannotations 0.7.2, ipykernel 7.4.0.
- `statannotations` was missing and installed into `base` from its matching PyPI package. The two recovered workbooks were present at the paths read by the notebook. Their SHA-256 values are recorded in the run manifest and match the recovery receipt.
- Target comparison: `target-comparisons.json` (66 numeric cells, zero failures; maximum absolute difference `4.991332902548606e-07`, within each cell's displayed-rounding tolerance).
- Visual acceptance is bound to the SHA-256 pair recorded in `target-comparisons.json`; the comparator reports uninspected and exits with failure if either image changes.
- Target figure: `figures/boxplot-Abeta-TUDCA-and-HCQ.png`; dimensions 475×558 px, same as the supplied reference.
- Machine-readable state: `manifest.json`, `baseline-execution.json`, `fixed-execution.json`, `verify-execution.json`, `ols-design-diagnostics.json`, and `nonestimable-ttests.csv`.

## Changes made

1. In the two regression-summary cells, changed `DataFrame.applymap` to `DataFrame.map`, which is available in the installed pandas version.
2. In the plotting helper, copied the subset before changing treatment order and used `cat.set_categories` through `assign`. Pandas 3 rejected assigning a reduced category set back into the original categorical column.
3. Removed the `%connect_info` cell because it prints live Jupyter kernel connection credentials and is not part of the analysis.
4. Kept the fresh run's displayed notebook outputs. Model formulas, animal/data selection, input measurements, tests, units, and figure comparisons were otherwise unchanged.

## Retained warnings and limits

- Ten of 88 unrelated WT t-test rows return NaN with SciPy `SmallSampleWarning` because one or both groups have too few non-missing values. These rows are listed in `nonestimable-ttests.csv`; the two target TUDCA + HCQ comparisons used for the target figure are estimable.
- Six of the 40 exploratory OLS fits are rank deficient. These are fits using the formula that includes `Sex`, including male-only subsets. The target formula excludes `Sex` and is full rank in both target subsets. No model specifications were changed to hide the warnings.
- Pandas emits a repeated future-compatibility warning when the notebook reassigns categories in `fit_one`; the operation completed, and the target estimates matched. This behavior remains documented rather than silently suppressing the warning.
- The reference image is a screenshot; the generated target image was compared visually by the 2×2 panel arrangement, treatment groups, data points, log axes, labels, and annotations. Pixel similarity metrics are descriptive, not an acceptance threshold.

## Paths and reproducibility

The recovered Excel inputs remain outside this Git repository under `../resources/QPS/5xFAD-mouse/`. Notebook-generated CSV/Excel outputs and all non-target figures are kept in the ignored run workspace and under the sibling results path when run normally. Only the target figure and compact validation records are retained in the repository. There are no posterior files to back up.

For ordinary Jupyter use, reopen the canonical notebook and choose **Revert** if Jupyter asks whether to reload the file from disk; **Overwrite** would replace the promoted file with the browser buffer.
