# Latency notebook reproduction

Execution: **completed**. Elapsed: 102.8 seconds.

Fresh bayes kernel; original notebook cells executed unchanged. Figure writes were redirected to this run directory. Each completed fit was immediately saved as NetCDF. No original figures or notebook outputs were overwritten.

## Target 1: day_summary

Numerical differences are reported for review; no acceptance tolerance has been approved. The original MCSEs are rounded, including values printed as zero. ESS and R-hat need not reproduce exactly.

| Experiment | Reference mean | New mean | Difference | Reference SD | New SD |
|---|---:|---:|---:|---:|---:|
| Amiloride 10 | -0.292 | -0.293 | -0.001 | 0.030 | 0.030 |
| Amiloride 12 | -0.230 | -0.230 | +0.000 | 0.024 | 0.025 |
| HCQ | -0.239 | -0.238 | +0.001 | 0.023 | 0.023 |
| HCQ nano | -0.221 | -0.221 | +0.000 | 0.024 | 0.024 |
| R-HCQ | -0.219 | -0.219 | +0.000 | 0.029 | 0.029 |
| S-HCQ | -0.238 | -0.238 | +0.000 | 0.029 | 0.030 |
| TUDCA | -0.216 | -0.216 | +0.000 | 0.023 | 0.023 |
| TUDCA WT | -0.235 | -0.234 | +0.001 | 0.023 | 0.023 |

Full numerical comparison: day-summary-comparison.csv.

## Target 2: condition-posteriors-study1.png

Generated PNG and PDF are in figures/. The reference screenshot is Pasted image 20261106213159.png in target-artifacts. Figure posterior values below use the same 95% HDI and ROPE [-0.1, 1] as the plotting cell.

- WT: mean -0.2581; 95% HDI [-0.4989, -0.0261]; ROPE 9.72%.
- 5xFAD + Amiloride: mean 0.0724; 95% HDI [-0.1880, 0.3233]; ROPE 91.03%.

## Sampling diagnostics

10 fits saved; total divergences: 0. Largest reported R-hat: 1.01; smallest reported bulk ESS: 1124. These use rounded ArviZ summaries; inspect fit-diagnostics.jsonl and full saved fits for details.

## Provenance and reuse

manifest.json records source/input hashes, Git revision and timing; conda-packages.json records installed packages. source.ipynb preserves the input notebook. executed.ipynb contains fresh outputs. The run harness saves results but does not yet automatically reuse caches; reuse must check input/code/environment compatibility. Figures may differ in fonts, rendering and Monte Carlo details. A scientific reproduction pass requires review against the target artifacts.


## Review notes

Visual inspection confirmed the same two-panel posterior structure, labels, ROPE and qualitative pattern as the reference. Reference/new genotype ROPE percentages: 8.8% / 9.72%; drug ROPE: 91.2% / 91.03%. These are Monte Carlo estimates, not pixel-identical figures.

The model-comparison cell emitted PSIS-LOO Pareto-k warnings (>0.7). This affects reliability of the LOO comparison and requires review; successful execution and zero divergences do not resolve it. See warnings.json. No model or data changes were made to suppress these warnings.

Source notebook and input hashes were unchanged after execution. All saved pointwise log likelihoods are finite. The maximum unrounded R-hat is 1.0069633833371305. Fits 01/02 are base/TgReflevel; fits 03–10 follow experiment insertion order (Amiloride 10, Amiloride 12, HCQ, HCQ nano, R-HCQ, S-HCQ, TUDCA, TUDCA WT).
