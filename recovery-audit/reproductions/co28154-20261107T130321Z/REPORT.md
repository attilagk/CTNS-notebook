# CO28154 behavioral notebook reproduction

## Execution

Fresh bayes kernel, with isolated working directory and figures. Existing external outputs were not loaded or overwritten. See manifest.json and conda-packages.json for inputs, code hashes and runtime. Saved posterior fits are local, ignored NetCDF files.

## Artifact 5: non-treatment posterior summaries

| Experiment | Parameter | Reference mean | New mean | Difference |
|---|---|---:|---:|---:|
| TUDCA + HCQ TG | Intercept | 4.776 | 4.776 | +0.000 |
| TUDCA + HCQ TG | Day | -0.325 | -0.325 | +0.000 |
| TUDCA + HCQ TG | censored(Latency, Status)_alpha | 2.892 | 2.891 | -0.001 |
| TUDCA + HCQ TG | 1|IRN_sigma | 0.268 | 0.268 | +0.000 |
| Arundine TG | Intercept | 4.587 | 4.583 | -0.004 |
| Arundine TG | Day | -0.241 | -0.241 | +0.000 |
| Arundine TG | censored(Latency, Status)_alpha | 3.147 | 3.147 | +0.000 |
| Arundine TG | 1|IRN_sigma | 0.220 | 0.220 | +0.000 |
| TUDCA + HCQ WT | Intercept | 4.396 | 4.396 | +0.000 |
| TUDCA + HCQ WT | Day | -0.406 | -0.405 | +0.001 |
| TUDCA + HCQ WT | censored(Latency, Status)_alpha | 2.528 | 2.522 | -0.006 |
| TUDCA + HCQ WT | 1|IRN_sigma | 0.324 | 0.323 | -0.001 |
| Arundine WT | Intercept | 4.137 | 4.138 | +0.001 |
| Arundine WT | Day | -0.352 | -0.352 | +0.000 |
| Arundine WT | censored(Latency, Status)_alpha | 2.815 | 2.809 | -0.006 |
| Arundine WT | 1|IRN_sigma | 0.256 | 0.255 | -0.001 |

Full comparison (including intervals, SD, diagnostics and derived columns): target-5-comparison.csv. The current Bambi parameter alpha is matched to the historical censored(Latency, Status)_alpha label for comparison only. ESS values are stochastic diagnostics, not exact reproducibility targets. No acceptance tolerance was supplied; the table reports observed differences.

## Artifact 3: BF_all posterior means

| Experiment | Treatment | Reference mean | New mean | Difference |
|---|---|---:|---:|---:|
| Amiloride 10 TG | Amiloride TG | 0.298 | 0.070 | -0.228 |
| Amiloride 10 TG | Saline WT | -0.299 | -0.261 | +0.038 |
| CO28152 TG | Amiloride TG | 0.282 | 0.276 | -0.006 |
| CO28152 TG | HCQ TG | -0.028 | -0.032 | -0.004 |
| CO28152 TG | HCQ nano TG | 0.092 | 0.086 | -0.006 |
| CO28152 TG | R-HCQ TG | 0.066 | 0.061 | -0.005 |
| CO28152 TG | S-HCQ TG | 0.103 | 0.096 | -0.007 |
| CO28152 TG | TUDCA TG | 0.033 | 0.028 | -0.005 |
| CO28152 TG | Saline WT | -0.292 | -0.294 | -0.002 |
| TUDCA + HCQ TG | TUDCA + HCQ TG | -0.398 | -0.395 | +0.003 |
| TUDCA + HCQ TG | Saline WT | -0.583 | -0.584 | -0.001 |
| Arundine TG | Arundine low dose TG | -0.140 | -0.136 | +0.004 |
| Arundine TG | Arundine mid dose TG | -0.151 | -0.144 | +0.007 |
| Arundine TG | Arundine high dose TG | -0.099 | -0.095 | +0.004 |
| Arundine TG | Vehicle (Arundine) WT | -0.681 | -0.679 | +0.002 |

The reference BF_all display omits columns and truncates its final row. target-3-BF_all.csv preserves the complete new result. Tail-count-based values labeled Bayes factors can vary substantially even when posterior means agree. This run preserves the existing pseudocount and computation; it does not validate the scientific interpretation of those values.

## Artifacts 1, 2 and 4

Artifact 1 is exported as target-1-ess.csv. Figures are figures/model-latency-CO28154-TUDCA_HCQ-TG.png and figures/CI-BF-all.png. Compare them with the screenshots identified in target-section.md.

## Known experiment-definition concerns

The CO26990 two-level WT entries use CO28152 data; the corresponding male-only WT entry also uses the full CO28152 dataset. The CO28152 three-level entries named TUDCA WT select TUDCA TG. These pre-existing definitions do not supply the requested target artifacts and were not changed silently. They require scientific review.

## Reuse

The isolated notebook cache uses fpaths.csv under workspace/results. Model-fit caches are also saved immediately under fits. Source and package versions must be checked before reusing a run. Raw reference cells remain unexecuted, matching their original notebook type.


## Validation outcome: partial reproduction

All original code cells completed, and the derived-table cache fix was rerun successfully using saved fits. Artifacts 2 and 5 are closely consistent with the references; Artifact 1 has healthy target-model diagnostics (ESS varies). Artifacts 3 and 4 are not fully reproduced: CO26990 Amiloride effect is 0.070 rather than 0.298. The separate November latency reproduction yielded 0.072. The fresh CO26990 alpha/intercept/day/random-effect scale also agree closely with the original saved non-treatment summary (2.833/4.527/-0.292/0.248 vs 2.836/4.525/-0.292/0.250). This suggests a historical cached-table/provenance inconsistency but does not prove its origin. Original inference files or clarification of original animal selection/model are needed. Reference targets were not edited.

The notebook previously loaded BF-all.csv/BF-all_m.csv solely by existence. Both cells now recompute these cheap derived tables from loaded posterior fits, preventing stale table reuse. No model formulas, priors, data, seeds or sampling settings were changed.

All seven target-producing fits have zero divergences, finite log likelihoods and maximum R-hat below 1.006. Several non-target male-only fits have divergences (including 30 in the CO28152 two-level Amiloride model); see model-diagnostics.json. These remain unresolved and must not be described as fully validated. The existing experiment-definition concerns listed above also remain unresolved.
