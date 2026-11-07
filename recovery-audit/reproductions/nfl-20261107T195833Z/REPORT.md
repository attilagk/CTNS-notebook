# NFL modeling reproduction

Executed all cells in a fresh base kernel. Four declared target figures reproduce:
TG and WT longitudinal NF-L curves, the TG t-test heatmap, and the OLS effect/p-value figure.
Visual review confirms groups, trajectories, colors, labels, axes, heatmap patterns and
scientific interpretation. Minor rendering/layout differences reflect package versions.
Historical dataframes (2,023 cells), t-test table (160 cells), and OLS table (98 cells)
agree at displayed precision; see table-comparisons.json and compare_nfl.py.

Inputs are the recovered animal-list CSVs, reconstructed CO28154 longitudinal NF-L sheet,
and restored CO28152 NF-L workbook. Source identities are in manifest.json. The known
CO28152 14631 A sex discrepancy remains preserved as documented in the input reconstruction.

Compatibility changes replace removed DataFrame.applymap with map and the obsolete
xarray .variables access with direct variable indexing. Formula, cohorts, priors, units
and sampling defaults are preserved. Three exploratory Gaussian Bambi fits used four
chains, 1,000 tune and 1,000 retained draws per chain; no seed was present originally.
Zero divergences; unrounded maximum R-hat 1.002733, minimum bulk ESS 1565.998.
Full per-parameter diagnostics and raw warning text are retained. Future pandas categorical
warnings and an obsolete exploratory forest-plot sigma exclusion warning remain; they do
not affect the four deterministic targets. The forest plots may display sigma as well.

Base initially lacked arviz/bambi/pymc/h5netcdf/h5py. Compatible PyMC 5/Bambi 0.17/ArviZ 0.23
were installed; Matplotlib 3.11 was replaced with 3.10 because ArviZ imports style.core.
Actual pip versions are in pip-packages.json; conda-packages.json may retain pre-pip conda
metadata, so the pip inventory and cache package identities govern runtime versions.

Execution ran in workspace/ with resources read and all results/figures isolated. Large
fits and transient notebook copies stay local. Canonical persistent caching is separately
verified during promotion. See manifest.json, validation-status.json, execution.log,
diagnostics.jsonl and the selected figures. No claim is made that Git backs up fits or inputs.
