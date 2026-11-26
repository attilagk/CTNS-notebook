# Initial reconciliation — 25 November 2026

The journal's 7–8 November statement that **all notebooks and artifacts were reproduced** is broader than the recorded validation supports. The original static audit excluded images. Five later notebook promotions explicitly tested execution only; the CO28154 behavior promotion retained a material CO26990/Amiloride target discrepancy. Successful execution, agreement with reference figures, survival of exports, and off-machine backup are separate questions.

Scope: the 12 notebook sections in `recovery-audit/target-artifacts/target-artifacts.md`, plus the specifically requested sigmoid-4 figure from `2023-09-26-cell-bayes-assays`. Other cell-assay exports and extra legacy filenames are retained as candidates, not added to the 12-notebook claim.

`initial-observation/` preserves the first filesystem observation before fresh recovery execution. Its initial 300-row CSV also included 50 other cell-assay exports; the current inventory narrows that notebook to the requested PNG/PDF pair. Counts are export files, so a PNG/PDF pair counts twice. They are not counts of distinct scientific panels.

The first scan found exact historic receipt hash matches for six surviving target PNGs (four NFL modeling, one biochemistry, one cytokines), plus numerous readable run-directory copies without contemporaneous per-file hash receipts. Those weaker copies must not be called cryptographically verified reruns just because they occupy a reproduction directory. Older copies survive under `/home/attila/CTNS~/notebooks/` and `/home/attila/projects/CTNS/notebooks~/`.

The `/tmp/ctns-rerun-20261108/` directory no longer exists. Git still lists its worktrees as prunable. The November/December NFL/GFAP manifests retain hashes for exports whose temporary files are missing; January NFL, Y-maze and maze reports document isolated exports that also need recovery. Older copies are reference candidates, not evidence that the November rerun exports survived.

## Sigmoid-4 trace

Canonical notebook: `2023-09-26-cell-bayes-assays/2023-09-26-cell-bayes-assays.ipynb`.

- Cell 2 defines the workbook reader and concentrations; cell 5 loads `Abeta release` with `skiprows=1`.
- Cell 21 reads `TI21_22_26_rawdata_sum-controls.csv`; cell 22 defines `extract_regr_data`. Its trailing demonstration requires cell 6 (`Abeta clearance`) even though that experiment is unrelated to the target.
- Cell 23 selects `Abeta release`, `Aβ40`, `TI21`, standardizes activity using the sample standard deviation ×10, and calls `src/cellbayesassay.py::sample_sigmoid_4`.
- Cell 32 defines `prior_posterior_curves`: prior predictive draw count 1000, seed 1986. Cell 33 adds the sigmoid-4 title and exports `named-figure/prior-posterior-curves-sigmoid-4.{png,pdf}` using `/home/attila/src/attila_utils.py`.
- The fit helper preserves four seeds `[1947, 1949, 1976, 2021]` and PyMC's sampling defaults. No historical fit-cache identity was found for this particular target.
- Inputs are missing from the active resources directory but survive in `/home/attila/CTNS~/resources/cell-based-assays/`: `TI21_22_26_rawdata_sum-edited.xlsx` and `TI21_22_26_rawdata_sum-controls.csv`. Workbook sheets/columns and control mapping were read successfully.
- Older reference PNG/PDF survive in `/home/attila/CTNS~/notebooks/2023-09-26-cell-bayes-assays/named-figure/` and the `notebooks~` copy. The PNG opens. The relevant generating cells match the preserved notebook, supporting its use as a historical visual reference. March 2025 modification times alone do not establish when or where the image was generated.
- No pre-existing recovery-era rerun receipt for this notebook was found. Its absence from the 12-notebook scope explains why the broad project completion wording did not cover it.

Fresh runs and preservation outcomes are recorded separately in `REPORT.md`, `inventory.json`, and the durable manifest. The initial observation is never rewritten as current status.
