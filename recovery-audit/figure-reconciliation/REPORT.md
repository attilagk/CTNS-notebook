# Figure recovery reconciliation — 25 November 2026

**The files are now recovered for the enumerated active exports; complete scientific reproduction of every figure is not established.** The journal overstated the evidence when it said all notebooks and artifacts were reproduced. See `INITIAL-REPORT.md` for the original claim, scope, and sigmoid dependency trace.

Current scope: 12 TUDCA + HCQ notebooks, plus the specifically requested sigmoid-4 figure. The inventory contains **313 export files**: 158 PNGs and 155 PDFs (158 figure basenames; PNG/PDF pairs count twice). **150 files were freshly regenerated in this task; 163 were preserved from surviving earlier runs.** No active enumerated export remains without a selected copy. Three disabled `if False` probe exports are excluded, not silently re-enabled. Other cell-assay figures are outside this requested scope.

Durable local project storage: `/home/attila/projects/CTNS/recovered-figures/2026-11-25/`. Open `selected/<notebook>/<filename>` for the preferred copy. Relative symlinks point to preserved versioned files within the same directory tree. Historical references and weaker candidates are separated. This location survives temporary-worktree cleanup; **off-machine backup has not been verified**.

Verification: all 313 selected paths reopened successfully; all 158 PNGs were decoded, and all 155 PDFs were rendered with Poppler. Checksums match the preservation manifest. Ten contact sheets were visually inspected for nonblank content and gross rendering problems; this is not a new scientific comparison of every plot.

## Notebook outcomes

| Notebook | Export files | Fresh | Earlier run | Reference validation |
|---|---:|---:|---:|---|
| 2023-11-07-5xfad-maze-latency | 8 | 0 | 8 | prior target validation documented |
| 2023-11-07-5xfad-maze-test-probe | 9 | 0 | 9 | prior target validation documented |
| 2024-02-29-Y-maze | 10 | 10 | 0 | execution-only |
| 2024-05-20-nfl-modeling | 60 | 0 | 60 | four targets previously validated |
| 2024-06-04-5xfad-behavior-CO28154 | 74 | 0 | 74 | partial reference reproduction |
| 2024-06-26-mwm-test-probe-CO28154 | 12 | 0 | 12 | prior table agreement; figures regenerated |
| 2024-08-08-biochemistry | 12 | 12 | 0 | prior target validation; fresh exports recovered |
| 2024-11-11-cytokines | 6 | 6 | 0 | prior qualified target agreement; fresh exports recovered |
| 2023-11-15-5xfad-nfl-gfap | 34 | 34 | 0 | execution-only |
| 2023-12-18-5xfad-nfl-gfap-frequentist | 24 | 24 | 0 | execution-only |
| 2024-01-05-5xfad-nfl-frequentist | 22 | 22 | 0 | execution-only |
| 2023-10-13-5xfad-maze | 40 | 40 | 0 | execution-only |
| 2023-09-26-cell-bayes-assays | 2 | 2 | 0 | focused visual reproduction; inference caveat |

## Evidence and files

- `recovery-inventory.csv`: one row per active export, initial observation, present status, action, preferred durable path and SHA-256. Initial means the start of this follow-up, not the original November pre-recovery baseline.
- `initial-observation/`: unchanged first scan, including older cell-assay candidates subsequently excluded from scope. Existing `recovery-audit/baseline/` remains untouched.
- `expected-figures.csv`, `found-copies.csv`, `inventory.json`: code/receipt expectations, every found path, provenance grades, timestamps and readability. Timestamps are descriptive evidence, never sole proof of origin.
- `legacy-or-extra-figures.csv`: older, disabled, or otherwise unselected candidate names; not counted as missing active exports.
- `export-sites.csv` and `dynamic-export-resolution.json`: source cells and runtime resolution of seven dynamic study export sites. Ten other dynamic cell-assay sites are outside the single-figure request. The Dirichlet package also exports PNGs through its `save_figure` argument; its three surviving run PNGs are included.
- `durable-files.csv`, `durable-manifest.json`, `durable-verification.json`: original and durable paths, hashes, evidence grade, selected copies, and reopen/render checks.
- `notebook-validation.csv`: reference agreement and inference-quality limitations, independently of survival.
- `sigmoid-comparison.json`: preserved reference/new image identities and focused comparison.

Evidence grades: historical per-file hash match (154 selected files) is stronger than a successful isolated run corroborated by its matching source snapshot and retained executed notebook (159 selected files). Newly recorded checksums protect future identity but do not retroactively create historical per-file receipts.

## Sigmoid case

Recovered `prior-posterior-curves-sigmoid-4.png` and `.pdf` from a fresh bayes kernel using backup workbook/controls copied into isolation. Original reference images are preserved separately; no original was overwritten. Reference/new PNGs are both 1131×965; average absolute RGB difference is 3.3044/255, descriptive only. Visual comparison supports qualitative agreement, not pixel identity or a posterior-equivalence test. The four-parameter model retained its inputs, priors, seeds and sampling defaults. It produced 27 divergences; max R-hat 1.008218, minimum bulk ESS 765.876, finite likelihood.

The initial focused attempts and their errors remain recorded. The first omitted a prerequisite used by an example expression. The next reached the unrelated sigmoid-2 comparison and failed initial evaluation at its sigma prior. The successful attempt excluded that unrelated fit/plot invocation while retaining the requested sigmoid-4 code. **The full cell-assay notebook has not been reproduced.**

## Remaining gaps

- Scientific verification: five November notebook promotions were explicitly execution-only. The new exports do not retrospectively establish agreement with all historical figures.
- CO28154 behavior retains the documented CO26990/Amiloride target discrepancy (mean 0.070 vs 0.298) and other limitations.
- Maze convergence/LOO problems remain accepted historical limitations; the new maze execution reused validated caches without sampling.
- Sigmoid-4 divergences and the unrelated sigmoid-2 failure remain unresolved.
- The probe scatter export still saves before its centimeter ylabel update; this previously accepted source defect is preserved, not silently changed.
- Hand-edited manuscript-final versions are not fully defined by the notebook exports; the journal lists them as a separate next step.
- The first Y-maze target uses `2024-02-29-Y-maze.png` as a screenshot label; the corresponding first CO26990 boxplot is `pct-correct-entries-boxplot-CO26990.png`. This is an alias, not a newly missing export filename.
- Local storage is verified; independent backup retention is not. No all-figures scientific-completion claim is warranted.

No canonical notebook, raw input or existing posterior cache was overwritten. No promotion, commit, push, or PR was performed.
