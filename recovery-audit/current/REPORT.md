# Notebook input recovery audit

Static audit of all 84 notebooks. No notebooks were executed; no source data or Dropbox files were changed or downloaded. Includes two checkpoint notebooks. Cell numbers are one-based positions including Markdown cells; the cell column identifies the calling cell and source #cell identifies a notebook helper definition.

## Findings

- 786 input records; 209 distinct paths missing at the expected location.
- 158 records point to existing files; 72 records require manual resolution.
- Local filename candidates searched across resources, results, and notebooks (1004 files indexed; AppleDouble files excluded).
- Supplied Dropbox /QPS folder fully listed: 51 files. Candidate matching is by filename, not verified contents.

Counts describe static references, including conditional and exploratory code, rather than guaranteed execution failures. Multiple records may reference the same file.

## Manuscript inputs missing at expected paths

| Expected path | Local / Dropbox candidates | Upstream producer |
|---|---|---|
| notebooks/2023-11-07-5xfad-maze-latency/.json | None found | Not identified |

## Missing input summary by notebook

| Notebook | Missing paths | Unresolved records |
|---|---:|---:|
| 2021-06-14-proximity/2021-06-14-proximity.ipynb | 2 | 0 |
| 2021-07-01-high-conf-ADgenes/2021-07-01-high-conf-ADgenes.ipynb | 1 | 0 |
| 2021-08-04-guney-tools/2021-08-04-guney-tools.ipynb | 5 | 6 |
| 2021-08-11-drugbank/2021-08-11-drugbank.ipynb | 2 | 1 |
| 2021-09-02-asthma-drugs/2021-09-02-asthma-drugs.ipynb | 2 | 1 |
| 2021-09-15-proximity-summary/2021-09-15-proximity-summary.ipynb | 6 | 0 |
| 2021-09-25-parallel/2021-09-25-parallel.ipynb | 5 | 2 |
| 2021-09-28-drugbank-screen-v1/2021-09-28-drugbank-screen-v1.ipynb | 3 | 1 |
| 2021-10-04-CMap-discussion/2021-10-04-CMap-discussion.ipynb | 11 | 5 |
| 2021-10-11-chembl-bindingdb/2021-10-11-chembl-bindingdb.ipynb | 3 | 0 |
| 2021-10-18-py4cytoscape/2021-10-18-py4cytoscape.ipynb | 2 | 1 |
| 2021-10-24-chembl-query/2021-10-24-chembl-query.ipynb | 5 | 1 |
| 2021-11-09-chembl-screen-v1/2021-11-09-chembl-screen-v1.ipynb | 4 | 1 |
| 2021-11-18-proximity-run-benchmark/2021-11-18-proximity-run-benchmark.ipynb | 6 | 2 |
| 2021-11-28-ad-de-genes-apoe/2021-11-28-ad-de-genes-apoe.ipynb | 0 | 0 |
| 2021-11-30-proximity-cmap/2021-11-30-proximity-cmap.ipynb | 16 | 0 |
| 2021-12-02-proximity-various-ADgenesets/2021-12-02-proximity-various-ADgenesets.ipynb | 22 | 0 |
| 2021-12-04-ad-subtypes/2021-12-04-ad-subtypes.ipynb | 0 | 0 |
| 2021-12-08-apoe-de-genes-tunahan/2021-12-08-apoe-de-genes-tunahan.ipynb | 1 | 0 |
| 2021-12-11-bbb-permeability/2021-12-11-bbb-permeability.ipynb | 0 | 0 |
| 2021-12-13-chembl-drug-info/2021-12-13-chembl-drug-info.ipynb | 6 | 2 |
| 2021-12-31-rank-aggregation/2021-12-31-rank-aggregation.ipynb | 2 | 0 |
| 2022-01-14-rank-aggregation/2022-01-14-rank-aggregation.ipynb | 3 | 0 |
| 2022-01-14-top-drugs/2022-01-14-top-drugs.ipynb | 5 | 2 |
| 2022-01-31-single-cell-de-genes/2022-01-31-single-cell-de-genes.ipynb | 3 | 0 |
| 2022-02-01-AD-DE-gene-sets/2022-02-01-AD-DE-gene-sets.ipynb | 5 | 1 |
| 2022-02-22-twas-update/2022-02-22-twas-update.ipynb | 2 | 0 |
| 2022-02-25-dilara-rmta-geneset/2022-02-25-dilara-rmta-geneset.ipynb | 4 | 0 |
| 2022-02-28-rat-study/2022-02-28-rat-study.ipynb | 2 | 0 |
| 2022-04-04-dilara-drug-rediscovery/2022-04-04-dilara-drug-rediscovery.ipynb | 7 | 0 |
| 2022-05-09-AD-drugs/2022-05-09-AD-drugs.ipynb | 5 | 0 |
| 2022-05-13-GO-new-AD-drugs/2022-05-13-GO-new-AD-drugs.ipynb | 0 | 0 |
| 2022-05-17-geneset-subtract/2022-05-17-geneset-subtract.ipynb | 11 | 0 |
| 2022-05-23-functional-enrichment/2022-05-23-functional-enrichment.ipynb | 3 | 0 |
| 2022-08-11-pascal/2022-08-11-pascal.ipynb | 13 | 0 |
| 2022-08-22-dream-modules/2022-08-22-dream-modules.ipynb | 4 | 0 |
| 2022-08-31-dream-modules-gprofiler/2022-08-31-dream-modules-gprofiler.ipynb | 1 | 0 |
| 2022-09-21-cell-based-assays/2022-09-21-cell-based-assays.ipynb | 12 | 0 |
| 2022-09-26-selected-drugs/2022-09-26-selected-drugs.ipynb | 6 | 1 |
| 2022-09-30-tutorial/2022-09-30-tutorial.ipynb | 0 | 1 |
| 2022-10-07-AD-subtypes-rMTA/2022-10-07-AD-subtypes-rMTA.ipynb | 1 | 0 |
| 2022-11-09-rat-metabolites/2022-11-09-rat-metabolites.ipynb | 12 | 5 |
| 2022-12-12-rat-metabolites/2022-12-12-rat-metabolites.ipynb | 6 | 0 |
| 2022-12-13-rat-metabolites/2022-12-13-rat-metabolites.ipynb | 9 | 2 |
| 2022-21-10-mathys-iMAT/2022-21-10-mathys-iMAT.ipynb | 6 | 0 |
| 2023-01-06-mixed-model-metab/2023-01-06-mixed-model-metab.ipynb | 9 | 1 |
| 2023-02-01-bibliography/2023-02-01-bibliography.ipynb | 0 | 2 |
| 2023-02-06-AD-subtypes-active-reactions/2023-02-06-AD-subtypes-active-reactions.ipynb | 3 | 0 |
| 2023-02-10-gem-subsystems/.ipynb_checkpoints/2023-02-10-gem-subsystem-checkpoint.ipynb | 4 | 2 |
| 2023-02-10-gem-subsystems/2023-02-10-gem-subsystem.ipynb | 4 | 2 |
| 2023-02-22-bayes-logi-model-fitting/2023-02-22-bayes-logi-model-fitting.ipynb | 1 | 0 |
| 2023-03-03-Y-AU-AI-boxplots/2023-03-03-Y-AU-AI-boxplots.ipynb | 5 | 0 |
| 2023-04-14-AD-subtype-specificity/2023-04-14-AD-subtype-specificity.ipynb | 2 | 4 |
| 2023-04-20-cell-based-assays/2023-04-20-cell-based-assays.ipynb | 1 | 0 |
| 2023-08-25-imat-validation/2023-08-25-imat-validation.ipynb | 3 | 3 |
| 2023-09-05-ipsc-results/2023-09-05-ipsc-results.ipynb | 1 | 2 |
| 2023-09-13-cell-based-assays-bayes/2023-09-13-cell-based-assays-bayes.ipynb | 5 | 2 |
| 2023-09-26-cell-bayes-assays/2023-09-26-cell-bayes-assays.ipynb | 8 | 4 |
| 2023-10-13-5xfad-maze/.ipynb_checkpoints/2023-10-13-5xfad-maze-checkpoint.ipynb | 0 | 0 |
| **2023-10-13-5xfad-maze/2023-10-13-5xfad-maze.ipynb** | 0 | 0 |
| **2023-11-07-5xfad-maze-latency/2023-11-07-5xfad-maze-latency.ipynb** | 1 | 6 |
| **2023-11-07-5xfad-maze-test-probe/2023-11-07-5xfad-maze-test-probe.ipynb** | 0 | 0 |
| **2023-11-15-5xfad-nfl-gfap/2023-11-15-5xfad-nfl-gfap.ipynb** | 0 | 0 |
| 2023-11-19-cell-based-assays/2023-11-19-cell-based-assays.ipynb | 3 | 1 |
| **2023-12-18-5xfad-nfl-gfap-frequentist/2023-12-18-5xfad-nfl-gfap-frequentist.ipynb** | 0 | 0 |
| **2024-01-05-5xfad-nfl-frequentist/2024-01-05-5xfad-nfl-frequentist.ipynb** | 0 | 0 |
| 2024-01-08-apoe-logistic-regression/2024-01-08-apoe-logistic-regression.ipynb | 1 | 0 |
| 2024-01-21-cell-bayes-assays-dim/2024-01-21-cell-bayes-assays-dim.ipynb | 9 | 1 |
| 2024-02-14-cell-bayes/2024-02-14-cell-bayes.ipynb | 8 | 0 |
| **2024-02-29-Y-maze/2024-02-29-Y-maze.ipynb** | 0 | 0 |
| 2024-03-04-cell-bayes-vs-frequentist/2024-03-04-cell-bayes-vs-frequentist.ipynb | 8 | 0 |
| 2024-04-09-qps-results-summary/2024-04-09-qps-results-summary.ipynb | 15 | 0 |
| 2024-04-21-cell-bayes/2024-04-21-cell-bayes.ipynb | 8 | 0 |
| 2024-04-29-cell-bayes/2024-04-29-cell-bayes.ipynb | 10 | 0 |
| 2024-05-09-cell-bayes/2024-05-09-cell-bayes.ipynb | 8 | 0 |
| 2024-05-14-cell-bayes/2024-05-14-cell-bayes.ipynb | 8 | 0 |
| 2024-05-20-nfl-modeling/2024-05-20-nfl-modeling.ipynb | 1 | 6 |
| **2024-06-04-5xfad-behavior-CO28154/2024-06-04-5xfad-behavior-CO28154.ipynb** | 0 | 0 |
| 2024-06-26-mwm-test-probe-CO28154/2024-06-26-mwm-test-probe-CO28154.ipynb | 0 | 0 |
| **2024-08-08-biochemistry/2024-08-08-biochemistry.ipynb** | 0 | 0 |
| **2024-08-08-biochemistry/statannotations-example.ipynb** | 0 | 0 |
| 2024-11-11-cytokines/2024-11-11-cytokines.ipynb | 0 | 0 |
| 2025-05-25-ad-subtype-random-fx/.ipynb_checkpoints/2025-05-25-ad-subtype-random-fx-checkpoint.ipynb | 0 | 0 |
| 2025-05-25-ad-subtype-random-fx/2025-05-25-ad-subtype-random-fx.ipynb | 0 | 1 |

## How to use the inventory

- `missing-files.csv`: one row per distinct missing expected path, with affected notebooks and recovery candidates.
- `input-inventory.csv`: one row per loading site and resolved path, or explicit unresolved expression. Filter `manuscript=yes` to prioritize the nine study directories.
- `output-producers.csv`: statically detected writes, matched to input paths where possible; this does not prove successful regeneration.
- `loading-sites.csv`: every syntactically recognized Python loading site, including deferred functions that could not be followed.
- `notebook-coverage.csv`: every notebook, including those with no detected inputs, plus shell/unparsed cell review locations.
- `dropbox-candidate-metadata.json`: metadata confirmation for six same-name backup candidates; workbook contents remain unchecked.
- `dropbox-qps-listing.json`: complete metadata snapshot of the supplied backup, with file IDs, paths, sizes, and modification times.
- `summary.json`: machine-readable counts and local traversal errors.

## Limits and next steps

This is conservative static analysis, not a successful rerun or proof of an exhaustive runtime dependency set. Paths assume the notebook directory as working directory; checkpoint snapshots use their original notebook directory. Libraries and executable availability are outside this data-file audit. Both sides of conditions are inspected; unknown branch-dependent values remain unresolved. Helper functions in this repository are followed where argument values can be resolved; external helpers are flagged rather than executed. R loaders are scanned for literal or scalar-variable paths; arbitrary R expressions remain unresolved. DataFrame-driven paths, callbacks, nested closures, environment variables, configuration files, manifests, archives, imports of external packages, and dynamically generated names may require manual inspection. Directory records do not establish that every required member is present. Shell cells are retained as unresolved review entries even when literal path references can be extracted. Relative output producer matching is exact-path only. No workbook sheets or columns have been validated. A same-name candidate may contain different data.

Review unresolved entries and filename candidates before restoring inputs. Validate workbook sheets, columns, study identifiers, and versions against notebook expectations; recover generated outputs from their producers or locate additional backups. Backup coverage outside /QPS is unknown.
