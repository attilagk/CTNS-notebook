# Inflammation input reconstruction

Recovered `resources/QPS/5xFAD-mouse/inflammation-cleaned.csv` on branch
`recover-inflammation-input`. The CSV has two header rows, six unnamed index
columns (Study, Cohort, Group, IRN, Sex, Genotype), and 20 measurement columns:
ten cytokines under Cortex and the same ten under Hippocampus. It contains
60 CO28154 animals and 59 CO28152 animals, matching the available raw records.
There are 2,251 numeric measurements and 129 blanks. Units remain pg/g tissue;
there is no unit conversion, normalization, imputation, or deliberate rounding.

## Sources

- CO28154: `Inflammation` sheet of the user-linked Dropbox workbook,
  https://www.dropbox.com/scl/fi/4pw5yj6qpp8x8ilv8x9a9/CO28154-rawdata-biochemistry.xlsx?rlkey=vlh7nygwyn0l4pr7bmpible1k&dl=0
- CO28152: `Inflammation` sheet of the previously recovered local
  `CO28152_IV_data all.xlsx`, reconciled independently to all animal records
  in the user-linked PDF sections 9.3.1.3 (Cortex, report/PDF pages 67-69)
  and 9.3.1.4 (Hippocampus, pages 70-72),
  https://www.dropbox.com/scl/fi/3bb6ow6upfmuoex34kh6e/CO28152_Report_final.pdf?rlkey=d6a0hzz17xisgyoxfi229vlwe&dl=0

The local CO28154 workbook and PDF have Dropbox content hashes identical to
those returned by the current connected Dropbox download-link API. Reusing
those identical local bytes avoids another download. Receipt records the
source identities, revisions, content hashes, SHA256 hashes, and destination.
All six PDF pages were rendered and visually inspected. Independent text
extraction matched all 1,180 measurement tokens, identifiers and flags for
59 animals in both regions. The raw CO28152 workbook supplies full precision;
the PDF is rounded and cannot recover those extra digits by itself.

## Cleaning decisions

Remove decorative headers, the leading empty column, and group mean/SEM
rows. Preserve every animal row, its source order and metadata, including
IRN suffixes. Map starred values and `n.m.` to blank measurements. Do not
discard an animal because individual assays are missing. Each changed cell
and its original token are listed in `validation.json`.

One additional exclusion is explicit in the notebook's saved Out[3]: animal
CO28152/16192, Cortex IFN-γ is NaN there, although its unflagged raw value is
1.3143269688796921 (PDF 1.314). Preserve that blank to honor the saved cleaned
data evidence. This is a single-cell reconstruction decision, not an inferred
general outlier algorithm; the biological reason is not recoverable here.

## Validation and limits

Read back all 119 CSV records and compare every identifier, numeric value and
missing cell to extraction. The actual notebook input code produces CO28152
118 rows x 18 columns, agreeing with saved Out[3], and CO28154 120 x 18.
The ten visible Out[3] records agree in metadata, seven missing cells and
93 numeric cells within 0.000001. Two displayed IL-6 values are 118.149094,
where the source is 118.1490934730664, a difference of 0.0000005269; full
source precision was retained. Out[3] is truncated, so hidden original
cleaned rows cannot be compared directly to that display.

Selected original OLS cells were also evaluated without writing canonical
results or figures. Of 400 visible saved estimate/SE/p-value/t-value cells,
398 match at displayed precision. Two CO28154 cells lie just beyond their
rounding boundaries: IL-10 intercept p-value 3.365444e-21 versus
3.3654434746700268e-21, and IL-2 Saline WT coefficient -5.483121 versus
-5.483120499611648. All 200 visible CO28152 cells match at displayed precision.
`ols-comparison.json` retains these discrepancies and pre-existing category
compatibility/rank-deficiency warnings. No values or models were adjusted
to force agreement. This is input recovery with targeted checks, not full
notebook reproduction or proof of byte-identical recovery of the lost CSV.

The recovery inventory was refreshed using `refresh_inventory.py`, preserving
the immutable original missing-file baseline and adding source provenance.
The notebook itself is unchanged. The recovered CSV lives outside this Git
repository; the audit scripts, evidence and report are local uncommitted
changes on the recovery branch. No push, PR or merge was requested here.

## Repeat

Use `pdftotext -layout` on the PDF, then run with base Python:

```
reconstruct.py RESOURCE_DIR PDF_LAYOUT_TEXT STAGE_DIR
compare_estimates.py STAGE_DIR/inflammation-cleaned.csv
```

The reconstruction writes the staged CSV and audit evidence. Install only
after reviewing validation; never overwrite a different existing input.
