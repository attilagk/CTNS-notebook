# CO28152 cleaning rules: independent and human comparison

All four rules in behavioral-data-cleaning.md agree with the independently inferred transformations. No conflicting cleaning rule was found.

The independent report and its cell-level evidence were saved before the human notes were opened. Neither workbook nor the human notes were changed.

## Rule-by-rule check

| Human rule | Result | Workbook evidence |
|---|---|---|
| 1. Remove Mean and SEM summary rows on all sheets | Confirmed | Each sheet loses exactly ten Mean rows and ten SEM rows. All 145 animal records on each sheet remain, in their existing order. |
| 2. Remove column A and row 1 if empty | Confirmed, with the stated condition | Both MWM sheets start at raw B2 and are shifted to cleaned A1. Y-Maze already starts at A1 with meaningful content; its column A is retained. Its redundant upper header is handled separately under rule 4. |
| 3. Merge the cells above Cohort, Group, Sex, IRN and add Covariates for the MWM training sheet | Confirmed | The cleaned training sheet has Covariates in merged A1:D1, with Cohort, Group, Sex, IRN in A2:D2. The five four-day metric blocks retain their two-level headers. |
| 4. Prepend the merged Y-Maze metric to each week label | Confirmed | Raw E1:F1 contains Total entries and G1:H1 contains Correct entries %. Combining these with E2:H2 gives the four cleaned headers listed below. |

The cleaned Y-Maze headers E1:H1 are exactly:

- `Total entries, week 12`
- `Total entries, week 13`
- `Correct entries %, week 12`
- `Correct entries %, week 13`

## Details to add to a reproducible specification

1. **Rename the training sheet.** The observed rename is `raw data MWM day 1 - 4` to `rawdata MWM day 1-4`. This is required for the existing notebook paths and sheet selections. The probe sheet remains `raw data MWM day 5`, and Y-Maze remains `Y-Maze`.
2. **After composing Y-Maze column names, remove the redundant original upper header and its merges.** The final table has one header row and no merged cells. The MWM training table retains two header rows and six merged ranges, including the new Covariates block.
3. **Preserve animal records and values, including blanks, source units, cell types, identifiers and ordering.** No animal exclusions, numerical corrections, unit conversions, rounding, latency capping, percent rescaling, or imputation were observed. Seven Y-Maze percent measurements remain blank. IRN suffixes such as `14627 A` remain intact.

These are additions or implementation details, not disagreements with the four human rules. The first two should be made explicit so a script reproduces the same workbook table structure.

## Independent verification

| Sheet | Raw meaningful range | Cleaned range | Animal rows | Summary rows removed | Animal-table cells checked | Numeric measurements | Blank measurements | Changed animal cells |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MWM days 1–4 | B2:Y168 | A1:X147 | 145 | 20 | 3,480 | 2,900 | 0 | 0 |
| MWM day 5 | B2:J167 | A1:I146 | 145 | 20 | 1,305 | 725 | 0 | 0 |
| Y-Maze | A1:H167 | A1:H146 | 145 | 20 | 1,160 | 573 | 7 | 0 |
| Total | | | 435 sheet records / 145 animals | 60 | 5,945 | 4,198 | 7 | 0 |

The other 1,740 compared cells are covariates. Exact comparisons use stored numerical precision, not displayed rounded numbers. The raw-to-clean transformation reproduces all nonempty data and header values and every cleaned merged header range. All 120 raw formulas occur in discarded summary rows; retained animal cells have no formulas.

Y-Maze and MWM have the same subject keys after accounting for an existing type difference: IRN 16006 is numeric in Y-Maze and text in MWM. Each cleaned sheet preserves its corresponding raw cell type. Y-Maze also preserves its own raw row order, which differs from the MWM order. A reconstruction should not standardize these differences unless separately requested.

## Applying the rules to CO28154

This comparison establishes CO28152 table-cleaning rules. It does not establish the layout of every CO28154 sheet. Before reconstructing that workbook, match its actual headers, merged cells, summary labels, units, and measurement periods. Use its sheet-specific names and periods rather than copying CO28152 row numbers, week 12/13 labels, group membership, or the 145-animal count. No CO28154 data have been cleaned or changed in this step.

## Files and comparison order

- Independent assessment: `co28152-cleaning-independent.md`
- Independent detailed evidence: `co28152-cleaning-evidence.json`
- Reusable read-only comparison: `compare_behavioral_workbooks.py`
- Human reference: `behavioral-data-cleaning.md`

The script does not read the human reference. Styling, themes, Excel UI editing history and byte-for-byte workbook packaging are outside this comparison; analytical values, headers, record order and merged ranges were checked.

Independent report SHA-256, recorded before reading the human notes:

`d77952feb0164e3cd1409e8bb2337f6fab92801b4c9f6c438443e0c5de7b4b65`

Human reference SHA-256 at comparison:

`b18d3d385238f371bfdc2f49304537c8928d7ed971c26a68f1b00bad60f7dfae`
