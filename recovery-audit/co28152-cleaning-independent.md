# Independent CO28152 behavioral cleaning rules

This assessment was written before opening the human comparison in behavioral-data-cleaning.md. The comparison script reads only the raw and cleaned workbooks. Neither workbook was modified.

## Result

Cleaning changes the table layout and headers. All 145 animal records per sheet, their identifiers, and their within-sheet ordering are preserved. Across the three sheets, 5,945 animal-table cells match exactly, including 4,198 numeric measurement cells, 1,740 covariate cells, and seven blank measurement cells. No measurement conversion, rounding, correction, imputation, clipping, or animal exclusion was observed.

| Raw sheet | Cleaned sheet | Animal rows | Removed summary rows | Measurement cells (numeric / blank) |
|---|---|---:|---:|---:|
| raw data MWM day 1 - 4 | rawdata MWM day 1-4 | 145 | 20 | 2900 / 0 |
| raw data MWM day 5 | raw data MWM day 5 | 145 | 20 | 725 / 0 |
| Y-Maze | Y-Maze | 145 | 20 | 573 / 7 |

## Shared rules

1. Delete the ten `Mean` rows and ten `SEM` rows from each sheet. These are interleaved group summaries, not animal records. Recognize them by their row labels rather than fixed row numbers.
2. Retain every animal row and its original position relative to the other animal rows. All sheets contain the same 145 composite `(Cohort, Group, Sex, IRN)` keys, but Y-Maze starts in a different order from the MWM sheets; do not impose a shared sort.
3. Preserve the covariate values and cell types exactly, including Roman-numeral cohorts, letter groups, lowercase sex codes, and IRN suffixes such as `14627 A`. Do not strip the letter suffix or coerce all IRNs to integers.
One pre-existing cross-sheet type difference is retained: IRN 16006 is numeric in Y-Maze and text in MWM. Cross-sheet subject matching accounts for this without editing either workbook.
4. Preserve all measurements and blanks exactly. Distance remains in cm and velocity in cm/s; latency remains in seconds, including values such as 60.08. Percent-valued measurements retain their source scale. Do not replace missing Y-Maze values with zero.

## Sheet-specific rules

### MWM training, days 1–4

- Rename `raw data MWM day 1 - 4` to **`rawdata MWM day 1-4`**.
- Remove the blank leading row 1 and column A: the meaningful range starts at raw B2. After removing summaries, raw B2:Y168 becomes cleaned A1:X147.
- Keep a two-row header. Add **Covariates** over the first four columns, merged as A1:D1. The second row remains Cohort, Group, Sex, IRN.
- Keep the five metric blocks and their four Day 1–4 subcolumns in their original order: Latency (s), Distance (cm), Velocity (cm/s), Thigmotaxis %, Floating %.
- Retain the metric header merges, shifted one column left and one row up: E1:H1, I1:L1, M1:P1, Q1:T1, U1:X1.

### MWM probe, day 5

- Keep the sheet name **raw data MWM day 5** unchanged, including its spaces.
- Remove the blank leading row 1 and column A. Raw B2:J167 becomes cleaned A1:I146 after dropping the summaries.
- Keep the existing single-row headers: Cohort, Group, Sex, IRN, Target Zone Crossings, SE (%), NE (%), NW (%), SW (%). Do not merge headers or rescale the quadrant percentages.

### Y-Maze

- Keep the sheet name **Y-Maze** unchanged. Its table already starts in column A; do not remove a leading column.
- Collapse the two raw header rows into one. Retain Cohort, Group, Sex, IRN. Expand the merged metric labels and combine each with its week using comma-space:
  - Total entries, week 12
  - Total entries, week 13
  - Correct entries %, week 12
  - Correct entries %, week 13
- After deleting the summaries and collapsing the header, raw A1:H167 becomes cleaned A1:H146. The cleaned sheet has no merged cells.

## Verification and limits

The inferred rules reconstruct every nonempty value and header in all three cleaned sheets, with matching merged ranges and dimensions. Animal cells were compared as exact stored Decimal values or strings, with type checks; this is stronger than comparing displayed rounded values. Numeric zero remains distinct from a blank.
All 120 raw formula cells are in removed summary rows (20 in training, 100 in probe, zero in Y-Maze). Neither workbook contains formulas in retained animal cells. This does not require recomputing any animal measurement.
This reconstructs the analytical tables and headers, not workbook formatting, styles, themes, package bytes, or editing history. The exact original sequence of manual Excel actions cannot be inferred uniquely.
No new cleaned workbook has been generated. Before adapting these rules to CO28154, inspect its own sheet layout and header wording; do not copy CO28152 row numbers, week labels, animal counts, or sheet names blindly.

## Source fingerprints

- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28152 rawdata.xlsx`
  SHA-256: `0c745efc8990c2ed8b08e406fe53cec175c8a8319a2ab7a86a5a05edf45492fb`
- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28152-behavioral-data-cleaned.xlsx`
  SHA-256: `bc46a377af66fb50b42ffe5aa781b071f99a18a544f5a2a511e683627b98bfbf`

Detailed row mappings, removed row numbers, and the seven retained blank locations are in co28152-cleaning-evidence.json.
