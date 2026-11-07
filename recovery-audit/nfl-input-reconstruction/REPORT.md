# NFL source input reconstruction

Reconstructed the two animal-list CSVs and the longitudinal NF-L worksheet required
by `2024-05-20-nfl-modeling`, using the Dropbox workbooks provided by the user.
This is input reconstruction and reader validation, not full notebook reproduction.

## Sources and rules

- `Animal Lists NIA in vivo projects.xlsx`: extract the CO28152 and CO28154 sheets,
  remove the empty leading column and empty trailing rows, preserve all 160 and 192
  animals respectively, and preserve IRN suffixes and source metadata.
- Export dates as ISO dates and calculate `Lifespan (days)` as death minus birth.
  One malformed date, `04:07.2023` for CO28152 animal 15998, is interpreted as
  4 July 2023. This animal is not among the 59 animals in the notebook's NF-L table.
- `CO28154 rawdata biochemistry.xlsx`, `NF-L invivo`: discard blank/header rows
  and aggregate mean/SEM rows; retain all 60 animal rows and five metadata columns.
  Map baseline/week4/week8/week12 to `NF-L week 0/4/8/12` in sheet
  `NFL in vivo clean`. Preserve numeric values without rounding or unit conversion.
- Convert seven starred measurements and one `n.m.` measurement to empty cells.
  The notebook's saved dataframes support these exclusions. See
  `transformations.json` for every source cell, IRN, and excluded token.
- The cleaned workbook contains **only `NFL in vivo clean`**. Other assays in the
  source workbook, including terminal NF-L, GFAP, amyloid and inflammation, have
  not been reconstructed. The raw workbook is retained separately.

## Validation and discrepancy

The actual `src/nfl.py:read_nfl_data` reader successfully loads both studies.
All 59 CO28152 and 60 CO28154 NF-L animals join to their animal lists without
loss or duplicate IDs. All 2,023 cells in the two resulting dataframes match
the notebook's historical HTML tables at their displayed precision, including
missingness, change scores, lifespan and history. Exported XLSX values were
also checked against the extracted raw numeric values after round trip.
The spreadsheet runtime's visual preview timed out; verification therefore
covers data and notebook compatibility, rather than a rendered layout review.

The source workbooks disagree on sex for CO28152 animal `14631 A`: `f` in the
animal list, `m` in the existing cleaned NF-L workbook. Both source values are
preserved. The reader takes Sex from the NF-L workbook, so this reconstruction
does not change the notebook's historical sex assignments. Scientific resolution
of that disagreement remains open.

`validation.json` records checks; `receipt.json` records source links, file hashes,
destinations and installation time. The recovery refresh consumes the receipt
while keeping the original missing-file baseline unchanged.
The refresh also excludes reproduction copies under `recovery-audit/` from
canonical notebook scans and records the three raw-to-clean provenance edges.

## Repeating the reconstruction

Use the base environment's Python to run `reconstruct.py SOURCE_DIR STAGE_DIR`.
Build the XLSX from `STAGE_DIR/nfl-values.json` with `build-workbook.mjs` and the
bundled `@oai/artifact-tool` runtime. Run `validate.py STAGE_DIR RESOURCE_DIR
VALIDATION_JSON` before installation. Existing differing outputs must never be
overwritten. Exact hashes apply to the installed files; regenerating XLSX may
produce different archive metadata, so validate its contents independently.
