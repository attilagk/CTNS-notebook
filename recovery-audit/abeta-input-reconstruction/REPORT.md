# Amyloid input reconstruction

Restored CO28152-biochemistry-cleaned-Abeta.xlsx and
CO28154-biochemistry-cleaned-Abeta.xlsx under resources/QPS/5xFAD-mouse.
Each contains one Abeta sheet, 60 animal rows, five identifying columns, and
8 measurement columns under two header rows: CX FA, HC FA, CX DEA, HC DEA,
with Aβ40 and Aβ42 under each region/fraction. No unit conversion or rounding.

## Sources and independent reconciliation

CO28154: the user-linked Dropbox CO28154 rawdata biochemistry.xlsx, sheet Aß.
The downloaded bytes match the preserved local raw workbook.

CO28152: the user-linked CO28152_Report_final.pdf, sections 9.3.1.1 and
9.3.1.2 on report/PDF pages 64–66, was downloaded, text-extracted and visually
reviewed. It prints rounded numbers for 42 TG animals and identifies three
FA outliers. Independent extraction matched all 336 PDF measurement tokens,
metadata and outlier flags to the existing local CO28152_IV_data all.xlsx,
sheet Abeta. That verified raw workbook supplies full precision and the 18
additional WT rows absent from these report tables. The PDF alone cannot
recover those exact values or the full original row set. No rounded PDF
values were represented as full-precision recovered measurements.

## Cleaning

Remove the empty leading column, decorative headers and group mean/SEM rows.
Preserve all animal identifiers (including suffixes), metadata and original
row order. Convert starred outlier tokens and n.m. to blank numeric cells;
retain every unflagged numeric value, including small WT measurements.
No animals or complete rows are discarded when their assays are missing.
extraction.json lists every excluded source cell and token.

After cleaning, 339 numeric measurements remain in CO28152 and 291 in
CO28154. The remaining 330 cells are missing. PDF extraction is preserved
in pdf-extracted-values.csv, including outlier flags and printed precision.
receipt.json records source/download identities and installed file hashes.

## Validation

Both XLSX files were read back and their metadata, unique IDs, numeric values
and missingness compared to extraction. The notebook's actual import and
engineering code produces 120 wide rows and 960 long rows. Selected original
OLS code reproduces all 66 saved parameter/SE/p-value cells for CO28154
TUDCA and HCQ and all 90 for CO28152 TUDCA or HCQ, at displayed precision,
including male-only models. Validation deliberately avoids notebook imports
unneeded for those cells and does not write canonical result files or plots.
This is input recovery and targeted validation, not full notebook reproduction.

The spreadsheet runtime exports successfully; its visual preview timed out.
Headers/schema, missingness and values were verified by independent XLSX reads.
The PDF source tables were rendered and visually reviewed.

Inventory refresh records recovered/passed status and raw/PDF provenance edges,
while leaving the original missing-file baseline unchanged. Work is on the
recover-abeta-inputs branch and is not committed or pushed by this task.

## Repeat

Run extract.py RESOURCE_DIR DOWNLOADED_CO28154_XLSX PDF_LAYOUT_TEXT STAGE_DIR
with base Python. Run build.mjs using the bundled @oai/artifact-tool runtime,
then validate.py STAGE_DIR. Use pdftotext -layout for the supplied PDF source.
Check receipt hashes and never overwrite different existing inputs.
