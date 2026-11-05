# CO28154 behavioral data reconstruction

Applied the independently verified CO28152 cleaning rules, adapted to CO28154 headers and saved notebook output. The source workbook is unchanged. No animals were excluded, measurements converted, missing values imputed, or records reordered within a sheet.

| Sheet | Animal records retained | Summary rows removed | Cleaned range |
|---|---:|---:|---|
| MWM day 1-4 | 162 | 20 | A1:X164 |
| Probe Trial day5 | 162 | 20 | A1:I163 |
| Y-Maze | 163 | 20 | A1:H164 |
| body weight | 196 | 24 | A1:S198 |

## Adaptations

- CO28154 sheet names retained, including MWM day 1-4 and Probe Trial day5, as required by its notebooks.
- Training headers retain their metric/day hierarchy; Covariates is added over the first four columns.
- MWM covariates reordered from Cohort, Group, IRN, Sex to Cohort, Group, Sex, IRN to match the saved notebook output and the reader index assumptions. Values and cell types remain unchanged.
- Y-Maze metric/week headers flattened to the established single-row names.
- The additional body weight sheet is retained and cleaned using the shared summary/padding rules. Its meaningful body weight [g] title, original column order and lowercase sex header are retained. Its twelve lowercase mean and twelve SEM rows are removed.
- Different sheet populations are retained: 162 training, 162 probe, 163 Y-Maze and 196 body-weight records. No subject intersection or group filtering was imposed.

## Verification

Every retained animal cell was compared to its source cell at exact stored precision and type. The source hash and output hash were checked. Header ranges, merged headers and notebook sheet names were checked. Workbook previews were reviewed. Full notebook execution and reproduction of fitted statistical results are separate work.

Output: `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28154-behavioral-data-cleaned.xlsx`

Output SHA-256: `fc00b858abb33944755f63d090c263ce888c38629a0b235fce3145b5f56ef101`

Raw SHA-256: `2d13f5c02ebf57c5517cf436f0378352c028b8e14b75e78b7b9d32c4e3a9ef8f`
