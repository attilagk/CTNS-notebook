# Manuscript input restoration

Five exact-name Dropbox backups were downloaded and checked against Dropbox content hashes and notebook sheet/header expectations. No notebooks were executed; data values were not changed. Passing these checks does not yet establish reproduced scientific results.

| File | Action | Required sheets/headers | SHA-256 |
|---|---|---|---|
| CO26990-MWM-mean-distance-to-platform.xlsx | restored | passed | `2727433d849724d1d6f9a96c235b33184ba53f9fe4796f349666a248e6ff2740` |
| CO28152-MWM-mean-distance-to-platform.xlsx | restored | passed | `96b2d3f69aeb2aee84ca28567fc34beda529876ae913f0b1c745965971b6813d` |
| CO28152_NFL-GFAP_datasum.xlsx | restored | passed | `7baef3675cca8620880c3d21b0f9c953494d743d42b32e1a35411df09926d88d` |
| CO26990-behavioral-data-cleaned.xlsx | restored | passed | `6b8526b151d3b02cf8166f1d27428dff0e7608837d14692ca5e9702b54b1705a` |
| CO28152-behavioral-data-cleaned.xlsx | restored | passed | `bc46a377af66fb50b42ffe5aa781b071f99a18a544f5a2a511e683627b98bfbf` |

## Compatibility notes

- **CO28152-behavioral-data-cleaned.xlsx**: The backup uses Distance (cm) and Velocity (cm/s). src/mwm.py and the older notebook rename list use Distance (m) and Velocity (m/s). Covariates, daily latency, thigmotaxis, floating, and Y-Maze headers are present. No values, units, or headers were changed. Review this compatibility difference before interpreting distance/velocity results.

Destination: `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse`

## Remaining missing source paths in the nine manuscript directories

- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28152-biochemistry-cleaned-Abeta.xlsx`
- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28154-behavioral-data-cleaned.xlsx`
- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28154-biochemistry-cleaned-Abeta.xlsx`
- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/CO28154-biochemistry-cleaned.xlsx`
- `/home/attila/projects/CTNS/resources/QPS/5xFAD-mouse/amiloride-behavioral-data-cleaned.xlsx`
- `/home/attila/projects/CTNS/resources/testdata/bambi/kidney.csv`

CO28154 raw data already exist locally; reconstruction of its cleaned workbook is deferred until the independent raw/cleaned comparison. The amiloride workbook alias, biochemistry inputs, exploratory sample data, and missing generated results require further work. Existing audit and graph files remain historical snapshots. The professional-vault report was not modified.
