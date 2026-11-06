# Behavioral analysis units

Training imports now expose Distance in cm and Velocity in cm/s under the existing generic column names. The shared mwm.read_data_train reader uses unit-bearing Excel headers to convert meters by 100, leaves centimeters unchanged, and rejects missing or ambiguous headers. Source workbooks remain unchanged.

The October maze notebook uses the shared reader, and its two data_train-CO26990/CO28152 CSV exports in results/2023-10-13-5xfad-maze were regenerated. CSV readers must interpret these columns as cm and cm/s. The November latency and June behavior notebooks document this convention. Training and probe plots label physical units; probe source distances were already centimeters and were not converted.

October models containing Velocity require new fits and scientific review of priors on the cm/s scale. A cell after the fitting section exports these fits to results/2023-10-13-5xfad-maze/cm-units. It does not load historical fits. Full sampling has not been performed. Existing saved notebook outputs may reflect earlier runs; restart kernels and rerun to refresh them. Latency-only inputs are unaffected by the distance/velocity conversion.

Validation: all distance and velocity values across CO26990 (172 long rows), CO28152 (580), and CO28154 (648) matched their source measurements with the expected factor. Missing and ambiguous headers were rejected. Both regenerated CSVs reloaded with generic measurement columns. All 12 October models built without sampling. Distance and velocity training plots rendered with explicit unit labels. Temporary plotting/PyTensor caches were placed in /tmp for sandbox compatibility.
