# Latency notebook promotion

The canonical November latency notebook now loads compatible local fits through src/mcmc_cache.py. Its ten fitted models were adopted from the completed latency-20261107T031926Z reproduction. No original notebook archive was created; Git history preserves previous versions.

Cache location (relative to the notebook directory): ../../results/2023-11-07-5xfad-maze-latency/fit-cache/. Entries are addressed by identity hashes, with NetCDF checksums and JSON manifests. Adoption records name the original fit and its SHA-256. Source reproduction fits remain unchanged. These local files are outside this Git repository and need separate backup.

The identity includes actual model data, column/index metadata, categorical ordering, formula, family, links, centering, recursive full-precision prior parameters, explicit sampling settings, input CSV hashes, all notebook code-cell source, cache helper hash, Python and scientific package versions. Output/Markdown changes do not invalidate fits; code edits conservatively do. Set force_refit=True to resample. Unsupported identities and corrupt entries fail explicitly rather than silently returning an uncertain result.

Both fitting paths use four chains, 1000 tuning and 1000 retained draws, original seeds, and the original native PyMC sampler. Missing entries are sampled and saved; valid entries print Cache hit. The normal notebook does not monkey-patch Bambi or depend on the reproduction run directory. The connection-info cell was removed.

Validation: fresh bayes kernel Run All with Model.fit replaced by a function that raises if invoked; all ten fits must load without sampling. Compact numerical summaries agree with the reproduction. Target posterior figure and labels are retained. Helper checks cover changed data, priors, input identity, draw count, valid cache hit, and checksum corruption. No formulas, priors or scientific input values were altered.

The pre-existing PSIS-LOO Pareto-k warnings remain visible. Promotion does not resolve the reliability limitation of that model comparison. Target mean differences relative to the original historical table remain at most 0.001 at displayed precision; this promotion retains the same reproduced posterior draws.
