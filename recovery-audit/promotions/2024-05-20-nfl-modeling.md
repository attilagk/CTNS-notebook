# NFL notebook promotion

Source reproduction: recovery-audit/reproductions/nfl-20261107T195833Z.
All four target figures and deterministic tables reproduced; exploratory diagnostics
and warnings are documented in that run's REPORT.md.

The canonical notebook includes import setup, pandas/xarray compatibility changes and
persistent mcmc_cache support with force_refit. Cache identity includes the four input
hashes, NFL/cache helper hashes, scientific code, actual model data/categories/design,
priors, formula/family/link and runtime versions. Sampling settings remain empty kwargs
(the captured runtime defaults); force_refit does not change identity. Original unseeded
sampling remains unseeded. Three validated cache entries were copied byte-for-byte from
the isolated results/fit-cache to ../results/2024-05-20-nfl-modeling/fit-cache. Originals
remain. Cache JSON files contain full identities and NetCDF checksums.

verify_nfl.py runs the canonical notebook in a fresh base kernel with Model.fit raising
if invoked: three cache hits, no sampling. It removes its bootstrap before saving outputs.
The canonical notebook needs neither a reproduction folder nor temporary validation scripts.
Normal execution saves outputs under the notebook and sibling results directory.
check_mcmc_cache.py verifies valid hits, misses, forced refit, failure propagation,
data/prior/contrast/settings/context/version invalidation, checksum rejection and refit toggle.
See nfl-verification.json and nfl-cache-check.log for results.

Retained warnings: pandas future categorical warnings and the exploratory forest plot's
historical sigma variable exclusion no longer matching the current sigma name. The four
targets are unaffected. No extra archive: the previous notebook is retained in Git history.
Large fits and recovered inputs are local and require a separate backup.
