---
name: ctns-reproduce-notebook
description: Execute and validate CTNS research notebooks against their declared target artifacts in the notebook-specific conda environment, preserving data, inference provenance, and recoverable fits. Use for reproduction work; promotion of a reproduction is a separate task.
---

# Reproduce a CTNS notebook

Work from the CTNS notebooks repository root and read its `AGENTS.md`. Produce a reviewable execution and artifact-validation record without automatically replacing the canonical notebook.

## Establish the run contract

1. Locate the requested canonical notebook and its exact `## <notebook-name>` section in `recovery-audit/target-artifacts/target-artifacts.md`. Read through the next peer heading. Take the **`conda env:` value from that section**, not another notebook's environment. Resolve referenced screenshots relative to the target document; it contains both Obsidian image links and embedded images.
2. Extract all requested targets, reference values, units, and figures. Record comparison criteria before execution. Deterministic results should agree at the documented precision; stochastic results need comparison of uncertainty, Monte Carlo error, and scientific interpretation, not exact draws. If a material acceptance tolerance or scientific choice is ambiguous, ask in chat while continuing independent inspection. Preserve the target document as user-authored evidence.
3. Inspect Git status, the dependency/recovery inventory, load/save cells, and shared helper code. Confirm required inputs and upstream generated files exist and have the correct schemas/units. Run prerequisites in dependency order only within the requested scope; report unavailable prerequisites rather than silently substituting data. Inspect missing dependencies directly, not only stale CSV status.
4. Select or create the task branch consistently with the user's current workflow. Record the Git revision and dirty paths. Notify the user before editing an open notebook and explain Jupyter Revert versus Overwrite when relevant.

### Check dependencies before execution

- Build a notebook-specific dependency checklist from code-cell imports, `%run`/`%load` magics, local helper imports, file reads, and invoked external programs. Include packages imported indirectly by the helpers that will run. Do not mistake imports in inactive/commented cells or optional error-handling branches for mandatory requirements without checking their use.
- In the environment declared for this notebook, probe `sys.executable` and import each required Python module in a fresh process; check command-line tools separately. Verify local imports resolve to the expected file and inspect required data paths and workbook sheets. Record the environment and concise check results in the run manifest before starting the kernel.
- When an import is missing, identify its owning package and source first. Install it into the declared environment only when the mapping is clear and installation is within the user's request; then rerun the probes and record package/version changes. Do not silently switch environments or install a similarly named package based on guesswork.
- For a missing custom/local module, search the repository, configured source directories, and available project resources for the expected implementation; use saved notebook output or historical paths as clues, not proof of availability. Do not replace it with an unrelated package or recreate scientific algorithms from output alone. If no verified source or equivalent exists, stop execution at that dependency, record the precise blocker, and report what source is needed.
- Likewise, if required input files are absent, identify recovery candidates and verify content/schema before using them. Do not proceed with a substitute input merely because its filename looks plausible.

## Execute with provenance and recoverable results

- Create a timestamped directory under `recovery-audit/reproductions/`. Capture the source snapshot/hash, input and relevant helper hashes, target-document hash, exact interpreter, conda package list, seeds/sampler settings, timestamps, execution log, and any execution-only instrumentation.
- Resolve the declared environment's interpreter and explicitly use it for the kernel. `conda run -n <declared-env> python ...` is one launcher option; confirm `sys.executable` inside the kernel. Never silently fall back to `base` or change packages to fix an interpreter mismatch.
- Run all cells in a fresh kernel with errors stopping execution. Use the expected notebook working directory or an isolated mirrored layout preserving relative `resources/` and `results/` relationships. Set repository `src/` and `/home/attila/src` import paths explicitly if needed. Redirect every discovered output path, including CSVs and NetCDFs, so the experiment does not overwrite canonical results accidentally. A resource symlink does not make inputs read-only: inspect write operations.
- Existing examples: `recovery-audit/reproductions/run_latency.py`, `run_co28154.py`, and their `compare_*.py` companions. Inspect before adapting: they hard-code notebook names, `bayes`, artifact variables, and instrumentation. The older CO28154 cache key is not a general model-identity contract. A now-cached canonical notebook can load fits without calling `Model.fit`; instrumentation must account for this.
- Preserve expensive fits as NetCDF with model mapping, checksums, provenance, and diagnostics. For compatible Bambi models prefer `src/mcmc_cache.py`; verify identity covers actual data, design/contrasts, full priors, scientific code/helpers, sampling settings, and versions. Do not adopt unidentified fits or use existence-only cache checks. Resume only when the relevant provenance still matches; otherwise create a new run and preserve previous evidence.
- Fix demonstrated implementation failures in the requested notebook/helpers, document the cause, and rerun affected work followed by fresh-kernel end-to-end validation. Do not remove difficult models, change cohorts/priors/units, or suppress warnings just to obtain a pass. Avoid rerunning unaffected MCMC when verified compatible fits suffice.

## Validate and hand off

- Compare every target table numerically with row/column alignment and units; inspect every target figure for data, labels, groups, intervals, and qualitative interpretation. Record reference, reproduced result, difference, criterion, and verdict. A plot looking plausible does not validate a numerical discrepancy.
- For MCMC record divergences, unrounded R-hat, bulk/tail ESS, Monte Carlo error where relevant, and finite log likelihood when required. Retain PSIS-LOO/Pareto-k and other consequential warnings. Distinguish target-producing models from exploratory models; disclose problems in either. Do not introduce universal diagnostic thresholds as if the user specified them.
- Write a compact report and machine-readable manifest/validation status with separate **execution**, **reference reproduction**, and **inference quality** outcomes, per-target evidence, unresolved issues, and exact artifact/cache paths. A completed run with a substantive target discrepancy is partial reproduction, not promotion-ready by default.
- Reuse existing per-run output conventions where useful; no new global schema is required. Update the recovery inventory only when requested or needed to record newly recovered/generated dependencies, preserving baseline history and excluding run copies from notebook coverage.
- Review the diff and ignore behavior. Keep large fits and transient notebook copies local; retain compact reports/provenance and selected target images. Finish by reporting what reproduced, what remains unresolved, and where the evidence and fits reside. Commit/push only when authorized. Do not promote as an implicit final step.
