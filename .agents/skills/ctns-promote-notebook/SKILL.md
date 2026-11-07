---
name: ctns-promote-notebook
description: Promote a validated CTNS notebook reproduction into its canonical notebook with persistent fit-cache support, fresh-kernel verification, and an authorized Git/PR workflow. Use when replacing an original notebook after reproduction, not for initial execution alone.
---

# Promote a CTNS reproduction

Read the repository's `AGENTS.md`. Promotion replaces the canonical notebook with the reviewed, validated reproduction and makes normal future execution usable. It does not establish scientific agreement merely by copying executed outputs.

## Select the evidence and preserve the workspace

1. Inspect Git status/current branch, the canonical notebook, the chosen reproduction's source/executed notebook, manifest, report, target comparisons, and diagnostics. Verify source/input/helper identities and explain any differences since the run. Read the exact target-document section and its **`conda env:` declaration**; do not inherit a runner's hard-coded environment.
2. Confirm the selected run satisfies the requested targets and disclose remaining inference warnings. If a material discrepancy or stale provenance prevents the promised reproduction, report it and resolve the question with the user before declaring promotion successful. A user can explicitly choose to promote a documented partial result; preserve that distinction in the record.
3. Create a dedicated promotion branch from the intended current base, preserving unrelated local edits. Do not reset/clean the workspace or pull across a dirty tree blindly. No extra archive is needed unless requested; Git history retains the previous notebook. Notify the user before replacing an open notebook so browser Overwrite cannot erase disk changes.

## Install a self-contained canonical notebook

- Preserve scientific model definitions, data selection, seeds, units, and validated results. Remove reproduction bootstrap cells, temporary absolute run paths, fit monkey-patches, connection-info output, and other harness-only behavior. Retain imports/setup needed for ordinary execution and correct figure labels **before** saving exported figures.
- Default to persistent cache support, matching this project's established promotion workflow. Reuse `src/mcmc_cache.py` for supported Bambi models after inspecting the notebook's actual model construction. Store local fits under the notebook's sibling results directory, typically `../results/<notebook-name>/fit-cache/` relative to repository root; compute the correct path from the notebook working directory.
- Cache identity must distinguish actual data/schema/category ordering, evaluated contrasts/design matrices, formulas/family/links/centering, full-precision recursive priors, sampling settings, relevant input/helper/scientific-code hashes, and runtime/package versions. Unsupported models need an explicit identity extension or a documented alternative, not a weaker key.
- Adopt validated existing posterior draws only after verifying their mapping and provenance; record source checksums. Retain source fits. Valid compatible entries load; absent entries fit and persist; corrupt or mismatched entries fail visibly for inspection. Preserve an explicit refit path. Keep the `force_refit` boolean operational: toggling it alone must not change the scientific cache identity. Use `analysis_code_hash` where appropriate.
- The promoted notebook must work without the reproduction folder or temporary validation scripts. Large fits stay outside Git and need separate backup; never claim a push backs them up. Do not upload caches unless requested.

## Verify and document

- Validate notebook structure and run all canonical cells in a fresh kernel using the target-declared environment. Test compatible-cache execution with sampling instrumented to fail if called, verify expected cache hits, and compare target summaries and figures with the chosen reproduction. Keep this instrumentation out of the final notebook.
- Exercise cache misses/refit and identity invalidation with focused checks when adding/changing cache behavior; mocks or a small smoke model can avoid unnecessary full MCMC. Check data, priors, contrasts, settings, helper/scientific-code and version changes, checksum corruption, and the refit toggle. `recovery-audit/promotions/check_mcmc_cache.py` is the existing focused check; inspect its coverage before extending it.
- Record source run, provenance, cache location/adoption, verification commands/results, target comparisons, and retained warnings in a compact `recovery-audit/promotions/` report. Keep successful execution and scientific reproduction verdicts separate. Review the final source/output diff for harness residue and unrelated edits.

## Authorized Git and review workflow

- Follow the user's task-specific scope for commit, push, PR creation/commenting, merge, and local synchronization. Prior authorization within the task is sufficient; these skill instructions alone confer none. Commit only intended notebook/helper/report changes; inspect staged diff and ignore rules first.
- If requested, push the promotion branch, open a PR into the intended base, and post a focused `@codex review` comment naming scientific invariants, cache correctness, and artifact preservation. Use the GitHub connector when available or authenticated `gh`. Do not treat absent CI checks as successful checks.
- Wait for a completed review of the **latest PR head**. No reply, an emoji alone, or a review of an earlier commit is not clearance. Inspect review comments and required checks; fix substantive in-scope findings, validate, push, and request review again. Escalate scientific ambiguities or findings requiring broader scope to the user; if review remains unavailable, leave the PR open and report the blocker.
- Merge only if authorized, current-head review has no substantive unresolved findings, and required checks pass. Use expected-head protection when supported. After merging, switch to local main and pull fast-forward only when workspace state allows it; preserve unrelated edits and stop on conflicts/divergence rather than forcing history.
- Report PR/merge state, local branch synchronization, validation evidence, and any remaining warning. Never describe an open PR as merged or an unreviewed head as approved.
