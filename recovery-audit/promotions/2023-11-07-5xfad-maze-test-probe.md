# Promotion: 2023-11-07-5xfad-maze-test-probe

- Branch: `promote-2023-11-07-5xfad-maze-test-probe-20261108`
- Source reproduction: `recovery-audit/reproductions/test-probe-20261107T161820Z/`
- Canonical source hash before promotion matched the reproduction source hash.
- Replaced the canonical notebook with the fresh-kernel verified reproduction, removing the temporary runner bootstrap and the interactive `%connect_info` cell. No scientific analysis cells were changed.
- Verification: all canonical cells completed in a fresh `base` kernel. Both Dirichlet test statistics and p-values match `recovery-audit/target-artifacts/target-artifacts.md` at displayed precision. Seven figures were generated.
- Environment dependency: `xuod/dirichlet` version 0.7, commit `78ce7c7cf45af75d473ae80d55cc2b6f4caadf78`, installed in `base`.
- Known warnings: the Dirichlet fitting routine reports divide-by-zero and invalid-value warnings, also present in the historical notebook output; retained and documented.
- Figures are generated under `named-figure/` relative to the notebook working directory. The reproduction directory preserves generated figure files.
