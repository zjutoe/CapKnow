# Phase 2-6 Correctness Repair and Revalidation Report

## Current Gate Status

Latest evidence gate: `PENDING INDEPENDENT REVIEW`.

The previous source/evidence package `dc6a837ac20aad967da76a69d50c0b5b2bfe7379` plus `4e235ab5baff6c9882100b390f39eeb4bf0ac22f` remains `REJECT`. The post-review source repair was frozen as `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40` and reviewed independently as an implementation repair. Phase 5/6 were formally rerun from that clean source and recorded in accepted evidence commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`. Phase 2-4 were then formally rerun from clean commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`. The combined Phase 2-6 evidence package is not accepted evidence until independent evidence review.

## Source Binding

- Baseline reference commit: `1648c8f201936e56f7eb0544b39c45cf0f431c9b`
- Source repair commit: `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`
- Current evidence commit: `1bfc6ced88cf3397f240c3e79d1996955e9d589f`
- Previous rejected source commit: `dc6a837ac20aad967da76a69d50c0b5b2bfe7379`
- Source binding: `git_commit`
- Worktree state before each run: clean source tree, with only the fixed output root excluded by the launcher gate.
- Phase 2-4 artifacts are bound to clean commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`; Phase 5/6 artifacts remain bound to clean source repair commit `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`.
- Superseded dirty-run source identifier: `source_snapshot_sha256=c5146b05b8ab21dbc61b781807d4e2c7e4530b4c3f0ed051defd6c37561f6175`
- Superseded dirty-run `git_head`: `b7f54fff90fb218dd57d04b2a7d90b9bd9387521`

## Repairs

1. Probabilistic posterior: log-space likelihoods, explicit impossible-observation semantics, incremental adaptive update without double-counting history, and shared MAP tie tolerance.
2. Identifiability: reject empty, invalid, duplicate, too-short, too-long, and non-binary declared inputs; use JSON-array state IDs.
3. Adaptive certificates: splitting-only built-in policies, loud custom-policy contract failures, strict recursive validation, and full tree serialization.
4. DSL: deterministic exact-solver signatures, concrete `ExecutionResult`, explicit missing-capability errors, boolean-only capability values, typed primitive semantics, strict non-boolean positive integer `LOOP` counts, and composition result collision checks.
5. Phase 5 evidence: fixed and adaptive runs now gate solver posterior maps, MAP states, and confidence against a batch posterior recomputed from complete observations; zero-noise adaptive runs also assert literal per-state depth, stopping flags, confidence, and response counts.
6. Phase 6 evidence: primitive, composite, held-out, certificate-transfer, and reproducibility claims now have executable literal or invariant oracles, including full output types and composite intermediate results.

## Tests Run

- Python: `3.13.9`
- pytest: `8.4.2`

| command | result |
| --- | --- |
| `python -m pytest -q tests/test_identifiability.py` | `7 passed in 0.01s` |
| `python -m pytest -q tests/test_certificate.py tests/test_adaptive_certificate.py` | `19 passed in 0.04s` |
| `python -m pytest -q tests/test_probabilistic.py` | `23 passed in 0.05s` |
| `python -m pytest -q tests/test_revalidation_common.py` | `1 passed in 0.03s` |
| `python -m pytest -q tests/test_dsl.py` | `18 passed in 0.06s` |
| `python -m pytest -q` | `74 passed in 0.14s` |
| `git diff --check` | `passed` |

No repository-level ruff, mypy, tox, setup.cfg, or pyproject static-check configuration was present.

## Experiments Run

| command | source commit | result artifact | result sha256 | manifest sha256 |
| --- | --- | --- | --- | --- |
| `PYTHONPATH=. python scripts/revalidation_phase2_identifiability.py` | `1bfc6ced88cf3397f240c3e79d1996955e9d589f` | `artifacts/phase2_6_revalidation/phase2_identifiability.json` | `fe0d5c959816fe9b6578eb1c45756f7d57079d914be6a5529e36c5ee5cea3a5c` | `fc31ab28ed66e8f54d808fcae28d94a5c44a8e495785211d219e296c66174bcd` |
| `PYTHONPATH=. python scripts/revalidation_phase3_fixed.py` | `1bfc6ced88cf3397f240c3e79d1996955e9d589f` | `artifacts/phase2_6_revalidation/phase3_fixed_regression.json` | `6b7186d7da60e7655036f88c39314264e9ee0004b272525ab11e9eae4744edef` | `acbdfaefcc0f64c8eac8db1f2e65744c3575a3cc8335966ca3ec6832d764a3c9` |
| `PYTHONPATH=. python scripts/revalidation_phase4_adaptive.py` | `1bfc6ced88cf3397f240c3e79d1996955e9d589f` | `artifacts/phase2_6_revalidation/phase4_adaptive_regression.json` | `1d00e83570164b7253c724cb68e4c5760c283d459a4097621bd88079d96dd39d` | `738fd13570b3ec4b3662ff4b84bf3e3076cebf49b7806af91c7076484237a9d3` |
| `PYTHONPATH=. python scripts/revalidation_phase5_robustness.py` | `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40` | `artifacts/phase2_6_revalidation/phase5_robustness.json` | `b554bafa863ace1dd4729ab3b0c435b3dfd0d0e28a3a8b5c9efe43c4b2457d2a` | `dee175d2eee039efece909d6e78d5c84c3bb86b63137a4197a6872e4404c20e2` |
| `PYTHONPATH=. python scripts/revalidation_phase6_dsl.py` | `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40` | `artifacts/phase2_6_revalidation/phase6_dsl.json` | `252d8043ea73a087b77be919bce91bc91e0b0cfaa40ecaca5887b69006abc5e0` | `814b3cc1320090677fe6d6feb73b5fe3804cb001ff96c445e238a1b4b14b487d` |

## Corrected Results

- Phase 2: chain, tree, and unstructured worlds are identifiable; the artificial full-vector collision world is not identifiable.
- Phase 3: exact, greedy, and random fixed certificates are valid on chain/tree/unstructured worlds; exhaustive exact search remains the accepted temporary method.
- Phase 4: entropy, balanced, and random adaptive policies have valid-run rate `1.0`; average depth and worst-case depth are reported separately.
- Phase 5: zero-noise MAP behavior matches deterministic results; zero-noise adaptive query counts match frozen literal entropy-tree depths for every state; zero-noise adaptive stopping flags, confidence, and response counts match the oracle; fixed and adaptive solver posterior consistency failures are both `0`; all sampled fixed/adaptive consistency rates are `1.0`; superseded adaptive robustness outputs are invalidated.
- Phase 6: all seven primitive DSL programs and composite DSL programs produce exact deterministic values and output types; composite and held-out programs expose expected intermediate results; held-out `FILTER+CONDITION->FILTER_CHECK` is absent from default rules and runs without a composite capability label; every reported DSL claim is backed by a frozen expected value or explicit invariant.

## Historical Results Invalidated

- Phase 2 claims based on silently filtered valid states are superseded.
- Phase 4 average-cost wording that used worst-case query count as average depth is superseded.
- Phase 5 robustness and adaptive stopping evidence generated before the posterior repair is not scientific evidence.
- Phase 6 Boolean membership-only execution evidence is superseded by the concrete execution contract.

## Remaining Limitations

- Phase 3 still uses exhaustive subset search by accepted exception.
- Tree-world mapping inputs still assume no parent with an empty child list.
- Phase 5 robustness curves are numeric JSON summaries; plotting remains optional and was not generated.
- The superseded dirty-run source was bound by `source_snapshot_sha256`, not by a clean commit. It remains invalidated historical context only.

## Independent Review

Historical strict read-only independent review verdict for the previous package: `ACCEPT`.

- That verdict was real historical context but is superseded by the later fresh-context review.
- The second fresh-context review verdict was `REJECT`.
- Confirmed second-review findings: adaptive Phase 5 consistency was not gated; Phase 6 had no executable expected-output oracles; DSL capability values used truthiness; DSL loop counts accepted `bool`/float boundary cases; zero-probability Bernoulli sampling used `<=`.
- This revision repairs those findings and regenerates the evidence package above.

Fresh strict read-only independent review verdict for the previous regenerated package: `REJECT`.

- Reviewer started with `fork_context=false`.
- Confirmed findings: Phase 5 zero-noise gate covered MAP but not adaptive query-depth/stopping semantics; the state-ID special-character test lacked brace-containing task IDs; the loop positive-integer test did not prove exact iteration count.
- This revision repairs those findings and regenerates the provenance-bound evidence package above.

Final strict read-only independent review verdict for the superseded repaired package: `ACCEPT`.

- Reviewer started with `fork_context=false`.
- No correctness or protocol findings were confirmed.
- Reviewer specifically checked adaptive posterior batch equivalence, zero-noise adaptive query-depth matching, brace-containing stable state IDs, exact loop iteration tests, Phase 6 executable oracles, strict DSL bool/loop contracts, Bernoulli zero boundary, source/artifact hash provenance, and report/artifact consistency.
- Reviewer did not rerun tests or experiments; acceptance relies on the recorded verification plus read-only inspection of files, diffs, manifests, and artifacts.

Latest follow-up repair addressed a later independent review that rejected this package for three remaining oracle weaknesses.

- Phase 5 fixed consistency now compares `solve_noisy_fixed_certificate`'s own posterior map, MAP state, and confidence against the batch posterior.
- Phase 5 zero-noise adaptive depth checks now use literal frozen expected depths and assert stopping flags, confidence, and response count.
- Phase 6 oracles now compare complete primitive `ExecutionResult` payloads and composite intermediate outputs.

Follow-up strict read-only independent review verdict for the superseded repaired package: `ACCEPT`.

- Reviewer started with `fork_context=false`.
- No correctness or protocol findings were confirmed.
- Reviewer specifically checked report provenance, artifact hash consistency, fixed-solver posterior exposure, Phase 5 fixed consistency, Phase 5 zero-noise adaptive stopping/depth/response-count gates, and Phase 6 full `ExecutionResult` and composite intermediate oracles.
- Reviewer did not rerun tests or experiments; acceptance relies on the recorded verification plus read-only inspection.

Latest strict read-only independent review verdict for source commit `dc6a837ac20aad967da76a69d50c0b5b2bfe7379` plus evidence commit `4e235ab5baff6c9882100b390f39eeb4bf0ac22f`: `REJECT`.

- Reviewer started with `fork_context=false` and did not rerun tests or experiments.
- Confirmed findings: reports exposed conflicting current evidence states; conditional execution did not require capabilities from inactive branches; `ADD` and `CONDITION` relied on Python coercion outside the declared DSL contract; exported probabilistic policies accepted boolean attempt counts.
- Artifact hashes and manifest source bindings were confirmed sound. Phase 2 response-matrix semantics, Phase 3 exhaustive minimal search, Phase 4 adaptive behavior, and Phase 5 posterior and aggregate calculations were also confirmed sound.
- Current evidence gate remains `PENDING INDEPENDENT REVIEW` until the combined Phase 2-6 artifacts are independently reviewed and accepted.

Post-review source repair commit `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`: implementation review `ACCEPT`.

- Reviewer started with `fork_context=false` and did not rerun tests or experiments.
- The review confirmed that the prior report conflicts, conditional branch capability contract, DSL type coercion, and boolean probabilistic attempt-count issues were repaired.
- The repair was then frozen as a source commit before formal Phase 5/6 revalidation.
