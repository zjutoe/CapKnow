# Phase 2-6 Correctness Repair and Revalidation Report

## Current Gate Status

Current gate: `PENDING CLEAN COMMITTED RERUN`.

The artifacts and hashes below were generated from a dirty worktree and are retained only as historical context. They must not be used as accepted scientific evidence until Phase 2-6 are rerun from a clean committed source revision.

## Source Binding

- Baseline reference commit: `1648c8f201936e56f7eb0544b39c45cf0f431c9b`
- Superseded dirty-run source identifier: `source_snapshot_sha256=c5146b05b8ab21dbc61b781807d4e2c7e4530b4c3f0ed051defd6c37561f6175`
- Superseded dirty-run `git_head`: `b7f54fff90fb218dd57d04b2a7d90b9bd9387521`
- Required replacement binding: exact clean source commit plus output hashes; source-package hashes are not accepted as the source identity.

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

| command | result artifact | result sha256 | manifest sha256 |
| --- | --- | --- | --- |
| `PYTHONPATH=. python scripts/revalidation_phase2_identifiability.py` | `artifacts/phase2_6_revalidation/phase2_identifiability.json` | `fe0d5c959816fe9b6578eb1c45756f7d57079d914be6a5529e36c5ee5cea3a5c` | `a29d3fbde9eb6b46bc211e50d53050ef4fe782170af87759db29b61bb2ca9187` |
| `PYTHONPATH=. python scripts/revalidation_phase3_fixed.py` | `artifacts/phase2_6_revalidation/phase3_fixed_regression.json` | `0ce0e94c970baf11b1abebfce0ee4df39caffac75c07fcb3f9b59d27ec6e78fe` | `488c60d78dfdc340a755a6d0d41d831399ac9ce2c9e8349f578c496a68379de3` |
| `PYTHONPATH=. python scripts/revalidation_phase4_adaptive.py` | `artifacts/phase2_6_revalidation/phase4_adaptive_regression.json` | `321ffaff8d13b235802fbcf41cbaf48665ae3e07d3aa14f7e9f05940ecb82005` | `c6a6ae90d5749a1debb2aded6ab2646e9c3fe34df2483e6a9f1fc813d70ba06b` |
| `PYTHONPATH=. python scripts/revalidation_phase5_robustness.py` | `artifacts/phase2_6_revalidation/phase5_robustness.json` | `202bb85f82dfcd524e2e2bc351ea1b830d13e86942142a17f0404dfd11093de2` | `c10516b6cacbac6d622b1a1541c634a54822984818b8a6994031b7678f311f12` |
| `PYTHONPATH=. python scripts/revalidation_phase6_dsl.py` | `artifacts/phase2_6_revalidation/phase6_dsl.json` | `05d98a08718271eb8af74ca053932037381f97b771bc1169d2e0a4108c1b1e73` | `e7719ba32e262afb82a37650b70be72ff9cc57207deaffd5aa482bbecefd8c0e` |

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
- The superseded dirty-run source was bound by `source_snapshot_sha256`, not by a clean commit.

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

Current gate: `PENDING CLEAN COMMITTED RERUN`.
