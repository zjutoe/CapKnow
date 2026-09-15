# Capability Certificate Task Handoff 008

## Second-Review Repair for the Phase 2-6 Revalidation Package

## 1. Objective

Resolve every confirmed finding from the 2026-08-09 fresh-context strict review of the
GPT-5.5 Phase 2-6 repair package, regenerate provenance-bound evidence, update the reports,
and obtain a new independent gate verdict.

This is a narrow correctness and evidence repair. Do not redesign the DSL, probabilistic
solver, artifact format, or experiment framework. Apply KISS and YAGNI; fail early on
contract violations and add only the assertions needed to make the scientific claims
machine-checkable.

Current frozen package:

```text
baseline HEAD: 1648c8f201936e56f7eb0544b39c45cf0f431c9b
source_snapshot_sha256: 0a4f9fb3b2e6a2b02a63dea938a7c248253574b838194cd03ce42489b4af65cd
recorded full test result: 60 passed in 0.11s
second-review gate verdict: REJECT
```

The model-quality comparison is recorded separately in
`phase2_6_gpt53_gpt55_review_comparison.md` and is not an implementation requirement.

## 2. Review Protocol

The second review used the same model as the main thread, started with
`fork_context=false`, and received only the frozen handoff, current diff, artifacts,
acceptance criteria, protocol constraints, and verification already run. The reviewer
did not modify files or rerun tests and experiments.

Confirmed findings:

```text
Critical: 0
High:     1
Medium:   3
Low:      2
Verdict:  REJECT
```

All findings below are required repairs. Do not commit or restore an `ACCEPT` statement
until the affected tests and experiments pass and a new independent review accepts the
frozen package.

## 3. Risk Order

Execute in this order:

1. Add adaptive posterior consistency gating to Phase 5 revalidation.
2. Freeze and gate exact expected outputs in Phase 6 revalidation.
3. Tighten the two DSL input contracts.
4. Correct the zero-probability Bernoulli boundary.
5. Add targeted regression tests.
6. Run targeted and full tests.
7. Rerun the provenance-bound Phase 2-6 evidence package.
8. Update reports and hashes.
9. Freeze the package and run a fresh independent strict review.

Do not start revalidation while any code or targeted-test item remains unresolved.

## 4. P0: Gate Adaptive Sequential/Batch Posterior Consistency

### 4.1 Confirmed defect

`scripts/revalidation_phase5_robustness.py:202` checks sequential/batch posterior
consistency only for fixed traces. Adaptive runs begin around line 224 without an
equivalent check. The zero-noise gate compares only the MAP state.

The original adaptive double-counting defect can preserve MAP identity while inflating
confidence and crossing the stopping threshold too early. Therefore the current adaptive
confidence, threshold-reach, and query-count evidence is not independently revalidated.

### 4.2 Required repair

For every adaptive run:

1. Flatten `adaptive.query_history` into the same task/response observations represented
   by the complete adaptive trace.
2. Recompute one batch posterior from the original prior and the complete flattened trace.
3. Compare the recomputed posterior with the solver's final sequential posterior.
4. Compare the full state-probability map within a tight deterministic tolerance, not only
   MAP identity. Also require equal MAP state and confidence within tolerance.
5. Record the consistency result in the adaptive aggregate instead of leaving adaptive
   consistency fields null.
6. Add adaptive failures to the experiment failure gate. Any mismatch must make the script
   exit unsuccessfully and report enough world/noise/attempt/seed context to reproduce it.

Reuse the existing posterior-consistency helper or minimally generalize it. Do not add a
new experiment abstraction.

### 4.3 Acceptance criteria

- Every adaptive run has a non-null consistency measurement.
- Fixed and adaptive consistency failure counts are both explicit and equal to zero.
- Full posterior, MAP, and confidence agreement are checked.
- A regression of historical-observation double-counting would fail the script even if MAP
  remains unchanged.
- `phase5_robustness.json` and `phase5_report.md` state the fixed and adaptive checks
  separately and accurately.

## 5. P1: Make Phase 6 Expected Outputs Executable Oracles

### 5.1 Confirmed defect

`scripts/revalidation_phase6_dsl.py:21` freezes the held-out input but not its expected
output. The script records values returned by `_primitive_checks`, `_composite_checks`,
`_held_out_composition`, `_certificate_transfer`, and `_reproducibility`, but it does not
fail if those values are semantically wrong.

A regression in `FILTER`, `CONDITION`, a composite sequence, certificate transfer, or
deterministic signatures can therefore produce a successful script and new artifact.

### 5.2 Required repair

Define literal expected results from the accepted DSL contract for:

- all seven primitive executions;
- the default composite execution;
- the held-out `FILTER + CONDITION -> FILTER_CHECK` execution;
- certificate-transfer invariants;
- deterministic reproducibility invariants.

The expected values must be independent literals or explicit invariants, not values copied
from the actual result during the same run. Compare actual results against them and raise on
any mismatch before writing successful evidence.

Record the expected values or invariant definitions in the manifest config so that the
artifact states what was tested. Keep the current direct script structure; do not introduce
a generic assertion or experiment framework.

### 5.3 Acceptance criteria

- A wrong primitive or composite result makes the script fail.
- A held-out output mismatch makes the script fail.
- Certificate transfer requires the declared state-count and certificate-cost relations.
- Repeated signatures must be equal and match the frozen deterministic expectation.
- The held-out input and expected output are both present in the provenance-bound config.
- `phase6_dsl.json` and `phase6_report.md` distinguish observed outputs from acceptance
  oracles.

## 6. P1: Reject Non-Boolean Capability Values

### 6.1 Confirmed defect

`capability_certificate_lab/dsl/executor.py:125` uses:

```python
bool(state_has.get(op_id, False))
```

This grants a capability for any truthy malformed value. For example,
`{"ADD": "false"}` executes `ADD` successfully.

### 6.2 Required repair

When a capability key is accessed:

- require its value to be an actual `bool`;
- reject any other type with a clear contract error;
- retain `MissingCapabilityError` for an absent key or an explicit `False` value;
- do not coerce strings, integers, containers, or custom objects through truthiness.

Use the existing DSL exception hierarchy. Do not add schema libraries or normalize malformed
input.

### 6.3 Required tests

- `{"ADD": "false"}` raises instead of granting `ADD`;
- at least one truthy non-boolean value raises;
- `{"ADD": False}` still raises `MissingCapabilityError`;
- `{"ADD": True}` still executes normally.

## 7. P1: Require a Non-Boolean Positive Integer Loop Count

### 7.1 Confirmed defect

`capability_certificate_lab/dsl/executor.py:108` checks only
`program.max_iterations <= 0`. A float such as `1.5` passes structural validation and later
raises an incidental `TypeError` in `range()`. Because `bool` is a subclass of `int`, `True`
is accepted as one iteration.

### 7.2 Required repair

Validate `LoopNode.max_iterations` before execution:

- type must be exactly an integer contract, excluding `bool`;
- value must be greater than zero;
- invalid values must raise `InvalidProgramError` with a clear message.

Do not silently cast values with `int()`.

### 7.3 Required tests

- `1.5`, `True`, `False`, zero, and a negative value are rejected as invalid programs;
- a positive integer still executes the expected number of iterations.

## 8. P2: Correct the Bernoulli Boundary

### 8.1 Confirmed defect

`capability_certificate_lab/probabilistic/response_model.py:48` samples with
`generator.random() <= p_yes`. If `p_yes == 0.0` and the RNG returns exactly `0.0`, the code
emits a positive response for a zero-probability event.

### 8.2 Required repair

Use the standard Bernoulli comparison:

```python
generator.random() < p_yes
```

Do not add epsilon handling or probability clipping.

### 8.3 Required tests

Use a minimal deterministic RNG stub returning `0.0` and verify that `p_yes == 0.0`
produces response `0`. Retain existing probability-range and reproducibility behavior.

## 9. Report and Gate Status

`phase2_6_revalidation_report.md` currently records an earlier independent `ACCEPT`. That
was a real historical review, but it is superseded by the second-review `REJECT`.

Required report handling:

1. Preserve the earlier verdict as historical context; do not present it as the current gate.
2. Record the second-review findings and `REJECT` status.
3. After repairs and reruns, record what changed and bind the new artifact hashes.
4. Replace the current gate with `ACCEPT` only if a new fresh-context independent review
   returns `ACCEPT`; otherwise keep `REJECT` and record unresolved findings.

Do not describe a successful script exit as scientific validation unless the script contains
the required semantic oracle.

## 10. Verification

Run after all code and test changes:

```text
python -m pytest -q tests/test_probabilistic.py
python -m pytest -q tests/test_dsl.py
python -m pytest -q
```

Run repository static checks if a configuration now exists. Otherwise state exactly that no
repository static-check configuration is present.

Then rerun the complete small evidence package from one frozen source state:

```text
PYTHONPATH=. python scripts/revalidation_phase2_identifiability.py
PYTHONPATH=. python scripts/revalidation_phase3_fixed.py
PYTHONPATH=. python scripts/revalidation_phase4_adaptive.py
PYTHONPATH=. python scripts/revalidation_phase5_robustness.py
PYTHONPATH=. python scripts/revalidation_phase6_dsl.py
```

Although only Phase 5 and Phase 6 semantics change, the current report and manifests bind all
five phases to one source snapshot. Rerunning all five keeps that provenance contract simple
and explicit.

Record:

- actual command strings, including `PYTHONPATH=.`;
- baseline HEAD and dirty/clean state;
- the new source snapshot hash;
- exact configs, worlds, seeds, and expected-output oracles;
- result and manifest SHA256 values;
- targeted and full test results.

## 11. Final Independent Review

After implementation, tests, experiments, artifacts, and reports are complete:

1. Stop all writers and freeze the working-tree diff and evidence package.
2. Spawn one strict read-only reviewer with `fork_context=false` and the same model as the
   main thread.
3. Provide only this handoff, the frozen change, bound artifacts, acceptance criteria,
   protocol constraints, and completed verification.
4. Require findings ordered by severity with exact file/line, failure scenario, impact, and
   minimal correction.
5. Require explicit checks for adaptive posterior batch equivalence and Phase 6
   claim-to-oracle mapping.
6. The main thread resolves every confirmed finding and owns the commit gate.

## 12. Completion Criteria

This handoff is complete only when all conditions hold:

- all six second-review findings are fixed;
- required targeted regressions pass;
- the full suite passes;
- adaptive fixed/batch posterior consistency is machine-gated with zero failures;
- every Phase 6 scientific claim is backed by a frozen expected value or explicit invariant;
- all five revalidation scripts succeed from one newly bound source state;
- reports match the regenerated JSON and hashes;
- the latest independent review verdict is `ACCEPT`;
- no unresolved higher-severity or protocol finding remains;
- no unrelated refactor, compatibility layer, or speculative abstraction was introduced.

Current commit gate: `REJECT`.
