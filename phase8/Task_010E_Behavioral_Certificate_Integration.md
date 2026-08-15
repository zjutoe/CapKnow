# Task 010E — Behavioral Certificate Integration

## Launch contract

```text
executor role: Spark
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010D prerequisite commit. Read master Sections
4.4, 10.1–10.2, 11, 12–13, and the accepted public identifiability, exact,
adaptive, and validation APIs. Do not alter those APIs to make Phase 8 pass.

## Objective

Convert raw evaluator counts into population response matrices and integrate the
accepted full-information identifiability and fixed/adaptive certificate machinery.
This task establishes exact structural outputs, not aggregate scientific claims.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/certificate_eval.py
tests/test_lm_bridge.py
```

No existing certificate/validation package file may change. Return an incompatibility
to `main` rather than patching the accepted baseline.

## Required implementation

- Construct one ordered `16 x 8` matrix per complete condition/scale/seed/checkpoint
  population from raw exact-success counts using the primary `52/64` threshold.
- Reconstruct sensitivity matrices only at `32/64`, `48/64`, and `64/64`; never
  select among thresholds by outcome.
- Run full-information identifiability before any certificate solver. Preserve exact
  collision groups and emit JSON `null` with reason `not_identifiable` for every
  behavioral fixed/adaptive certificate field when rows collide.
- For identifiable matrices, run the accepted exact fixed solver and independent
  validator; independently enumerate all `2^8` task subsets and retain every minimum
  certificate.
- Run entropy and balanced adaptive solvers and validators, then independently
  reconstruct leaf identities, per-state depths, average depth, and worst-case depth
  from serialized trees.
- Independently reconstruct assigned-state full-signature matching and
  ground-truth-certificate state identification. Do not report behavioral
  self-identification as ground-truth recovery.
- Keep task/state order explicit in every serialized result and fail on incomplete,
  duplicate, reordered, or mixed-population inputs.

## Acceptance checks

Add targeted tests for master Section 13 items 3 and 18–20, plus collision groups,
all-minimum enumeration, adaptive reconstruction disagreement, non-ground-truth
signatures, and incomplete-population rejection.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

## Stop boundary

Do not implement family metrics, paired aggregation, interpretation prose, shard
orchestration, manifests, benchmark code, or any experiment run. Empirical
non-identifiability is a valid future result, not an implementation defect. Do not
commit unless explicitly authorized. Return using the common format.
