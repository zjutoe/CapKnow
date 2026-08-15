# Task 010F — Metrics, Sensitivity, and Aggregation

## Launch contract

```text
executor role: Spark
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010E prerequisite commit. Read master Sections
10.3–10.4, 11, 13, 15, and 17. Scientific interpretations are predeclared output
constraints, not choices delegated to the executor.

## Objective

Implement pure, independently testable metric and aggregation functions that turn
raw evaluation/certificate results into report-ready structured records without
changing the primary response definition or statistical unit.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/evaluator.py
capability_certificate_lab/lm_bridge/certificate_eval.py
tests/test_lm_bridge.py
```

Do not create a report or runner in this task.

## Required implementation

- Compute primitive, seen-composition, and held-out mastery, false-positive answer,
  correct-refusal, and all-state answer-success metrics with the exact `32/32`,
  `12/36`, and `4/12` positive/negative cell denominators.
- Compute primary-matrix false positives, false negatives, family-stratified bit
  errors, and behavioral regret over exactly 128 cells.
- Emit ground-truth identification, exact full-signature matching, fixed task
  ratio/savings, canonical and all-minimum Jaccard statistics, adaptive depths, and
  earliest identifiable checkpoint with exact null behavior.
- Report task-family query units and raw-generation cost separately, with the latter
  equal to family-query count times 64.
- Compute A/C response-target byte/token diagnostics by state/task family, including
  minima, medians, means, maxima, answer/refusal lengths, unavailable ratios, and
  paired differences. These fields cannot alter records or weights.
- For each sensitivity matrix emit primary Hamming distance, collisions,
  identifiability, certificate fields, canonical-set Jaccard, and
  minimum/mean/maximum pairwise Jaccard over the Cartesian product of the primary and
  sensitivity all-minimum-certificate families when defined.
- Emit per-seed values first. Aggregate a certificate metric only when all three
  seeds define it; otherwise emit JSON `null` while retaining defined seed values.
  Apply the same rule to paired A-C and A-B vectors and summaries.
- Emit machine-readable interpretation strata: primitive readiness; exact seen
  composition cells; held-out outcomes by `parent_ready` and explicit/indirect style;
  identifiability scope; and certificate-comparison scope. Never infer a broad claim
  from these records.

## Acceptance checks

Add targeted tests for master Section 13 items 17 and 22, all declared denominators,
undefined-seed handling, family-query conversion, threshold non-selection, target
diagnostics, parent-readiness strata, and style splits.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

## Stop boundary

Do not implement execution shards, manifests, status files, resource measurement,
formal reports, or scientific conclusions. Do not add p-values, confidence
intervals, pooled pseudo-replicates, fallback thresholds, or nearest-state mapping.
Do not commit unless explicitly authorized. Return using the common format.
