# Capability Certificate Task Handoff 009: Phase 7 Structural Compressibility

## 1. Milestone Boundary

Branch:

```text
milestone/phase7-structural-compressibility
```

Accepted baseline:

```text
main commit: 02380501e5def5d9f624578a93bc0580de1e030a
```

Phase 2-6 is accepted evidence. Do not edit or regenerate its source-bound artifacts,
manifests, reports, or revalidation scripts in this milestone. Phase 7 starts a new
evidence boundary.

No implementation or experiment is authorized until this handoff passes a
fresh-context milestone-start scientific protocol review.

## 2. Scientific Objective

Determine exactly when deterministic membership observations permit a fixed
certificate smaller than the task universe, and separate three claims that were
previously easy to conflate:

1. prerequisite structure alone reduces the number of legal states;
2. observational redundancy permits a smaller fixed certificate;
3. an ordered state population permits an adaptive decision tree to outperform a
   fixed certificate under a uniform state prior.

This milestone must explain why the accepted chain, tree, and unstructured worlds
required every task in their fixed certificates, then introduce the smallest
parameterized world families that exhibit genuine, protocol-visible fixed
compression.

## 3. Non-goals

Do not add any of the following:

- neural or language models;
- noisy responses, repeated probes, or Phase 5 inference;
- learned capability graphs;
- unequal priors or unequal task costs;
- ILP, SAT, CP-SAT, or new third-party dependencies;
- a generalized experiment framework;
- claims about real LLM capability measurement;
- confidence intervals or significance tests for the matched random controls.

The exact solver remains exhaustive for this bounded milestone. Scaling it is a
separate future milestone.

## 4. Frozen Mathematical Contract

Let the task universe be `Q`, the declared state population be
\(\mathcal K \subseteq 2^Q\), and the deterministic response be

\[
Y(K,q)=\mathbf 1[q\in K].
\]

A fixed certificate `S ⊆ Q` is valid exactly when the restriction map

\[
K \mapsto K\cap S
\]

is injective on \(\mathcal K\). Equivalently, `S` is a hitting set of all pairwise
symmetric differences `K_i △ K_j`.

Use the following names without overloading the existing identifiability report's
`compression_ratio` field:

- `fixed_task_ratio = exact_certificate_size / task_count`;
- `fixed_task_savings = task_count - exact_certificate_size`;
- `information_lower_bound = ceil(log2(state_count))` for `state_count > 1`, else
  `0`;
- `fixed_excess_over_information_bound = exact_certificate_size - information_lower_bound`.

### 4.1 Single-coordinate witness theorem

A task `q ∈ Q` is individually indispensable if there are declared states
`K_0, K_1` such that

\[
K_0\triangle K_1=\{q\}.
\]

Every valid fixed certificate must contain an individually indispensable task. If
every task has such a witness pair, the unique minimum certificate size is
\(|Q|\).

Proof obligation: if `q ∉ S`, the witness pair has identical restrictions to
`S`, contradicting injectivity. The implementation must expose the witness pairs
as diagnostics; it must not infer indispensability merely from the exact solver's
selected output.

This theorem explains the no-compression result for the accepted chain and full
unstructured membership worlds. It should also be checked on the accepted tree
example, not generalized to every object labeled `structured`.

### 4.2 Task response-column equivalence

For a fixed declared population, task `q`'s response column is

\[
c_q=(Y(K,q):K\in\mathcal K)
\]

under the canonical state ordering. Tasks are observationally equivalent when their
response columns are identical. A certificate never needs more than one task solely
to distinguish identical columns, although distinct column classes can still have
additional combinatorial redundancy.

The artifact must record the response-column equivalence classes. Column equality is
a diagnostic, not a replacement for exact certificate validation.

## 5. Parameterized Structured Worlds

Add two explicit generators in
`capability_certificate_lab/generators/blocks.py` and export them from
`capability_certificate_lab/generators/__init__.py`:

```python
generate_independent_block_world(blocks: Sequence[Sequence[str]]) -> KnowledgeSpace
generate_prefix_block_world(blocks: Sequence[Sequence[str]]) -> KnowledgeSpace
```

Both generators must reject:

- an empty block list;
- an empty block;
- duplicate task IDs within or across blocks.

They must preserve the declared block and task order, emit deterministic state order,
and use JSON-serializable metadata and generator rules only. Do not store callable
validators in the world.

### 5.1 Rule representation

Represent block synchronization with ordinary prerequisites so that
`KnowledgeSpace.is_valid_state` enforces the declared structure:

- every task requires every other task in its own block;
- in a prefix world, every task in block `j > 0` additionally requires every task in
  all preceding blocks.

An independent-block state may therefore contain any union of complete blocks. A
prefix-block state may contain only the union of the first `j` blocks for some
`j ∈ {0, ..., B}`.

The generated `valid_states` must be the complete population admitted by these rules
for the declared task universe, not a sampled subset.

### 5.2 Independent block world

For `B` non-empty blocks, the state population contains all `2^B` unions of
blocks. For uniform block size `s`:

- task count: `n = B * s`;
- state count: `2^B`;
- response-column equivalence classes: exactly `B`, one per block;
- exact fixed certificate size: exactly `B`;
- fixed task ratio: `1/s`;
- entropy and balanced adaptive average and worst-case depth: exactly `B` under
  the uniform state weighting already used by the adaptive solver.

The fixed-size proof has two parts: one representative per block is sufficient, and
the pair of states differing by one complete block forces every certificate to hit
that block.

### 5.3 Prefix block world

For `B` ordered blocks, the state population contains the empty state and the `B`
successive block prefixes. For uniform block size `s`:

- task count: `n = B * s`;
- state count: `B + 1`;
- response-column equivalence classes: exactly `B`, one per block;
- exact fixed certificate size: exactly `B`;
- fixed task ratio: `1/s`;
- entropy and balanced adaptive worst-case depth: exactly
  `ceil(log2(B + 1))`;
- the adaptive tree contains exactly `B + 1` leaves and `2 * (B + 1) - 1` nodes;
- for `B >= 2`, adaptive average depth must be strictly smaller than the fixed
  certificate size; for `B >= 3`, adaptive worst-case depth must also be strictly
  smaller.

Adjacent prefix states differ on one complete block. This proves the fixed lower
bound of one representative per block. Balanced threshold questions over the
ordered prefixes attain the frozen adaptive worst-case bound.

## 6. Frozen Experiment Grid

Use uniform block names `B{block_index}_T{task_index}` and exactly these
`(block_count, block_size)` cells:

| block count | block size | task count | independent states | prefix states |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 4 | 4 | 3 |
| 3 | 2 | 6 | 8 | 4 |
| 4 | 2 | 8 | 16 | 5 |
| 5 | 2 | 10 | 32 | 6 |
| 6 | 2 | 12 | 64 | 7 |
| 2 | 3 | 6 | 4 | 3 |
| 3 | 3 | 9 | 8 | 4 |
| 4 | 3 | 12 | 16 | 5 |

For every structured cell run:

- identifiability audit;
- exact fixed solver and independent certificate validator;
- entropy adaptive solver and independent adaptive validator;
- balanced adaptive solver and independent adaptive validator;
- the frozen theoretical and structural diagnostics above.

Do not run the random adaptive policy in this milestone.

## 7. Negative and Matched Controls

### 7.1 Canonical no-compression controls

Run the existing generators with deterministic membership responses:

- chain worlds with task counts `4`, `6`, and `8`;
- the accepted four-task tree shape `A -> {B, C}, B -> {D}`;
- full unstructured worlds with task counts `4`, `6`, and `8`.

For each control, gate the independently computed single-coordinate witness list,
exact certificate size, and certificate validity. Each task must be indispensable and
the exact certificate size must equal the task count. If the accepted tree example
does not satisfy the witness oracle, stop and report the discrepancy rather than
weakening the gate.

### 7.2 State-count- and task-count-matched random controls

For each structured family and each grid cell, create 20 deterministic random
declared populations with seeds `0..19`. Match both task count and state count to the
corresponding structured world.

Construct a control by sampling distinct non-constant binary task-response columns
of length `state_count`, transposing them into declared membership states, and
rejecting a draw if any state rows are duplicated. Requirements:

- every task column is non-constant;
- task columns are pairwise distinct;
- state rows are pairwise distinct;
- the identifiability audit passes;
- generation is deterministic for the seed;
- fail loudly after 10,000 complete-population attempts rather than changing the
  seed or constraints.

The exact sampler contract is: initialize one `random.Random(seed)`; for each
complete-population attempt, use `getrandbits(state_count)` to draw task columns,
reject the all-zero column, the all-one column, and columns already drawn in that
attempt, stop when `task_count` columns have been collected, then transpose. If the
rows are not unique, discard the complete population and continue with the same RNG
stream. The artifact records the one-based successful attempt number.

The matched control is an unconstrained declared state population; do not invent
prerequisite metadata for it.

Run only the exact fixed solver and independent validator on matched controls. Record
all 20 runs, their canonical state signatures, generation attempt counts, exact
certificate sizes, selected tasks, and validity. Aggregate exact sizes with
equal-weighted `min`, `median`, `mean`, and `max` over seeds.

These controls are descriptive. A structured cell being smaller, equal, or larger
than its matched controls is a scientific result, not an engineering acceptance
gate. Do not report a p-value, confidence interval, or causal effect of “structure.”

## 8. Claim-to-Oracle Requirements

The formal script must independently compute and gate all of the following without
calling private exact-solver helpers:

1. canonical state signatures;
2. response columns and their equivalence classes;
3. single-coordinate witness pairs and indispensable tasks;
4. theoretical state count for each structured family;
5. expected block coverage of the selected exact certificate;
6. fixed task ratio, savings, and information lower bound;
7. adaptive leaf count, node count, per-state leaf depths, average depth, and
   worst-case depth reconstructed from the serialized tree;
8. agreement between reconstructed tree metrics and solver-reported metrics;
9. exact and adaptive independent validator results.

Runtime may be recorded but is not a scientific outcome. Do not claim scalability
from these bounded runs.

For floating-point comparisons use absolute tolerance `1e-12`. Counts, selected
block coverage, state signatures, leaf state IDs, and worst-case depths require exact
equality.

## 9. Statistical Semantics

- States within an adaptive tree are equally weighted.
- Deterministic structured and canonical-control cells have one run; do not describe
  their validity as a multi-run rate.
- Matched-control seeds are equally weighted and are not independent estimates of a
  real population parameter.
- Do not pool independent-block and prefix-block cells.
- Do not pool cells with different task or state counts.
- Do not turn the eight parameter cells into pseudo-replicates for significance
  testing.
- Report all cells, including hypothesis failures.

## 10. Implementation Scope

Expected implementation paths:

```text
capability_certificate_lab/generators/blocks.py
capability_certificate_lab/generators/__init__.py
scripts/phase7_structural_compressibility.py
tests/test_block_worlds.py
```

Expected evidence paths after an accepted formal run:

```text
artifacts/phase7_structural_compressibility/structural_compressibility.json
artifacts/phase7_structural_compressibility/manifest.json
phase7_report.md
Capability_Certificate_Research_Progress_Summary.md
```

Do not modify the exact, adaptive, identifiability, or validation semantics merely to
make a Phase 7 oracle pass. If a confirmed existing defect blocks the protocol,
return it to `main` with the smallest affected scope before continuing.

## 11. Required Tests

Targeted tests must cover at least:

1. rejection of empty block collections, empty blocks, and duplicate task IDs;
2. deterministic task, block, and state ordering;
3. exact independent-block and prefix-block state populations;
4. every generated state passes `is_valid_state`;
5. representative illegal partial-block and skipped-prefix states fail
   `is_valid_state`;
6. expected response-column equivalence classes;
7. exact fixed-size and block-coverage formulas across the full frozen grid;
8. independent-block adaptive depth formula;
9. prefix-block adaptive leaf, node, average-depth, and worst-case gates;
10. canonical no-compression witness oracles;
11. matched-control reproducibility and constraint enforcement;
12. a deliberately malformed or duplicate-row control is rejected.

Run targeted tests first, then the repository-wide canonical command:

```text
python -m pytest -q tests/test_block_worlds.py
python -m pytest -q
git diff --check
```

Record exact test counts and versions. Do not use historical Phase 2-6 test counts as
Phase 7 verification.

## 12. Formal Experiment and Provenance

The only formal command for this milestone is:

```text
PYTHONPATH=. python scripts/phase7_structural_compressibility.py
```

Formal execution is permitted only after implementation and tests are committed and
the worktree is clean. The operator handoff must bind:

- exact source commit;
- exact script path and command;
- Python version;
- frozen grid and seeds;
- fixed output root;
- pre-run clean status;
- result and manifest paths.

The script may exclude only its fixed output root from its clean-worktree check.
During generation, SHA-256 binds the uncommitted result bytes to the manifest. Final
scientific review must bind the committed evidence SHA; Git is the final identity for
versioned source, configuration, scripts, reports, manifests, and artifacts.

Store enough raw evidence to audit results without rerunning the experiment:

- canonical structured and control state signatures;
- exact solver outputs;
- complete structured adaptive trees;
- reconstructed per-state depths;
- witness pairs and column equivalence classes;
- all matched-control per-seed results;
- aggregate inputs as well as aggregate outputs.

Do not overwrite an accepted or rejected run in place. A rerun must use a new fixed
output root or a new evidence commit with its provenance stated explicitly.

## 13. Hypotheses Versus Acceptance Gates

The following are protocol or implementation gates. A mismatch blocks acceptance and
must be investigated:

- the single-coordinate witness theorem checks;
- independent-block and prefix-block state-count formulas;
- exact fixed certificate size `B` for both block families;
- independent-block adaptive depth `B`;
- prefix-block adaptive worst-case depth `ceil(log2(B + 1))`;
- validator, tree reconstruction, source-binding, and artifact-consistency gates.

The following are empirical scientific outcomes and may be negative without making
the run invalid:

- whether block worlds use fewer fixed tasks than matched random controls;
- the magnitude and direction of matched-control differences;
- how much fixed compression exceeds the information lower bound in random controls.

Do not change the grid, seeds, controls, or claims after observing these outcomes.

## 14. Acceptance Criteria

Phase 7 is complete only when all of the following hold:

1. The mathematical definitions and single-coordinate witness proof are reflected in
   executable, independent diagnostics.
2. Both block generators enforce their declared rules and enumerate their complete
   legal populations deterministically.
3. All structured grid cells satisfy their frozen state-count, fixed-size,
   equivalence-class, and adaptive-tree oracles.
4. Canonical chain, tree, and unstructured controls satisfy the frozen
   no-compression witness oracles.
5. All 320 matched-control populations satisfy their construction constraints and
   retain complete per-seed evidence.
6. Exact and adaptive results pass independent validators; no invalid run is averaged
   away.
7. Statistical denominators and aggregation semantics match Section 9.
8. Targeted and repository-wide tests pass with exact recorded results.
9. The formal experiment runs from a clean committed source with the exact frozen
   command, grid, seeds, and output root.
10. The report distinguishes theorem-backed results, descriptive controls, unsupported
    hypotheses, and accepted limitations.
11. Phase 2-6 accepted evidence remains byte-identical and is not reused as Phase 7
    output.
12. A fresh-context strict read-only final scientific review finds no unresolved
    correctness, protocol, provenance, statistical, or overclaiming finding.

## 15. Allowed Claims and Required Limitations

If all frozen structured oracles pass, the strongest allowed claims are:

- membership spaces with synchronized task blocks have fixed certificates requiring
  one representative per block;
- acyclic prerequisite examples with single-coordinate witness pairs need not provide
  fixed compression;
- ordered prefix-block populations permit lower adaptive query depth than their fixed
  representative sets under uniform state weighting;
- matched controls show where these cells fall among explicitly constrained random
  populations.

Required limitations:

- synchronized blocks are constructed observational redundancy, not learned latent
  capabilities;
- matched random controls do not identify a causal effect of structure;
- the grid is small and exact, not a scalability result;
- uniform state weighting is not a claim about real deployment priors;
- deterministic membership observations do not establish noisy or real-model
  validity.

## 16. Milestone-start Scientific Protocol Review Gate

Before implementation, freeze this handoff in a commit and start a fresh-context
`gpt-5.6-sol` strict read-only reviewer. Provide only:

- this handoff path and exact commit SHA;
- baseline `02380501e5def5d9f624578a93bc0580de1e030a`;
- the accepted Phase 2-6 summary and current relevant APIs;
- explicit instruction not to implement, edit, or run experiments.

The reviewer must audit:

- correctness of the witness and block-family proofs;
- whether generator rules exactly match generated populations;
- whether matched controls are feasible, reproducible, and scientifically fair;
- whether adaptive depth oracles follow from available membership questions;
- denominator, aggregation, and information-bound semantics;
- whether any acceptance gate encodes a desired empirical conclusion;
- artifact sufficiency and provenance;
- scope discipline and scientific overclaiming.

Verdict must be `ACCEPT` or `REJECT`. Resolve every confirmed finding in a new commit
and re-review the repaired handoff before opening implementation.

## 17. Return Format

Return to `main` at each later gate:

```text
Frozen commit or range:
Changed paths:
Protocol deviations:
Tests and exact results:
Formal command and source commit:
Artifacts and provenance:
Structured oracle status:
Matched-control outcomes:
Hypotheses supported or unsupported:
Independent review verdict and findings:
Remaining limitations or blockers:
```
