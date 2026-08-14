# Phase 7 Report: Structural Compressibility

Status: formal Phase 7 artifacts were generated from clean source commit
`b44cad06dbe1859008c9ce47f6434ef1ac7774a4`. Final independent scientific
review is pending.

## Source Binding

- Baseline reference commit: `02380501e5def5d9f624578a93bc0580de1e030a`
- Accepted Phase 7 protocol commit: `a727c5e9dd8619e0d8135b8530af691d5a9e6694`
- Implementation source commit: `b44cad06dbe1859008c9ce47f6434ef1ac7774a4`
- Formal command: `PYTHONPATH=. python scripts/phase7_structural_compressibility.py`
- Pre-run worktree status: clean
- Fixed output root: `artifacts/phase7_structural_compressibility`
- Result artifact: `artifacts/phase7_structural_compressibility/structural_compressibility.json`
- Result sha256: `9ffd95abfe4ad85ee55aa833342de175d8bebdb39bf9ac808259943282842270`
- Manifest artifact: `artifacts/phase7_structural_compressibility/manifest.json`
- Manifest sha256: `a496f502f0230089d220a90c86873b0ae1ed258c8a267fa887f960b7c9e6caa0`

The manifest records the exact source commit, command, Python version, frozen grid,
seeds, output root, and complete output-file inventory. The result bytes are bound
inside the manifest by SHA-256; the manifest hash above is bound by Git once this
report and artifact package are committed.

## Implementation Checks

Phase 7 adds two synchronized block generators:

- independent block worlds, where any union of complete blocks is legal;
- prefix block worlds, where only the empty state and successive ordered block
  prefixes are legal.

Both generators reject empty block collections, empty blocks, and duplicate task
IDs. They preserve declared block/task order, store only JSON-serializable metadata
and prerequisite rules, and do not store callable validators.

The formal script independently computes canonical state signatures, response
columns and equivalence classes, single-coordinate witness pairs, exhaustive
structured-population closure against `KnowledgeSpace.is_valid_state`, fixed
certificate validity, block coverage, adaptive tree reconstruction metrics, and
matched-control construction constraints.

Implementation verification before the formal run:

| command | result |
| --- | --- |
| `python -m pytest -q tests/test_block_worlds.py` | `11 passed in 0.18s` |
| `python -m pytest -q` | `85 passed in 0.32s` |
| `git diff --check` | `passed` |
| non-writing `run_phase7_analysis()` smoke check | `16` structured cells and `320` matched controls passed |
| dirty-source formal command check | refused execution before creating outputs |

Python version was `3.13.9`; pytest version was `8.4.2`.

Independent strict read-only implementation review accepted source commit
`b44cad06dbe1859008c9ce47f6434ef1ac7774a4` with no blocking findings. The
reviewer did not run the formal experiment or create outputs.

## Formal Grid

The structured grid contains eight `(block_count, block_size)` cells. Each cell is
run for both the independent-block and prefix-block family, giving 16 structured
cells total. Each structured family/cell has 20 matched random controls with seeds
`0..19`, giving 320 matched-control populations total.

| family | B | s | tasks | states | fixed size | fixed ratio | info lower bound | adaptive avg | adaptive worst | matched fixed mean | matched fixed range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independent | 2 | 2 | 4 | 4 | 2 | 0.5000 | 2 | 2.0000 | 2 | 2.30 | 2-3 |
| prefix | 2 | 2 | 4 | 3 | 2 | 0.5000 | 2 | 1.6667 | 2 | 2.00 | 2-2 |
| independent | 3 | 2 | 6 | 8 | 3 | 0.5000 | 3 | 3.0000 | 3 | 4.25 | 4-5 |
| prefix | 3 | 2 | 6 | 4 | 3 | 0.5000 | 2 | 2.0000 | 2 | 2.10 | 2-3 |
| independent | 4 | 2 | 8 | 16 | 4 | 0.5000 | 4 | 4.0000 | 4 | 5.60 | 5-6 |
| prefix | 4 | 2 | 8 | 5 | 4 | 0.5000 | 3 | 2.4000 | 3 | 3.00 | 3-3 |
| independent | 5 | 2 | 10 | 32 | 5 | 0.5000 | 5 | 5.0000 | 5 | 7.20 | 7-8 |
| prefix | 5 | 2 | 10 | 6 | 5 | 0.5000 | 3 | 2.6667 | 3 | 3.05 | 3-4 |
| independent | 6 | 2 | 12 | 64 | 6 | 0.5000 | 6 | 6.0000 | 6 | 9.05 | 8-10 |
| prefix | 6 | 2 | 12 | 7 | 6 | 0.5000 | 3 | 2.8571 | 3 | 3.20 | 3-4 |
| independent | 2 | 3 | 6 | 4 | 2 | 0.3333 | 2 | 2.0000 | 2 | 2.10 | 2-3 |
| prefix | 2 | 3 | 6 | 3 | 2 | 0.3333 | 2 | 1.6667 | 2 | 2.00 | 2-2 |
| independent | 3 | 3 | 9 | 8 | 3 | 0.3333 | 3 | 3.0000 | 3 | 3.95 | 3-4 |
| prefix | 3 | 3 | 9 | 4 | 3 | 0.3333 | 2 | 2.0000 | 2 | 2.00 | 2-2 |
| independent | 4 | 3 | 12 | 16 | 4 | 0.3333 | 4 | 4.0000 | 4 | 5.05 | 5-6 |
| prefix | 4 | 3 | 12 | 5 | 4 | 0.3333 | 3 | 2.4000 | 3 | 3.00 | 3-3 |

For every structured cell, both entropy and balanced adaptive policies produced the
same average and worst-case depths shown above. Every structured cell passed its
state-count, exact fixed-size, response-column equivalence, block-coverage,
exhaustive closure, fixed-validator, adaptive-validator, and adaptive metric
reconstruction gates.

## Canonical No-compression Controls

Each canonical control satisfies the single-coordinate witness oracle for every
task. Therefore every task is individually indispensable and the exact fixed
certificate size equals the task count.

| control | tasks | states | exact fixed size | all tasks indispensable |
| --- | ---: | ---: | ---: | --- |
| chain_n4 | 4 | 5 | 4 | true |
| chain_n6 | 6 | 7 | 6 | true |
| chain_n8 | 8 | 9 | 8 | true |
| accepted_tree_A_BC_BD | 4 | 7 | 4 | true |
| unstructured_n4 | 4 | 16 | 4 | true |
| unstructured_n6 | 6 | 64 | 6 | true |
| unstructured_n8 | 8 | 256 | 8 | true |

This explains why the accepted chain, tree, and full unstructured worlds did not
show fixed task compression despite having different prerequisite structures.

## Matched Controls

Matched controls are descriptive, not acceptance gates. They match each structured
family/cell on task count and state count, use deterministic seeds `0..19`, and are
constructed from distinct non-constant task-response columns with pairwise distinct
state rows. No prerequisite metadata is assigned to matched controls.

Independent-block fixed sizes were below the matched-control mean in every frozen
cell. Prefix-block fixed sizes were equal to the matched-control mean for `B=2` and
above the matched-control mean for the frozen `B>=3` cells. This is expected from
the protocol boundary: prefix worlds have fixed compression relative to task count,
but their ordered state population does not imply smaller fixed certificates than
unconstrained populations with the same task/state counts.

No p-values, confidence intervals, or causal claims about structure are reported.

## Supported Claims

The formal artifact supports these theorem-backed claims within the frozen grid:

- the two complete declared block families have fixed certificates requiring one
  representative per block;
- the independent-block fixed certificate size equals the information lower bound
  for the uniform cells in this grid;
- the prefix-block adaptive worst-case depth equals `ceil(log2(B + 1))`;
- prefix-block adaptive average depth is strictly below fixed size for `B >= 2`;
- prefix-block adaptive worst-case depth equals fixed size for `B = 2` and is
  strictly below fixed size for `B >= 3`;
- the accepted chain, tree, and full unstructured controls require all tasks because
  every task has a single-coordinate witness pair.

These claims are limited to deterministic membership observations, uniform state
weighting, and the exact small declared populations in the artifact.

## Limitations

- Synchronized blocks are constructed observational redundancy, not learned latent
  capabilities.
- Matched random controls do not identify a causal effect of structure.
- The grid is small and exact, not a scalability result.
- Uniform state weighting is not a claim about real deployment priors.
- Deterministic membership observations do not establish noisy or real-model
  validity.
