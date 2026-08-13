# Phase 6 Report: DSL Bridge and Executable Capability World

Status: formal Phase 6 revalidation was rerun from clean source commit `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`. Current evidence gate: pending independent review.

## Corrected Contract

- `execute(program, state_has, input_context)` returns an explicit `ExecutionResult(value, output_type)`.
- Missing primitive capabilities raise `MissingCapabilityError`.
- Invalid programs and invalid input types raise `InvalidProgramError`.
- `make_dsl_response_signature` is deterministic and no longer accepts noisy RNG-backed callbacks.
- Signature generation maps missing capabilities to response `0` but propagates malformed programs.
- Composition result IDs must be unique and must not collide with primitive IDs.
- Capability values must be actual booleans; truthy malformed values are contract errors.
- `LoopNode.max_iterations` must be a non-boolean positive integer.
- `COMPARE` requires a comparison pair and rejects bare boolean input.
- Conditional programs require every primitive declared by the complete program, including inactive branches.
- `ADD` rejects booleans as numeric inputs. `CONDITION` accepts only booleans, mappings with a boolean `condition`, or non-string sequences with explicit non-empty semantics.

## Minimal Primitive Semantics

| primitive | input type | output type | deterministic behavior |
| --- | --- | --- | --- |
| ADD | numeric pair | number | returns sum |
| COMPARE | comparison pair | bool | returns equality |
| MEMORY | memory lookup | memory context | reads key from memory and writes `value` |
| SEARCH | search context | bool | checks whether target/value is in items |
| FILTER | filter context | list | keeps items equal to target/value |
| LOOP | loop value | loop value | returns input unchanged as a primitive |
| CONDITION | condition value | bool | returns boolean condition |

`SequenceNode` passes each step's output value to the next step. `ConditionNode` evaluates a boolean condition and executes one branch on the original input. `LoopNode` requires `LOOP` and feeds each iteration's output into the next iteration.

## Tests

- `python -m pytest -q tests/test_dsl.py`
- Result: `18 passed in 0.06s`

## Revalidation Summary

Latest formal artifact: `artifacts/phase2_6_revalidation/phase6_dsl.json`

- source commit: `66c0efade42d85c2ca9c5eca1d3cdb4fc19e3d40`
- result sha256: `252d8043ea73a087b77be919bce91bc91e0b0cfaa40ecaca5887b69006abc5e0`
- manifest sha256: `814b3cc1320090677fe6d6feb73b5fe3804cb001ff96c445e238a1b4b14b487d`
- pre-run source tree: clean
- review state: pending independent evidence review

- Observed primitive execution exact outputs: `ADD=5`, `COMPARE=true`, `MEMORY=42`, `SEARCH=true`, `FILTER=["keep","keep"]`, `LOOP="unchanged"`, `CONDITION=false`.
- Acceptance oracle for primitive execution freezes those seven exact outputs.
- Observed composite `SEQ[MEMORY,SEARCH]` returns `true` from `MEMORY` and `SEARCH` alone, without a `RETRIEVAL` state label.
- Acceptance oracle for the default composite freezes the program id, required primitives, and final output.
- Observed held-out composition uses `FILTER + CONDITION -> FILTER_CHECK`, which is absent from default rules, and executes from primitive capabilities only.
- Acceptance oracle for the held-out composition freezes the input context and final output.
- The script computes `present_in_default_rules` from `DEFAULT_COMPOSITION_RULES`; the artifact records the default rules as `MEMORY+SEARCH->RETRIEVAL` and `RETRIEVAL+CONDITION->PLANNING`.
- Certificate transfer on the pure primitive DSL world matches the direct-task world: both have `state_count=128`, `certificate_size=7`, and `valid=true`; these are gated as explicit invariants.
- Reproducibility check: repeated DSL signatures are identical and match the frozen signature `[0,0,1,1,0,0,0,1,0]`.

## Historical Correction

The previous Phase 6 scaffold returned only Boolean capability membership from `execute` and ignored concrete inputs. That scaffold is superseded by the explicit deterministic execution contract above.
