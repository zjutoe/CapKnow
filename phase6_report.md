# Phase 6 Report: DSL Bridge and Executable Capability World

Status: pending clean committed revalidation. The artifact summary below is historical dirty-run context until regenerated from a clean source commit.

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

Superseded dirty-run artifact: `artifacts/phase2_6_revalidation/phase6_dsl.json`

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
