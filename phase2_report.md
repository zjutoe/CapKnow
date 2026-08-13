# Phase 2 Report: Identifiability Audit

Status: the clean revalidation below is bound to source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`. The combined Phase 2-6 evidence package remains pending independent review.

## Corrected Contract

Phase 2 now treats `KnowledgeSpace.valid_states` as the declared scientific population.

- Empty declared populations raise `ValueError`.
- Invalid declared states raise `ValueError`; they are not silently filtered.
- Duplicate declared `KnowledgeState` entries raise `ValueError`.
- Response signatures must have exactly one binary coordinate per queried task.
- Stable state IDs use canonical JSON arrays, so task IDs containing commas, quotes, braces, or backslashes do not collide.

## Tests

- `python -m pytest -q tests/test_identifiability.py`
- Result: `7 passed in 0.01s`

## Revalidation Summary

Current clean artifact: `artifacts/phase2_6_revalidation/phase2_identifiability.json` (source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`; result `fe0d5c959816fe9b6578eb1c45756f7d57079d914be6a5529e36c5ee5cea3a5c`; manifest `fc31ab28ed66e8f54d808fcae28d94a5c44a8e495785211d219e296c66174bcd`).

| case | tasks | states | unique signatures | identifiable | collisions |
| --- | ---: | ---: | ---: | --- | ---: |
| chain | 4 | 5 | 5 | true | 0 |
| tree | 4 | 7 | 7 | true | 0 |
| unstructured | 3 | 8 | 8 | true | 0 |
| artificial full-vector collision | 2 | 2 | 1 | false | 1 |

Malformed-world cases are covered by pytest and are not counted as scientific experiment conditions.

## Historical Correction

The earlier report said the audit filtered states through `is_valid_state`. That behavior is invalid for this milestone because it can turn malformed declared populations into vacuous success.
