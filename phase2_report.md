# Phase 2 Report: Identifiability Audit

Status: pending clean committed revalidation. The artifact summary below is historical dirty-run context until regenerated from a clean source commit.

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

Superseded dirty-run artifact: `artifacts/phase2_6_revalidation/phase2_identifiability.json`

| case | tasks | states | unique signatures | identifiable | collisions |
| --- | ---: | ---: | ---: | --- | ---: |
| chain | 4 | 5 | 5 | true | 0 |
| tree | 4 | 7 | 7 | true | 0 |
| unstructured | 3 | 8 | 8 | true | 0 |
| artificial full-vector collision | 2 | 2 | 1 | false | 1 |

Malformed-world cases are covered by pytest and are not counted as scientific experiment conditions.

## Historical Correction

The earlier report said the audit filtered states through `is_valid_state`. That behavior is invalid for this milestone because it can turn malformed declared populations into vacuous success.
