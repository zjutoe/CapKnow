# Phase 4 Report: Adaptive Certificate Solver

Status: the clean revalidation below is bound to source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`. The combined Phase 2-6 evidence package remains pending independent review.

## Corrected Contract

- Built-in adaptive policies only choose unasked tasks that split the current candidate set into two non-empty branches.
- Custom policies that return unknown, repeated, or non-splitting tasks fail loudly.
- Custom policies may return `None` only when no unasked splitting task remains.
- Empty declared state populations fail loudly instead of producing vacuous adaptive success.
- `validate_adaptive_certificate` recursively checks both branches, non-empty progress, no repeated questions, and exactly one remaining state per leaf.
- `AdaptiveCertificate.to_dict()` includes the full root decision tree.
- `average_depth` is equal-weighted over states within each tree. Cross-seed summaries are equal-weighted over runs. `worst_case_depth` is reported separately.

## Tests

- `python -m pytest -q tests/test_certificate.py tests/test_adaptive_certificate.py`
- Result: `19 passed in 0.04s`

## Revalidation Summary

Current clean artifact: `artifacts/phase2_6_revalidation/phase4_adaptive_regression.json` (source commit `1bfc6ced88cf3397f240c3e79d1996955e9d589f`; result `1d00e83570164b7253c724cb68e4c5760c283d459a4097621bd88079d96dd39d`; manifest `738fd13570b3ec4b3662ff4b84bf3e3076cebf49b7806af91c7076484237a9d3`).

| world | fixed exact size | policy | valid run rate | mean average depth | max worst-case depth | mean nodes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| chain | 4 | balanced | 1.0 | 2.4000 | 3 | 9.0 |
| chain | 4 | entropy | 1.0 | 2.4000 | 3 | 9.0 |
| chain | 4 | random seeds 0..99 | 1.0 | 2.5580 | 4 | 9.0 |
| tree | 4 | balanced | 1.0 | 2.8571 | 3 | 13.0 |
| tree | 4 | entropy | 1.0 | 2.8571 | 3 | 13.0 |
| tree | 4 | random seeds 0..99 | 1.0 | 3.0357 | 4 | 13.0 |
| unstructured | 3 | balanced | 1.0 | 3.0000 | 3 | 15.0 |
| unstructured | 3 | entropy | 1.0 | 3.0000 | 3 | 15.0 |
| unstructured | 3 | random seeds 0..99 | 1.0 | 3.0000 | 3 | 15.0 |

## Historical Correction

Earlier aggregate wording could confuse worst-case `query_count` with average depth. The corrected report separates `average_depth` and `worst_case_depth`.
