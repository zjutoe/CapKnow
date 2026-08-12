# Phase 5 Report: Probabilistic Certificate

Status: pending clean committed revalidation. The artifact summary below is historical dirty-run context until regenerated from a clean source commit.

## Corrected Contract

- `infer_state_posterior` computes likelihoods in log space and normalizes with log-sum-exp.
- Mathematically impossible observations remain explicit: all posterior masses are zero and `map_state=""`.
- Adaptive Bayesian updating uses the previous posterior as prior and only the newly observed responses as likelihood evidence.
- Complete query history is retained for reporting but is not counted twice.
- MAP selection uses a shared deterministic tie rule with tolerance `1e-12`.
- Phase 5 consistency gates compare batch, sequential, and solver-observed posterior MAP states, confidence, and near-tie state sets.
- Adaptive simulation requires `true_state` to belong to the declared state population.
- Attempt counts and query limits reject booleans instead of treating them as integers.

## Tests

- `python -m pytest -q tests/test_probabilistic.py`
- Result: `23 passed in 0.05s`

Exact regression coverage includes:

- zero-probability Bernoulli sampling returns `0` when the RNG boundary value is exactly `0.0`;
- sequential updates equal batch updates;
- Phase 5 consistency rejects near-tie MAP-set drift between batch and sequential posterior paths;
- exact two-task posterior values for `A=1, B=0`;
- long possible observation sequences do not underflow;
- impossible zero-probability observations are not smoothed;
- adaptive stopping does not cross threshold solely because old evidence was counted again;
- undeclared adaptive true states and boolean attempt controls are rejected.

## Revalidation Summary

Superseded dirty-run artifact: `artifacts/phase2_6_revalidation/phase5_robustness.json`

Conditions:

- worlds: chain, tree, unstructured;
- noise pairs: `(0,0)`, `(0.05,0.05)`, `(0.10,0.10)`, `(0.20,0.20)`, `(0.30,0.30)`, `(0.30,0.05)`, `(0.05,0.30)`;
- seeds: `0..99`;
- attempts per queried task: `1, 3, 5`;
- methods: fixed certificate and entropy-reduction adaptive certificate.

Sanity checks:

- zero-noise fixed/adaptive MAP results match deterministic behavior for all three worlds;
- zero-noise adaptive query counts match deterministic entropy-tree leaf depths for every state;
- zero-noise average/worst adaptive query counts equal deterministic depths: chain `2/2`, tree `2.4/3`, unstructured `2/2`, for attempts `1`, `3`, and `5`;
- fixed sequential/batch posterior consistency failures: `0`;
- adaptive sequential/batch posterior consistency failures: `0`;
- fixed and adaptive consistency rates are both `1.0` for all sampled conditions;
- maximum fixed/adaptive posterior-map difference across summaries is `1.5543122344752192e-15`, below the `1e-12` gate tolerance;
- symmetric-noise aggregates are reported descriptively, and sampled non-monotonicity is not interpreted as improved evidence.

Selected high-noise examples from the artifact:

| world | method | slip | guess | attempts | MAP accuracy | mean confidence | threshold rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain | fixed | 0.30 | 0.30 | 5 | 0.74 | 0.7025 | 0.08 |
| tree | fixed | 0.30 | 0.30 | 5 | 0.67 | 0.6735 | 0.03 |
| unstructured | fixed | 0.30 | 0.30 | 5 | 0.74 | 0.6815 | 0.01 |

## Historical Correction

Superseded Phase 5 adaptive robustness results generated before this repair are not scientific evidence. The prior implementation could double-count old observations and the earlier consistency report failed to gate MAP disagreements.
