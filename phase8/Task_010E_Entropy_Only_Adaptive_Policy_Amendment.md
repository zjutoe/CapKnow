# Task 010E — Entropy-Only Adaptive-Policy Successor Amendment Proposal

## 1. Status and authority

This is a bounded successor-amendment proposal. Before exact-commit independent
review and explicit main acceptance, it has:

```text
current scientific effect: none
implementation authority: none
execution authority: none
experiment authority: none
artifact mutation authority: none
selection authority: none
Task 010D authority: false
Task 010E execution authority: false
```

```text
Shared protocol path: phase8/README.md
Shared protocol commit: 8b393af06f2cdcdc0197b7c30f5c48f817567604
Inherited sections: Sequence and mutation lease, Common return format
Task-specific overrides: Sequence and mutation lease — this proposal changes only the two documentation paths named here and grants no implementation or execution authority before exact-commit review and main acceptance; Common return format — New artifacts means no experiment or evidence artifacts.
```

Even after acceptance, this proposal only supersedes the exact future
adaptive-policy clauses named below and authorizes main to freeze a separate,
bounded implementation diff. It never changes accepted historical evidence,
authorizes execution, or unblocks Phase 8.

## 2. Rationale and exact-policy proof

Adaptive responses are binary and candidate entries have equal weight. For a fixed
candidate count `N`, let `k = min(no_count, yes_count)`. Binary Shannon entropy is
strictly increasing in `k` on `[0, N/2]`. Therefore, exact integer maximization of
`min(no_count, yes_count)` gives the intended entropy argmax. Evaluating questions
in `task_ids` order and replacing the current best only on a strictly larger score
preserves first-question tie-breaking.

The current floating-point `log2` implementation is not globally identical to
integer balance. Nearby huge counts can round to the same float entropy. A concrete
case is `N=268435456`: the perfect split `134217728/134217728` and the neighboring
split `134217727/134217729` have distinct exact balance scores, but their computed
float entropies can tie. The future canonical policy therefore uses the exact
integer score, not floating entropy arithmetic.

Accepted Phase 4 and Phase 7 finite cells produced identical entropy/balanced
scientific outputs. Historical results consequently require no reinterpretation or
rerun. The motivation is one canonical exact policy and removal of a redundant
experimental dimension; it is not a claim that the old implementations were
identical on every allowed input.

## 3. Exact successor contract

If and only if this proposal's exact commit is independently accepted by main, the
following contract governs a separately frozen future implementation:

1. Future built-in policy names are `random` and `entropy`. `balanced` is
   unsupported and must fail loudly through `resolve_policy` and the solver entry.
2. `select_entropy_reduction_question` uses exact integer maximization of
   `min(no_count, yes_count)`, not floating `log2`. Equal scores retain `task_ids`
   order.
3. `select_balanced_split_question` and `POLICIES["balanced"]` are removed without
   an alias or compatibility shim.
4. Future Phase 4 and Phase 7 script executions emit only the entropy deterministic
   policy dimension, plus already-authorized random results where applicable.
   Existing tracked artifacts and reports stay byte-identical historical evidence
   and must not be overwritten.
5. Only these future clauses are superseded:
   - master Section 10.2's line requiring entropy and balanced solvers becomes one
     canonical entropy solver, validator, and tree reconstruction;
   - master Section 10.3's entropy-and-balanced depth metrics become canonical
     entropy depths only; and
   - the entropy-and-balanced bullet in
     `Task_010E_Behavioral_Certificate_Integration.md` becomes canonical entropy
     only.
6. All other master and Task 010E clauses, thresholds, matrices, validators,
   metrics, paths, gates, and blocked status remain unchanged.
7. Historical Phase 4 and Phase 7 handoffs, reports, artifacts, hashes, and policy
   labels remain archive-in-place and valid as historical evidence.
8. This change does not authorize Task 010D, Task 010E execution, selection,
   training, artifacts, or any experiment.

Acceptance of this proposal makes this narrow successor contract current for the
separate source implementation and any later Task 010E execution handoff. It does
not itself implement code, alter accepted evidence, authorize Task 010E execution,
or change any scientific result or Phase 8 gate.

## 4. Separately authorized implementation boundary

After proposal acceptance, main may freeze a separate implementation handoff
limited to:

```text
capability_certificate_lab/certificate/policies.py
scripts/revalidation_phase4_adaptive.py
scripts/phase7_structural_compressibility.py
tests/test_adaptive_certificate.py
tests/test_block_worlds.py
```

That implementation must test:

- existing entropy behavior and adaptive validation;
- explicit `balanced` rejection;
- `task_ids`-order tie-breaking;
- a synthetic large-count regression proving that exact integer entropy order
  selects the perfect split where float entropy would tie;
- entropy-only script result keys and loops without writing artifacts;
- targeted and repository-wide pytest, static, compile, and import checks; and
- byte identity of historical artifacts.

No historical artifact rerun is required or allowed. The implementation commit
requires a fresh-context `gpt-5.6-sol` high scientific/API review.

## 5. Preservation and non-authority

This proposal leaves all accepted historical Phase 4 and Phase 7 evidence exactly
as accepted. Their entropy and balanced names, outputs, reports, artifacts, and
hashes remain historical records rather than inputs to migration or rewriting. No
new artifact, scientific result, selection decision, training authority, execution
authority, or Phase 8 gate follows from this document.

The amendment is deliberately narrow: it changes only the three named future
clauses after acceptance. It does not revise adaptive thresholds, task matrices,
validator semantics other than the named policy consolidation, metrics other than
the named deterministic-policy dimension, or any blocked-status condition.

## 6. Stop conditions

Stop without further edits if the proposal would need to alter frozen documents,
artifacts, code, scripts, tests, or reports; claim that the old implementations were
globally equivalent; broaden beyond the three named future clauses; or grant
execution or scientific-result authority.

Any later implementation also stops on scope growth, an attempted compatibility
alias, acceptance of `balanced`, floating-point entropy scoring, changed historical
evidence, artifact output, or any attempt to treat proposal acceptance as Task 010D,
Task 010E execution, selection, training, or experiment authority.

## 7. Review and next gate

This docs-only proposal requires diff, scope, required-text, README-row, and
protected-path byte-identity checks. It requires no pytest, artifact validator,
experiment, GPU command, rerun, or artifact mutation.

Main must first commit the exact proposal and index-row diff. A fresh-context
`gpt-5.6-sol` high protocol/scientific reviewer must then review that exact commit.
Only explicit main acceptance of the reviewed commit may permit main to freeze the
separate implementation handoff described above. No implementation or execution is
authorized by this proposal itself.

## 8. Common return format

```text
Scope completed: entropy-only adaptive-policy successor amendment proposal and README index row
Prerequisite commit used: 8b393af06f2cdcdc0197b7c30f5c48f817567604
Changed paths: phase8/Task_010E_Entropy_Only_Adaptive_Policy_Amendment.md; phase8/README.md
Scientific contract changes: proposed future entropy-only adaptive-policy contract; no current scientific effect
Engineering-only changes: proposal document and one README index row
Tests run and exact results: docs-only diff, scope, text, protected-path identity, and exact README-row checks
New artifacts: none
Unresolved decisions: none for this bounded proposal
Protocol deviations: none
Resource observations: none; no pytest, artifact validator, experiment, or GPU command run
Independent review required: yes — exact-commit fresh-context gpt-5.6-sol high protocol/scientific review
Recommended next authorized step: main commit plus independent review only
```
