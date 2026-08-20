# Phase 8 GPT-5.5 Execution Handoffs

This directory contains the bounded implementation handoffs for
`Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md`.
The master handoff is the scientific source of truth; these files only decompose its
implementation and evidence-preparation work.

## Fixed execution route

Every task `010A` through `010H` must be launched as:

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
context: fresh
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

The required model identifier is exactly `gpt-5.5`; do not infer or append a role or
model suffix. `main` must give the executor the exact accepted prerequisite commit
and the relevant handoff paths in its launch packet. Branch names and an uncommitted
working tree are not valid bindings.

## Sequence and mutation lease

```text
010A -> 010B -> 010C -> 010D -> 010E -> 010F -> 010G -> 010H
     -> final-source feasibility selection -> resource benchmark/authorization
     -> formal shards

010C feasibility FAIL -> 010C-D1 non-selection/fixed-factor diagnostic -> main decision
    -> 010C-D2 independently reviewed one-shot protocol-amendment proposal
    -> 010C-D2-I weight-tying implementation -> independent implementation review
```

The post-010C feasibility decision is an early engineering gate. Because later tasks
complete the shared formal paths, the run selected by formal `prepare` must be rerun
from the exact final accepted implementation commit before the resource benchmark.

Only one executor may hold the writer lease. At each boundary:

1. the executor returns without committing unless `main` explicitly authorized a
   commit;
2. `main` inspects the diff, runs or confirms verification, and freezes it in a
   commit;
3. high-risk protocol or implementation changes receive a fresh-context independent
   read-only review;
4. `main` accepts or repairs that exact commit before launching the next task.

An executor must not implement a later task, change a scientific constant, loosen a
gate, run the formal experiment, or authorize resources. Ambiguity is a stop
condition and must be returned to `main`.

## Handoff index

- `010A`: DSL oracle, state/task order, split oracle, and immutable probe pack.
- `010B`: A/B/C corpora, held-out exclusion, and randomized-control validation.
- `010C`: byte tokenizer, causal model, loss primitives, software smoke control, and
  held-out feasibility runner.
- `010C-D1`: non-evidence failure diagnostic for an accepted immutable feasibility
  failure; it cannot create a selection or advance to `010D`.
- `010C-D2`: source-controlled one-shot protocol-amendment proposal based only on the
  accepted D1 diagnostic. The proposal has no implementation or experiment authority;
  acceptance at an exact reviewed commit is required before a separate implementation
  handoff may be frozen. The accepted proposal commit is
  `9a767c6708c7c69f5ba98848250afcf50c8c5a6f`.
- `010C-D2-I`: implementation of the independently accepted D2 weight-tying proposal,
  including exact historical validation, D1 decision-provenance binding, byte-exact
  tied-checkpoint validation, the exact `model_protocol_revision` configuration key,
  and a real-`__main__` one-shot `feasibility_005` gate. The implementation must be
  committed and independently reviewed before `feasibility_005` can be launched; it
  cannot run the experiment or authorize `010D`.
- `010D`: formal training/checkpoint primitives and isolated evaluator.
- `010E`: behavioral response matrices and exact/adaptive certificate integration.
- `010F`: frozen metrics, sensitivity matrices, aggregation, and report-ready
  interpretation records.
- `010G`: eighteen-shard formal runner, status publication, retries, aggregation, and
  provenance.
- `010H`: non-scientific resource benchmark and evidence for `main`'s formal-run
  authorization.

## Common return format

```text
Scope completed:
Prerequisite commit used:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests run and exact results:
New artifacts:
Unresolved decisions:
Protocol deviations:
Resource observations:
Independent review required:
Recommended next authorized step:
```

`Protocol deviations` must be `none` for acceptance. A blocked task returns the
smallest concrete ambiguity or failing contract and leaves later work untouched in
this form:

```text
Problem:
Minimal reproduction:
Evidence:
Likely cause:
Scientific or engineering classification:
Decision required from main:
Work explicitly not attempted:
```
