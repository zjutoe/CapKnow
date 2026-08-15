# Phase 8 Spark Execution Handoffs

This directory contains the bounded implementation handoffs for
`Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md`.
The master handoff is the scientific source of truth; these files only decompose its
implementation and evidence-preparation work.

## Fixed execution route

Every task `010A` through `010H` must be launched as:

```text
role: Spark executor
model override: gpt-5.5
context: fresh
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

The available model identifier is `gpt-5.5`; `Spark` is the execution role. Do not
substitute an invented `gpt-5.5-spark` identifier. `main` must give the executor the
exact accepted prerequisite commit and the relevant handoff paths in its launch
packet. Branch names and an uncommitted working tree are not valid bindings.

## Sequence and mutation lease

```text
010A -> 010B -> 010C -> 010D -> 010E -> 010F -> 010G -> 010H
```

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
