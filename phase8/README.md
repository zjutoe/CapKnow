# Phase 8 GPT-5.5 Execution Handoffs

This directory contains the bounded implementation handoffs for
`Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md`.
The master handoff is the scientific source of truth; these files only decompose its
implementation and evidence-preparation work.

## Current Phase 8 state

This status snapshot is bound as of repository commit
`0cabe0c7a457939eb440063a2dd207ae84b005f1`. The sole master scientific authority is
`Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md`; this
index summarizes status and does not replace or extend that contract.

- `feasibility_005` is bound to source commit
  `2dc8c50fab9ece27a07eb0dc04021b78de52895c`, terminated `FAILED`, and passed
  `11/24` cells. It produced no selection. The current protocol provides no
  `feasibility_006` fallback and no `010D` execution authority.
- The accepted D3 root is
  `artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001`. Its operational
  terminal is `DONE` and its artifact class is
  `non_evidence_feasibility_postmortem`.
- D3 is bound to implementation commit
  `e3f99fcabbce812408793b62a1cdad536b8b303b` and to the accepted proposal at commit
  `06ee71d958eb1c4cb446ede3d120e99eb64bba97`, blob
  `5ea32f67d078408a6764adbb2d66518f713245d7`.

| Run | `manifest.json` SHA-256 | `summary.json` SHA-256 | Terminal marker SHA-256 |
| --- | --- | --- | --- |
| `feasibility_005` | `51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b` | `bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785` | `FAILED.json`: `6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08` |
| `feasibility_postmortem_001` | `1cf55f231cc064ba1e059c084dac38b26888d59907d5469c803cb061dd61d503` | `d8e9d78aa32147d697accdfb92068751724f0de87a0d969ce69f89278201a1cd` | `DONE.json`: `f566a21e2ac2a1dd22dad90c77aaef2ee2331dd8532d824e0083e6f1382caec4` |

D3 `DONE` means only that the read-only postmortem job completed. Its interpretation
limits are `non_evidence`, `no_verdict_change`, `no_causal_weight_tying_claim`, and
`no_010d_authority`: it does not change the `feasibility_005` `FAILED` verdict and
does not grant selection or `010D` authority. Later source or test cleanup commits
do not retroactively rebind, rewrite, replace, or reinterpret the `feasibility_005`
or D3 artifacts.

## Lifecycle vocabulary and archive-in-place

- **Current authority** identifies the exact master or accepted successor that owns
  a scientific or execution contract. Here that is the master handoff named above,
  not this index.
- **Current status** is the bounded snapshot in this index, tied to its explicit
  commit and master/successor bindings.
- **Conditional future / blocked** identifies work whose handoff exists but whose
  prerequisites or separate authorization are absent.
- **Historical provenance** identifies a frozen point-in-time input, decision,
  handoff, or review record. Historical does not mean invalid.
- **Superseded/rejected as current authority** means a record is not the operative
  contract; it does not erase the record or its provenance value.

The archive rule is **archive-in-place**: frozen historical documents remain at
their exact paths and bytes because paths and blobs may be provenance-bound.
Pending, `REJECT`, or future language in their bodies records their frozen
point-in-time state; current status comes from this index together with its exact
master/successor binding. This index does not grant execution, experiment,
selection, commit, or resource authority.

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
    -> feasibility_005 FAIL -> 010C-D3 read-only postmortem proposal -> review
    -> 010C-D3-I read-only postmortem implementation -> independent implementation review
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

## Future-only shared handoff protocol

Only a future handoff may inherit common sections from this README, and only by
explicitly naming both `phase8/README.md` and the exact Git commit whose README
bytes it uses. Its opt-in declaration must enumerate every inherited section by
name:

```text
Shared protocol path: phase8/README.md
Shared protocol commit: <exact-40-hex-Git-commit>
Inherited sections: <exact section name>[, <exact section name> ...]
Task-specific overrides: <exact inherited field/section and replacement, or none>
```

Permitted shared material is limited to generic process text in `Fixed execution
route`, the writer lease/main-owned commit and review gate in `Sequence and mutation
lease`, `Common return format`, and its blocked return format. The shared protocol
and this index grant no scientific, execution, experiment, selection, commit, or
resource authority.

This rule is future-only and has no retroactive effect. Existing A-H and D1-D3
handoffs keep their exact frozen bytes and do not implicitly inherit it. Never
inherit or deduplicate task-specific scientific constants, schemas, metrics, stop
or selection gates, artifact/source/path/hash bindings, allowed paths, tests,
prerequisites, resource or experiment authorization, or accepted/rejected lineage.

An override must name the exact inherited fields or sections and affect only those
fields. It cannot broaden execution, scientific, or resource authority unless the
task's actual accepted authority explicitly grants that broader authority.

A missing or mismatched shared-protocol path, exact commit, or named section fails
closed and must be returned to `main`. A branch name or moving `HEAD` is not a valid
binding.

## Handoff index

The role labels below are navigational summaries, not replacement task contracts.

| Path | Document role | Lifecycle/current authority | Evidence role | Execution authority |
| --- | --- | --- | --- | --- |
| `README.md` | Navigation and status index | Current status only; never scientific authority | None | None |
| `Task_010A_DSL_Oracle_and_Probe_Pack.md` | DSL oracle and immutable probe-pack handoff | Historical provenance for a completed prerequisite | Evidence-preparation contract, not evidence | No current authority |
| `Task_010B_Corpus_and_Control.md` | Corpus, exclusion, and randomized-control handoff | Historical provenance for a completed prerequisite | Evidence-preparation contract, not evidence | No current authority |
| `Task_010C_Tokenizer_Model_and_Feasibility.md` | Tokenizer, model, loss, and original feasibility handoff | Historical provenance; D2 replaced only its pre-D2 model/checkpoint revision and associated one-shot 005 protocol, as incorporated into master Sections 8.2/9.4 | Historical implementation/evidence-preparation contract, not result evidence | No current authority |
| `Task_010C_D1_Feasibility_Failure_Diagnostic.md` | Fixed-factor failure-diagnostic handoff | Historical provenance | Non-evidence diagnostic only; cannot select | None |
| `Task_010C_D2_Feasibility_Protocol_Amendment_Proposal.md` | One-shot weight-tying amendment proposal | Historical provenance; accepted proposal commit `9a767c6708c7c69f5ba98848250afcf50c8c5a6f` | Proposal, not evidence | No direct implementation or experiment authority |
| `Task_010C_D2_Weight_Tying_Implementation.md` | Accepted-amendment implementation handoff | Historical provenance after `feasibility_005` | Implementation provenance, not result evidence | No current authority; cannot authorize `010D` |
| `Task_010C_D3_Feasibility_005_Postmortem_Proposal.md` | Checkpoint-read-only postmortem proposal | Historical provenance after the accepted D3 run | Proposal, not evidence | No direct implementation or experiment authority |
| `Task_010C_D3_Postmortem_Implementation.md` | Fail-closed read-only postmortem implementation handoff | Historical provenance after D3 `DONE` | Defines a non-evidence analysis only | No current authority; cannot authorize `010D` |
| `Task_010C_D3_Postmortem_Verifier.py.txt` | Frozen inline D3 launch verifier / pre-runner trust root | Historical provenance; not scientific authority | Executable verification provenance, not scientific evidence | None by itself |
| `Future_Only_Postmortem_Process_Isolation_Design.md` | Future-only postmortem process-isolation simplification design | Conditional future / design only; never current scientific authority | None | None |
| `Task_010D_Training_Checkpoint_and_Evaluator.md` | Formal training, checkpoint, and evaluator handoff | Conditional future / blocked by no selection | No current evidence | No current authority |
| `Task_010E_Behavioral_Certificate_Integration.md` | Behavioral-certificate integration handoff | Conditional future / blocked behind `010D` | No current evidence | No current authority |
| `Task_010F_Metrics_and_Aggregation.md` | Metrics and aggregation handoff | Conditional future / blocked behind prior tasks | No current evidence | No current authority |
| `Task_010G_Sharded_Runner_and_Provenance.md` | Formal sharded-runner handoff | Conditional future / blocked behind prior tasks | No current evidence | No current authority |
| `Task_010H_Resource_Benchmark_and_Formal_Authorization.md` | Resource benchmark and authorization-evidence handoff | Conditional future / blocked behind prior tasks | No current evidence | No current authority |

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
