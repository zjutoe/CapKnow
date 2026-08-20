# Task 010C-D3-I — Read-Only Postmortem Implementation

## Status and authority

```text
status: implementation handoff pending independent review
accepted proposal commit: 336afdc7cb079086998a0db527f544f16b68950c
accepted proposal blob: 1c52056e6579262045d81315224132f8897bb9a3
implementation authority: none until this handoff is accepted at an exact commit
execution authority: none
training authority: none
selection authority: none
Task 010D authority: false
```

This handoff is subordinate to the independently accepted proposal at
`phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md`. The proposal is the
scientific contract. This document freezes the bounded implementation package,
command surface, tests, and return format. It does not authorize a postmortem run.

After this handoff is accepted, `main` may delegate implementation to one
fresh-context Codex subagent with model identifier exactly `gpt-5.5`,
`fork_turns="none"`, working directory `/home/mye/src/llm/CapKnow`, and one mutation
lease. The executor returns an unstaged diff. `main` freezes and independently reviews
the implementation before separately deciding whether to run the postmortem.

## Immutable input

The implementation binds only this accepted artifact package:

```text
input root: artifacts/phase8_toy_lm_bridge/feasibility_005
input source commit: 2dc8c50fab9ece27a07eb0dc04021b78de52895c
manifest SHA-256: 51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b
summary SHA-256: bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785
FAILED SHA-256: 6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08
artifact review: ACCEPT, no findings
manifest inventory entries: 49
cells: 24
generation rows: 1536
terminal: FAILED
```

The implementation must validate the exact canonical path, source commit, top-level
hashes, terminal schema/hash chain, configuration, record hashes, ordered 24 cells,
complete inventory, every generation row, every checkpoint payload/metadata/state,
and every checkpoint/generation path/size/hash before CUDA construction or output
creation. A copied or equivalent-looking root is invalid. Do not reinterpret the
input terminal as a postmortem failure or a feasibility selection.

The four 005 predecessor roots and its D1 decision diagnostic are transitive
provenance and source-cleanliness inputs only. They may never supply a model, record,
metric, or result to D3. Their canonical paths and top-level hashes come only from the
accepted 005 manifest; shallow cleanliness authorization additionally requires each
root's exact terminal/manifest and complete path/size/hash inventory. Do not authorize
a copied root or the artifact parent directory.

## Allowed paths

The executor may modify only:

```text
Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md
phase8/README.md
phase8/Task_010C_Tokenizer_Model_and_Feasibility.md
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

Do not modify this handoff, the accepted D3 proposal, model/tokenizer/train modules,
corpus generators, earlier handoffs, any artifact, or any later Phase 8 task. If the
implementation cannot fit these paths without changing an accepted scientific
contract, stop and return the smallest ambiguity to `main`.

## Required command surface

Add exactly one CLI subcommand named `postmortem-failure`. Its future canonical
command is:

```text
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_sequence_feasibility.py postmortem-failure --device cuda:0 --input-root artifacts/phase8_toy_lm_bridge/feasibility_005 --output-root artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001 --accepted-proposal-commit 336afdc7cb079086998a0db527f544f16b68950c
```

The implementation encodes and tests this command but must not execute it. Refuse
programmatic `main(argv)` and direct-import invocation of the normative writer. The
writer itself, before model construction or output creation, must require its real
`__main__` process context, exact `/proc/self/cmdline`, exact environment variables,
`--device cuda:0`, the two exact roots, and the exact accepted proposal commit.

The proposal binding written later is:

```text
commit: 336afdc7cb079086998a0db527f544f16b68950c
path: phase8/Task_010C_D3_Feasibility_005_Postmortem_Proposal.md
blob: 1c52056e6579262045d81315224132f8897bb9a3
```

The implementation binding is the clean runtime HEAD plus the Git blob of
`scripts/phase8_sequence_feasibility.py`. Do not embed the future implementation
commit into source. `main` supplies and reviews the exact implementation commit in a
later launch packet.

## Preflight order

The exact fail-closed order is:

1. validate argument types and canonical root basenames;
2. require exact real main context, kernel argv, environment strings, and absent final
   and same-parent `.tmp` output roots;
3. shallow-validate the input path, terminal, three top-level hashes, 49-entry
   inventory paths/sizes/hashes, plus the exact 005-bound paths/top-level hashes and
   complete inventories of predecessors 001–004 and D1; derive only those exact files
   as source-clean exclusions;
4. require clean tracked/staged source and reject every ignored executable input;
5. verify current HEAD, proposal commit/blob, runner blob, full A800 runtime object,
   and deterministic flags;
6. deep-validate input manifest/summary/terminal equality, 24 ordered cells,
   generation rows and scores, tied checkpoint schemas/duplicate byte equality, and
   complete input lineage;
7. reconstruct the eight record sets with local PRNGs, verify all eight frozen hashes,
   and prove global Python/Torch CPU/all-CUDA RNG states unchanged before any restore;
8. only then create the `.tmp` root and enter the checkpoint-forward loop.

No tensor load, model construction, CUDA forward, record schedule, or output path may
occur before source cleanliness. Shallow validation may hash files but must not load
checkpoints or construct models. Tests must use call-order sentinels for both failure
and passing paths.

## Read-only checkpoint execution

Implement the accepted proposal literally for all 24 cells. Use the existing tied
checkpoint validator; construct each model on CPU only inside the proposal's single
Torch RNG snapshot/restore scope, load and validate exact state, then move it to
`torch.device("cuda:0")`. Python RNG must not change. After load, snapshot every named
parameter's dtype, shape, layout, and contiguous bytes; require equality after train
teacher forcing, after eval teacher forcing, and before publication.

Use `model.eval()` and `torch.inference_mode()`. Do not instantiate an optimizer,
enable gradients, call backward, mutate parameters/buffers, save a checkpoint, or call
any generation/greedy-decode function. A sentinel around every forbidden operation is
mandatory in production and tests. The only generation evidence is the retained
input JSONL.

For each cell, evaluate exactly its own family records: 512 train records in eight
contiguous 64-row batches followed by 64 eval records in one batch. Preserve exact
record order, `[64,256]` encoding, response-only shifted labels including EOS,
float32 unreduced CE, argmax, `math.fsum` row/cell order, `float.hex()` NLL
serialization, integer-first metrics, and every Named/Array/common rule in the
accepted proposal and its bound D1 metric oracle.

Snapshot process-global Python, Torch CPU, and all-CUDA RNG states at every boundary
required by the proposal. Record reconstruction must compare before any restore.
Only CPU model construction may restore Torch RNG in its local scope. No restore is
allowed afterward. Verify runtime environment/deterministic flags and RNG-state
equality again immediately before publication.

## Closed outputs and publication

Implement exactly the proposal schemas and cardinalities:

```text
teacher_forced_rows.jsonl: 13824 rows
cell_metrics.jsonl: 24 rows
error_taxonomy.jsonl: 1536 rows
summary.json
manifest.json
DONE.json only after complete validation
```

Do not add a file, field, compatibility alias, display float, checkpoint, regenerated
output, record pack, log, status file, or selection. Use the proposal's canonical
JSON/JSONL serialization, exact nested schemas, sparse histogram rules, conditional
denominators, `all_items_copied` predicate, paired four-cell counts, input/proposal/
implementation bindings, runtime objects, row order, and complete four-file manifest
inventory. Independently reconstruct every aggregate from retained rows before
publication; a second forward for aggregation is forbidden.

The temp root is exactly:

```text
artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001.tmp
```

Create it exclusively only after preflight. A finalized FAILED root is forbidden. On
any exception, remove `DONE.json` and `FAILED.json` if present, preserve the incomplete
temp root, do not rename, and re-raise. On success, write summary, then manifest, then
DONE; validate every schema, cardinality, path, hash, aggregate, binding, parameter
snapshot, RNG state, source/runtime snapshot, and inventory; repeat the final
source/runtime/RNG check; then use Linux atomic rename-no-replace. Never overwrite,
delete, retry, or create `feasibility_postmortem_002`.

## Required tests

Add focused tests covering at least:

1. exact CLI argv/environment/main-context acceptance and rejection, including direct
   import and programmatic-main refusal before output/model work;
2. exact input root/top-level hashes/configuration/record hashes/cells/inventory and
   rejection of copies, substitutions, symlinks, missing/extra fields/files, size/hash
   drift, terminal drift, generation mutation, and tied-checkpoint mutation;
3. shallow → clean source → deep input → records/RNG → `.tmp` → model/forward call
   order, with no early tensor/model/CUDA/output work;
4. local record PRNG success, global RNG mutation rejection before restore, the single
   constructor restore scope, and byte-identical Python/Torch CPU/all-CUDA states
   after load/inference/publication;
5. exact 24-cell order, 13,824 teacher rows, 24 cell rows, 1,536 taxonomy rows, batch
   shapes/order, labels including EOS, float32/no-autocast CE, `math.fsum`,
   `float.hex()`, and no second aggregate forward;
6. an independent test oracle for common/Named/Array row fields, null conditions,
   sparse histograms, item-count strata, conditional denominators, item multisets,
   paired exact table, and all summary aggregates;
7. checkpoint metadata copied descriptively without comparing it to fresh
   teacher-forced statistics;
8. sentinels rejecting optimizer construction, grad enablement, backward, post-load
   parameter/buffer mutation, checkpoint save, generation, extra records, or new data;
9. exact JSON/JSONL keys, types, enums, nullability, ordering, canonical bytes,
   cardinalities, bindings, four-file inventory, and one-way DONE→manifest checksum;
10. publication mutation tests for every output file, aggregate, input/proposal/
    implementation binding, runtime/source/RNG callback, no-clobber target, and atomic
    rename; every failure must leave no terminal marker and no final root;
11. permanent rejection of selection use, feasibility verdict change, retry,
    `feasibility_006`, alternate postmortem root, or `010D` authorization;
12. regression coverage for existing D1/D2 validators, current feasibility artifacts,
    and selection rejection.

Run in this order:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q tests/test_lm_bridge.py -k 'd3 or postmortem'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q tests/test_lm_bridge.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m py_compile scripts/phase8_sequence_feasibility.py tests/test_lm_bridge.py
git diff --check
```

Tests must use temporary roots and synthetic fixtures. They must not read CUDA unless
explicitly marked as a non-artifact device smoke, and must never create the normative
postmortem root.

## Documentation boundary

Update only the allowed existing documents to record the implementation status,
exact command, non-evidence classification, DONE-only publication, and permanent 005
stop. Do not mark the implementation accepted before `main` freezes and reviews it.
Do not record a postmortem outcome because this task cannot run it.

## Executor return

Return exactly:

```text
Scope completed:
Prerequisite handoff commit used:
Accepted proposal commit/blob used:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests run and exact results:
New artifacts:
Forbidden-operation sentinels:
Unresolved decisions:
Protocol deviations:
Resource observations:
Independent review required:
Recommended next authorized step:
```

For acceptance, `Scientific contract changes`, `New artifacts`, `Unresolved
decisions`, and `Protocol deviations` are `none`. The recommended next step is only
`main` inspection, commit, and fresh independent strict read-only review of the exact
implementation commit. It is never the postmortem run.

## Independent handoff-review gate

Before delegation, freeze this handoff and review it at an exact commit with a
fresh-context `gpt-5.6-sol`, `xhigh`, strict read-only agent. The review must verify:

- exact proposal and 005 evidence binding;
- executable allowed paths and command/main-context contract;
- shallow/source-clean/deep/RNG/output ordering;
- teacher-forced and taxonomy semantic fidelity;
- RNG-neutral construction and post-load parameter immutability;
- closed schemas, cardinalities, provenance, runtime, DONE-only atomic publication;
- forbidden-operation and tamper test completeness; and
- no training, generation, selection, retry, verdict reversal, or `010D` authority.

Return `ACCEPT` or `REJECT` with findings ordered by severity, exact file/line
references, impact, and the smallest repair. A rejected handoff has no delegation
authority.
