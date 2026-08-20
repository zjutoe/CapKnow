# Task 010C-D3 — Feasibility 005 Read-Only Postmortem Proposal

## Status and authority

```text
status: proposal pending independent protocol review
artifact class: source-controlled postmortem proposal
implementation authority: none until this proposal is accepted at an exact commit
execution authority: none until a separate implementation commit is accepted
training authority: none
feasibility retry authority: none
selection authority: none
Task 010D authority: false
```

This proposal does not reopen the failed feasibility gate. It freezes a possible
checkpoint-read-only postmortem that may explain which already-observed failure modes
are consistent with optimization, teacher-forced prediction, or greedy decoding. It
does not amend the model, data, optimizer, `52/64` threshold, 1500-step budget, D2
stop rule, or scientific interpretation.

Acceptance of this proposal would authorize only a separate implementation handoff.
That implementation must be delegated to a fresh-context `gpt-5.5` executor, frozen
in a commit, tested, and independently reviewed before `main` may authorize one
postmortem command. This proposal itself authorizes no code change or execution.

## Immutable input and accepted outcome

The sole normative input is:

```text
root: artifacts/phase8_toy_lm_bridge/feasibility_005
source commit: 2dc8c50fab9ece27a07eb0dc04021b78de52895c
manifest SHA-256: 51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b
summary SHA-256: bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785
FAILED SHA-256: 6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08
artifact review: ACCEPT, no findings
terminal: FAILED
cells: 24
passed cells: 11
failed cells: 13
```

The accepted cell scores, ordered by small seeds `0,1,2` and medium seeds `0,1,2`,
are:

| Family | Small | Medium |
| --- | --- | --- |
| `hex_copy` | `57,59,63` | `51,64,62` |
| `named_value_json` | `0,0,0` | `2,2,8` |
| `boolean_json` | `64,64,64` | `64,64,64` |
| `array_json` | `2,0,0` | `12,15,13` |

The accepted read-only failure diagnosis found that EOS absence affected only
`14/1536` rows and does not explain the failures. Named-value outputs usually chose
the target key prefix but failed to copy the held-out suffix. Array outputs often had
valid structure and item count but failed to copy all item contents. These are
descriptive observations, not causal conclusions about weight tying.

The postmortem must reject any input path, top-level checksum, terminal, source
commit, configuration, cell identity, inventory entry, checkpoint metadata, or
generation-row mismatch. It must not accept copied files or an equivalent-looking
root.

## Frozen questions

The postmortem may answer only these questions:

1. For every existing checkpoint, how different are response-only teacher-forced
   sequence exactness, token accuracy, and loss on its exact frozen training and
   evaluation records?
2. Are low greedy exact-match cells also weak under teacher forcing, or is there a
   material teacher-forced/greedy gap consistent with exposure or decoding failure?
3. Within the already-retained Array evaluation rows, how do exactness, valid schema,
   item-copy accuracy, first-error position, EOS, and generated length vary across
   target item counts `1,2,3,4`?
4. Within the already-retained Named evaluation rows, how often are JSON syntax,
   target color prefix, four-hex suffix, exact prompt-value copying, first-error
   position, EOS, and target length correct?
5. Do the small checkpoints' final recorded training metrics agree with a fresh
   teacher-forced evaluation of the frozen training records, or is there evidence of
   checkpoint/metric inconsistency?

The postmortem may not construct a seen/held surface-by-operand factorial pack. Such
a pack would introduce a new evaluation distribution and requires its own separately
reviewed record-pack proposal with exact record hashes. D1's historical factorial
results may appear only as clearly labelled descriptive background.

## Exact read-only computation boundary

Use only the 24 inventory-bound `checkpoint_step1500.pt` files, their corresponding
24 retained `generations.jsonl` files, and the exact source-controlled train/eval
records whose eight hashes are already bound by the input manifest.

For each family, model size, and seed:

- load the tied checkpoint fail-closed under its exact configuration and metadata;
- use `model.eval()` and `torch.inference_mode()`;
- perform no optimizer construction, gradient computation, backward pass, parameter
  update, RNG-dependent operation, or checkpoint write;
- evaluate response-only shifted labels on all `512` training and `64` evaluation
  records in their frozen order with the frozen batch size and padding semantics;
- record sequence exactness, correct/total response tokens, token accuracy, summed
  response NLL, response-token count, and mean loss;
- derive greedy/error statistics only from the retained generation rows. Do not call
  generation or produce replacement outputs.

All forward passes use exactly `torch.device("cuda:0")` on the frozen A800 runtime.
CPU fallback and alternate devices are forbidden. Hashing, JSON validation, and
checkpoint schema inspection may use CPU. The exact environment remains:

```text
PYTHONDONTWRITEBYTECODE=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=.
```

The future implementation must freeze the exact command and require real kernel argv,
real environment, clean committed source, the absent output root, and complete input
validation before CUDA model construction or output-root creation.

## Proposed output contract

The only proposed future root is:

```text
artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001
```

Its fixed classification is:

```text
artifact_class: non_evidence_feasibility_postmortem
feasibility_selection_eligible: false
task_010d_authorized: false
changes_feasibility_005_verdict: false
```

It may contain only:

- `teacher_forced_rows.jsonl`, one row per original record per cell;
- `cell_metrics.jsonl`, one exact aggregate per 24 cells;
- `error_taxonomy.jsonl`, aggregates derived from retained generation rows;
- `summary.json`, `manifest.json`, and exactly one terminal marker.

No checkpoint, generated replacement, corpus, model-facing record, selection record,
or scientific evidence may be written. Every row binds its input checkpoint and
generation path/hash and the original record identity. The manifest binds the exact
input root/top-level hashes, complete input inventory, source commit, command,
environment, configuration, file inventory, and all aggregate identities.

`DONE` means only that the complete postmortem was computed and validated. It is not
a feasibility pass. An operational or validation error publishes `FAILED` if a valid
terminal can be constructed; scientific values never cause postmortem failure and
must never trigger a retry. No `feasibility_postmortem_002` fallback is proposed.

## Frozen metrics and interpretations

All metrics must be integer-count-first. Ratios are derived display fields. Report at
least:

- train/eval teacher-forced sequence exact, token correct/total, and NLL/token;
- retained greedy exact and the teacher-forced-minus-greedy sequence-exact gap;
- EOS present, generation-cap hit, target-length match, valid UTF-8, and first-error
  byte position;
- Named valid JSON string, valid color-four-hex grammar, target-prefix match, and
  exact prompt-value copy;
- Array valid JSON string-array, correct item count, per-position exact item copy,
  all-items-copied, and the same metrics stratified by target item count `1..4`.

Permitted interpretations are deliberately bounded:

- low train teacher-forced performance is consistent with incomplete fitting or
  capacity limits, but does not distinguish them;
- high train and low eval teacher-forced performance is consistent with held-out
  generalization failure;
- high eval teacher-forced and low retained greedy performance is consistent with an
  exposure/decoding gap;
- degradation with Array item count is consistent with a composition-length burden;
- agreement between recorded and recomputed training metrics supports checkpoint
  consistency.

None of these patterns proves a unique cause. Cross-root comparisons with 001–004 or
D1 are descriptive only because source, protocol, and parameterization differ. The
postmortem cannot establish that weight tying caused improvement or degradation.

## Stop rule and prohibited actions

Regardless of postmortem values:

- `feasibility_005` remains the accepted formal `FAILED` result;
- do not create a feasibility selection;
- do not authorize or implement `010D`;
- do not train, fine-tune, resume, increase steps/data, change thresholds, add seeds,
  select a model, change decoding, regenerate rows, or launch `feasibility_006`;
- do not use the postmortem to choose among unregistered repairs;
- do not implement or run a new protocol without a new source-controlled proposal
  and independent review.

After an accepted postmortem, the only allowed main-thread actions are reporting the
failure, archiving the accepted evidence, or drafting a separate protocol proposal.

## Required implementation tests if this proposal is accepted

A future D3 implementation handoff must require tests for:

1. exact input path, three top-level hashes, source commit, terminal, configuration,
   24 cell identities, complete inventory, and every checkpoint/generation hash;
2. rejection of copied roots, missing/extra files, symlinks, untied or mismatched
   checkpoints, divergent generation rows, and source/environment drift;
3. exact teacher-forced label alignment, response masks, record order, batching,
   integer aggregates, and loss reconstruction;
4. independent recomputation of Named and Array taxonomies, including item-count
   strata and first-error positions;
5. sentinels proving no optimizer, backward, parameter mutation, checkpoint save, or
   generation call can occur;
6. source-clean preflight before CUDA construction or output creation, no-clobber
   temporary-root publication, terminal/inventory validation, and source-unchanged
   checks before atomic rename;
7. permanent rejection of selection use, feasibility verdict changes, `010D`
   authorization, retry, or alternate-root fallback.

## Independent proposal-review gate

Review this proposal at an exact commit with a fresh-context `gpt-5.6-sol` agent at
`xhigh` reasoning. The review must return `ACCEPT` or `REJECT` and decide whether:

1. every computation reads only frozen 005 evidence and source-controlled records;
2. the proposed metrics answer the bounded questions without creating a new data
   distribution or silently repeating generation;
3. teacher-forced execution cannot mutate models or become training;
4. artifact schemas, lineage, device, environment, and atomic publication are
   reproducible and fail closed;
5. the postmortem cannot become selection evidence, reverse the 005 verdict, or
   authorize `010D`; and
6. interpretations avoid causal overclaiming.

Findings must be ordered by severity with exact file/line references, impact, and the
smallest repair. A rejected proposal has no implementation or execution authority.
