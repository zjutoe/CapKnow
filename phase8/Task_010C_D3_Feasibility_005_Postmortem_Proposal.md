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
2. What is the paired per-record relationship between teacher-forced sequence exact
   and retained greedy exact, reported as the four integer concordance cells?
3. Within the already-retained Array evaluation rows, how do exactness, valid schema,
   item-copy accuracy, first-error position, EOS, and generated length vary across
   target item counts `1,2,3,4`?
4. Within the already-retained Named evaluation rows, how often are JSON syntax,
   target color prefix, four-hex suffix, exact prompt-value copying, first-error
   position, EOS, and target length correct?
5. What are the frozen checkpoint metadata values, reported descriptively and kept
   separate from the newly computed teacher-forced metrics?

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
  update, stochastic inference, selection, or checkpoint write;
- evaluate response-only shifted labels on all `512` training and `64` evaluation
  records in their frozen order with the frozen batch size and padding semantics;
- record sequence exactness, correct/total response tokens, token accuracy, summed
  response NLL, response-token count, and mean loss;
- derive greedy/error statistics only from the retained generation rows. Do not call
  generation or produce replacement outputs.

Record construction may use only the private local `random.Random` instances needed
to reproduce the eight hash-bound record sets. Snapshot process-global Python, Torch
CPU, and every CUDA RNG state immediately before reconstruction and compare all states
immediately afterward, before any restore; any difference fails and no restore is
permitted. Model construction is the sole exception: snapshot Torch CPU/all-CUDA
states immediately before the constructor, construct on CPU, restore exactly once in
the same scope, and verify equality immediately. Python RNG must remain unchanged.
After that scope, no restore is allowed and all states must remain byte-equal through
load, inference, and publication. Sentinels must reject global RNG API use outside the
single constructor scope.

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

The complete runtime `environment` and `deterministic_flags` objects must equal the
input manifest field-for-field before CUDA construction and before publication.
Parameters, logits, and loss are float32; autocast is disabled; CUDA/cuDNN TF32 are
false; deterministic algorithms and cuDNN determinism are true; and cuDNN benchmark
is false.

## Exact teacher-forced and taxonomy semantics

The metric oracle is the complete `## Frozen metric semantics` definition in Git
blob `b5a45e3a33baf53d4de9c809bc9ab2c59f400e37` of
`phase8/Task_010C_D1_Feasibility_Failure_Diagnostic.md`. A future implementation must
reproduce that definition rather than call an opaque aggregate helper.

Teacher forcing predicts every response byte plus EOS with exact response-only
shifted labels, `ignore_index=-100`, float32 unreduced cross-entropy, and
`logits.argmax(dim=-1)` at selected source positions. Inputs/labels are int64; exact
shapes are `[64,256]` inputs, `[64,256]` labels, `[64,256,259]` logits, and
`[64,256]` loss. Process original record order as eight contiguous train batches and
one eval batch, with no partial batch, shuffle, autocast, or second aggregate forward.
Per-row NLL is `math.fsum` of selected host Python float values in ascending position
order; cell NLL is `math.fsum` of row NLL values in record order. Serialize NLL sums
and means with `float.hex()`; rates retain only integer numerator and denominator as
authoritative values.

The bound oracle also fixes generation slicing, EOS/cap and token/UTF-8 length
semantics, byte exact/Hamming/Levenshtein/first-error rules, JSON top-level types,
Named grammar/prefix/prompt-occurrence/suffix rules, and Array string-list,
positional, multiset, duplicate, and null rules. First-error is zero-based, uses the
shorter byte length for a strict prefix, and is null for exact or invalid decode.
Missing Named suffix bytes count incorrect; extra suffix bytes are ignored only for
positional accuracy. Invalid Arrays have false schema/count flags and null item
metrics. Array strata use target item count `1,2,3,4`; all aggregates sum row-level
numerators and denominators and never use macro means.

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
- `error_taxonomy.jsonl`, row-level facts derived from retained generation rows;
- `summary.json`, `manifest.json`, and exactly one terminal marker.

No checkpoint, generated replacement, corpus, model-facing record, selection record,
or scientific evidence may be written. Every row binds its input checkpoint and
generation path/hash and the original record identity. The manifest binds the exact
input root/top-level hashes, complete input inventory, source commit, command,
environment, configuration, file inventory, and all aggregate identities.

`DONE` means only that the complete postmortem was computed and validated. It is not
a feasibility pass. A finalized FAILED root is forbidden. Any operational or
validation error removes both terminal markers and preserves only an explicitly
incomplete temporary root. Scientific values never cause failure and never trigger a
retry. No `feasibility_postmortem_002` fallback is proposed.

All JSON is UTF-8 with sorted keys, two-space indentation, and one trailing newline.
JSONL is UTF-8 with sorted keys, compact separators, `ensure_ascii=false`, and one
newline per row. NaN and infinity are forbidden. `schema_version` is exactly
`phase8_feasibility_postmortem_v1`. The closed schemas are:

- `teacher_forced_rows.jsonl`: exactly `13,824` rows ordered by family, model size,
  seed, train before eval, and original record index. Exact keys are
  `schema_version,family,model_size,seed,split,record_index,template_id,operand_id,`
  `batch_index,row_within_batch,selected_token_count,correct_token_count,`
  `sequence_exact,nll_numerator_hex,checkpoint_path,checkpoint_sha256`.
- `cell_metrics.jsonl`: exactly 24 rows in frozen cell order. Exact keys are
  `schema_version,family,model_size,seed,checkpoint_path,checkpoint_sha256,`
  `generation_path,generation_sha256,checkpoint_metadata,train_teacher_forced,`
  `eval_teacher_forced,retained_greedy_exact,paired_eval_exact`. Each teacher-forced
  object contains exactly `record_count,sequence_exact_numerator,selected_token_count,`
  `correct_token_count,nll_numerator_hex,nll_per_token_hex`; retained greedy contains
  `numerator,denominator`; paired eval contains integer `both_exact,tf_only,`
  `greedy_only,neither_exact,denominator,count_difference`, where
  `count_difference = tf_only - greedy_only`.
  `checkpoint_metadata` has exactly `family,model_size,seed,training_steps,`
  `training_loss,training_accuracy`, copied without reinterpretation from the bound
  checkpoint.
- `error_taxonomy.jsonl`: exactly 1,536 rows in frozen cell/eval-record order. Exact
  top-level keys are `schema_version,family,model_size,seed,record_index,`
  `generation_path,generation_sha256,common,named,array`. `common` contains exactly
  `prompt,expected_response,generated_response,raw_token_ids,generation_error,`
  `greedy_exact,decode_valid,has_eos,hits_generation_cap,hits_context_cap,`
  `generation_token_count,generation_utf8_bytes,target_utf8_bytes,length_matches,`
  `hamming_distance,edit_distance,first_error`. `named` is null outside Named and
  otherwise has exactly `surface_source,operand_source,valid_json_string,`
  `valid_named_grammar,target_prefix,occurs_in_prompt,exact_target_value,`
  `exact_distractor_value,suffix_positional_correct,suffix_positional_total,`
  `suffix_hamming_distance,suffix_edit_distance,suffix_first_error`; source labels are
  exactly `held_surface` and `held_operand`, positional total is 4 only for a parsed
  target-prefix row, and all suffix fields are null when the bound oracle declares
  them undefined. `array` is null outside Array and otherwise has exactly
  `comparison_source,training_steps,target_item_count,valid_json_syntax,`
  `valid_array_schema,correct_item_count,positional_exact_count,`
  `positional_denominator,missing_items,extra_items,all_items_copied,`
  `all_items_exact,first_wrong_item_position`. `comparison_source` is exactly
  `feasibility_005_retained_eval`, `training_steps` is 1500, and target item count is
  `1..4`. `all_items_exact` means valid schema, correct count, and positional exact
  count equal to target count. `all_items_copied` means valid schema and the generated
  item multiset is a sub-multiset of all literal string operands in the prompt;
  duplicate multiplicity is respected. Invalid arrays have false booleans and null
  positional/missing/extra/first-wrong fields. Missing/extra lists are the
  lexicographically sorted expanded `Counter` differences. First wrong is the first
  unequal position, the shorter length for a strict prefix, and null for exact or
  invalid schema.
- `summary.json`: exact keys `schema_version,artifact_class,input_binding,`
  `proposal_binding,implementation_binding,configuration,row_counts,cells,`
  `named_aggregates,array_aggregates,interpretation_limits,terminal_status`.
  `cells` is the exact ordered 24-row `cell_metrics` content. `named_aggregates` is
  exactly six rows ordered by model size and seed, each with exact keys
  `model_size,seed,denominator,valid_json_string_count,valid_named_grammar_count,`
  `target_prefix_count,occurs_in_prompt_count,exact_target_value_count,`
  `exact_distractor_value_count,suffix_positional_correct,`
  `suffix_positional_denominator,suffix_first_error_histogram,`
  `suffix_first_error_observation_count`; the histogram has exactly string keys
  `0,1,2,3,4`. `array_aggregates` is exactly 24 rows ordered by model size, seed, and
  target item count, each with exact keys
  `model_size,seed,target_item_count,denominator,valid_json_syntax_count,`
  `valid_array_schema_count,correct_item_count_count,positional_exact_count,`
  `positional_denominator,all_items_copied_count,all_items_exact_count,`
  `missing_item_count,extra_item_count,first_wrong_histogram,`
  `first_wrong_observation_count`; its histogram has exactly string keys
  `0,1,2,3,4`. Every denominator is 64 for Named and 16 for an Array stratum.
  `interpretation_limits` has exactly four true boolean keys
  `non_evidence,no_causal_weight_tying_claim,no_verdict_change,no_010d_authority`.
- `manifest.json`: exact keys `schema_version,artifact_class,input_binding,`
  `proposal_binding,implementation_binding,runner_path,exact_command,environment,`
  `deterministic_flags,rng_state_contract,configuration,row_counts,file_inventory,`
  `terminal_status`. Its inventory covers exactly the four non-manifest,
  non-terminal files, including `summary.json`; it excludes itself and the terminal.
- `DONE.json` exact keys are `schema_version,status,manifest_path,manifest_sha256,`
  `input_manifest_sha256,row_counts`. A finalized `FAILED.json` is invalid. The DONE
  terminal binds the manifest in one direction; the manifest never embeds its own
  checksum.

All identity/path/hash/NLL fields are strings; hashes are lowercase 64-hex and paths
are canonical root-relative paths. Counts and indices are JSON integers (booleans are
not integers); flags are JSON booleans. `family`, `model_size`, `split`, and status
are closed enums. Null is permitted only for the decode- or family-conditional fields
declared null by the bound metric oracle. Lists contain only their declared scalar
type. Nested objects have the exact keys above or in the bound oracle; extra keys and
implicit coercions are rejected.

`proposal_binding` records the exact independently accepted proposal commit and blob
supplied externally by `main`; it is not self-embedded here. `implementation_binding`
records runtime HEAD and runner blob. `input_binding` separately records the 005
source commit, three top-level hashes, terminal, full 49-entry inventory, and all 24
cell/checkpoint/generation identities. No field may be omitted or added.

Those nested schemas are also closed. `proposal_binding` has exactly
`commit,path,blob`; `implementation_binding` has exactly `commit,runner_path,`
`runner_blob`; `input_binding` has exactly `root,source_commit,manifest_path,`
`manifest_sha256,summary_path,summary_sha256,terminal_path,terminal_sha256,`
`configuration,record_hashes,file_inventory,cells`. `row_counts` has exactly
`teacher_forced_rows:13824,cell_metrics:24,error_taxonomy:1536`.
`rng_state_contract` has exactly true booleans
`record_reconstruction_global_state_unchanged,constructor_scope_restored,`
`post_load_state_unchanged,post_inference_state_unchanged`. `configuration`,
`environment`, and `deterministic_flags` equal their 005 manifest objects exactly.
Every `file_inventory` row has exactly `path,sha256,bytes`, is sorted by path, and
binds the complete four-file set.

Publication uses same-parent
`artifacts/phase8_toy_lm_bridge/feasibility_postmortem_001.tmp`. Both final and temp
must be absent before exclusive no-clobber creation. DONE requires all cardinalities,
schemas, checksums, aggregates, lineage, RNG-state equality, runtime/source equality,
and inventory to validate before a final source/runtime check and atomic
rename-no-replace. Any exception removes both terminal markers and preserves an
explicitly incomplete temp root; it never renames. Scientific values never cause
failure or retry.

## Frozen metrics and interpretations

All metrics must be integer-count-first. Ratios are derived display fields. Report at
least:

- train/eval teacher-forced sequence exact, token correct/total, and NLL/token;
- retained greedy exact and the paired per-record teacher-forced/greedy integer table
  (`both_exact`, `tf_only`, `greedy_only`, `neither_exact`) plus count difference;
- EOS present, generation-cap hit, target-length match, valid UTF-8, and first-error
  byte position;
- Named valid JSON string, valid color-four-hex grammar, target-prefix match, and
  exact prompt-value copy;
- Array valid JSON string-array, correct item count, per-position exact item copy,
  all-items-copied, and the same metrics stratified by target item count `1..4`.

Permitted interpretations are deliberately bounded:

- train teacher-forced counts describe checkpoint fit but cannot distinguish
  incomplete optimization from capacity limits;
- the separately reported train and eval numerator/denominator pairs describe fit and
  held-out behavior; raw count subtraction across denominators 512 and 64 is forbidden;
- the paired eval four-cell table localizes teacher-forced/greedy discordance without
  assigning a registered materiality threshold;
- item-count-stratified Array counts describe composition-length association without
  establishing a causal burden;
- checkpoint metadata `training_accuracy` and `training_loss` are descriptive only.
  The former is a post-training greedy pass over 512 train records; the latter is the
  pre-update mean CE of the final sampled batch. Neither is commensurate with the new
  full-set teacher-forced metrics, so no agreement or inconsistency claim is allowed.

Terms such as “high”, “low”, “material”, and “degradation” have no registered
threshold and must not classify results. Report integer tables, denominators, count
differences, and NLL hex values; qualitative interpretation remains non-decisional.

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
   generation call can occur, plus byte-identical process-global Python, Torch CPU,
   and all-CUDA RNG states across reconstruction, load, inference, and publication;
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
