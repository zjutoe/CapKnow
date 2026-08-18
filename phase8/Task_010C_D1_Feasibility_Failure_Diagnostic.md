# Task 010C-D1 — Feasibility Failure Diagnostic

## Status and authority

This handoff freezes a bounded, non-scientific diagnostic after the independently
accepted `feasibility_004` failure. It supplements Task 010C but does not amend the
master Phase 8 model, optimizer, feasibility gate, or acceptance contract.

The diagnostic cannot:

- turn a failed feasibility root into a pass;
- create or validate a feasibility selection record;
- authorize Task 010D, a resource benchmark, a formal root, or any Phase 8
  scientific corpus or checkpoint;
- supply evidence for changing the formal `1500`-step budget, model sizes, exact
  scoring, or `52/64` threshold.

Only `main` may interpret the completed diagnostic and authorize a later protocol
decision in a separate source-controlled commit.

## Frozen prerequisite evidence

```text
source commit:
  3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea

feasibility_004:
  root:     artifacts/phase8_toy_lm_bridge/feasibility_004
  manifest: 19cf2db4df6f6928d0afa7ab8aef8d891e944152e480541f5aaf8702565e8d00
  summary:  4eaf7b5742161ff6ea5612f7c53668ee71c38f598e18da27a375fafebb6dc506
  FAILED:   6b3c81d3be78c8e3deeaf991dad4a1e2df41171187ba4e20533799162a46f3ae
```

The evidence-package review verdict is `ACCEPT`; the empirical feasibility decision
is `FAIL`. The diagnostic implementation must reject any checksum, terminal,
configuration, source, inventory, or predecessor-lineage mismatch in this input.

The implementation commit may differ from the input source commit only by this
reviewed handoff/index plus the reviewed diagnostic implementation and its tests. It
must regenerate the current Named and Array records and prove that their prompts,
expected answers, template IDs, operand IDs, ordering, and split semantics still
match `feasibility_004` before using its checkpoints or generations.

The following core Git blobs from the input source commit must remain exact in the
implementation and execution commits:

```text
tokenizer.py: 79ded37d148d40b0ba4e501881f585d74e971798
model.py:     70e57d8ccea8773b99ec797634e3ea32fe0bb9bf
train.py:     fe2d6a901c96dc35a5e90b12996d6c01df2f2f56
```

Canonical record serialization is UTF-8 JSONL with one `dataclasses.asdict(record)`
object per line, `ensure_ascii=False`, `sort_keys=True`, separators `(",", ":")`,
and exactly one trailing newline per record. The frozen record-set hashes are:

```text
named train512:      9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8
named train first64: 6806025fa541c4f71d84a2dfce29777e0ac4e056c1bad762aeb10d309f5503ad
named eval64:        6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20
array train512:      6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94
array eval64:        8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a
```

These precomputed hashes are the input oracle. Recomputing them from the new
implementation is a required comparison, not a way to define new expected values.

## Launch contract

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact independently accepted implementation commit, this
handoff, the input-root bindings above, the fixed output root, and the only
authorized command. The implementation and diagnostic run use separate mutation
leases. The diagnostic run is permitted only after its implementation commit passes
targeted and full tests and an independent strict read-only review.

The only initial execution command is:

```text
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_sequence_feasibility.py diagnose-failure --device cuda:0 --input-root artifacts/phase8_toy_lm_bridge/feasibility_004 --output-root artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

The runner must reject every other device string. All model forward passes,
diagnostic training, teacher-forced evaluation, and greedy decoding run on
`cuda:0`; file hashing and checkpoint-schema inspection may run on CPU.

## Objective

Probe the remaining failures without tuning on a Phase 8 scientific outcome:

1. For `named_value_json` medium, distinguish sensitivity to held-out prompt
   surfaces from failure to copy held-out operands after target-key selection.
2. For `array_json` small, measure sensitivity to one doubled diagnostic budget and
   compare it descriptively with the frozen model-size package while retaining the
   same records, templates, item-count balance, tokenizer, scoring, seeds, and
   optimizer.

No other family or hypothesis is in scope. Hex and Boolean are immutable controls;
their `feasibility_003` and `feasibility_004` generations and checkpoints were
byte-identical and must not be rerun.

## Diagnostic A — Named surface × operand matrix

Use only the three frozen `feasibility_004` medium checkpoints. Do not retrain them.
For each seed, evaluate exactly four 64-record cells:

| cell | prompt surface | operand source |
|---|---|---|
| `seen_surface_seen_operand` | train | first 64 train records |
| `held_surface_seen_operand` | eval | first 64 train records |
| `seen_surface_held_operand` | train | 64 eval records |
| `held_surface_held_operand` | eval | 64 eval records |

For every operand record, render the train and eval surface with the same surface
index, target key, complete four-field roster, and expected answer. Train/eval
surface pairs must retain their declared byte-length equality. The first 64 train
records and the 64 eval records must each retain exact `template × target-key`
balance: four records in every one of the sixteen cells.

Each row retains the original operand source split, source index, operand ID, roster,
target, and answer. It uses the existing exact template ID for the chosen surface
split; it must not relabel an eval operand as train or a train operand as eval. The
diagnostic cell label is separate metadata and must not enter model-facing text.

The matrix is paired within each operand source:

- `seen_surface_seen_operand` versus `held_surface_seen_operand` changes only the
  surface;
- `seen_surface_held_operand` versus `held_surface_held_operand` changes only the
  surface.

Across operand sources, report differences descriptively. Seen operands are training
examples and therefore measure memorized-operand transfer across surfaces, not
held-out generalization. Do not call the four cells independent replicates or use
them as an acceptance gate.

Retain raw generations and report per seed/cell:

- exact sequence matches;
- valid JSON-string and fixed Named grammar counts;
- requested target-prefix counts;
- exact target-value and exact distractor-value counts;
- whether the decoded value occurs anywhere in the prompt;
- suffix character accuracy, Hamming/edit distance, and first-error position;
- `template × target-key` counts and exact matches;
- EOS, generated-length, generation-cap, and context-window diagnostics.

This matrix separates observable target-prefix selection from exact suffix copying.
Do not add a novel one-field or copy-only prompt: an untrained prompt family would
introduce another surface/task confound.

## Diagnostic B — Array budget × model-package comparison

Use the exact current `array_json` train/eval records and the three frozen seeds.
The fixed comparison is:

| model | steps | source |
|---|---:|---|
| small | 1500 | reuse `feasibility_004` checkpoint/generations |
| medium | 1500 | reuse `feasibility_004` checkpoint/generations |
| small | 3000 | one new diagnostic training run per seed |

The only new training consists of exactly three `small / 3000-step` runs. The single
alternative budget is frozen at exactly twice the formal 1500-step value before the
diagnostic result is observed. Initialize from scratch with the existing seed
contract; do not resume the 1500-step checkpoint. Use the unchanged Task 010C
tokenizer, small configuration, response-only loss, AdamW settings, batch size `64`,
deterministic backend, greedy decoding, and train record shuffle schedule. Extending
to step 3000 reuses the same deterministic shuffle RNG stream across successive
cycles.

Each 3000-step run must use one continuous model, optimizer, and precomputed
3000-batch schedule. Immediately after optimizer step 1500 and before step 1501, it
must compare the config, parameter count, state-dict keys, tensor dtypes, shapes, and
every tensor byte-for-byte with the corresponding `feasibility_004` small/1500
checkpoint. A mismatch publishes a diagnostic `FAILED` root and the run must not
continue to step 1501. Snapshotting/comparison must not reset or advance any model,
optimizer, corpus-order, Python, or Torch RNG state.

Because the frozen core training file cannot be edited, the diagnostic-only loop in
the feasibility script must call the existing public `make_optimizer`,
`deterministic_batch_indices`, `encode_record_batch`, and `response_only_loss`
primitives and preserve the operation order of `train_text_records`. The step-1500
tensor equality gate is the executable oracle for that loop.

The execution environment dictionary must exactly equal the `feasibility_004`
manifest environment, including Python, PyTorch, CUDA runtime, GPU, driver, and
platform fields. Together with the frozen core Git blobs, record hashes, and
step-1500 tensor equality, this permits the `small/1500` versus `small/3000`
comparison to be described as changing only the number of continued optimizer
updates. Without every equality check, the diagnostic must fail and no step
sensitivity claim is allowed.

The `small/1500` versus `medium/1500` comparison changes the whole frozen model-size
package, not a single identifiable capacity parameter. Do not add an intermediate
model, another step count, another seed, early-stopping selection, or a threshold
search after seeing results.

Retain the three new checkpoints and generations. For all three comparison rows,
report per seed:

- train and eval teacher-forced sequence exact, token accuracy, and loss;
- greedy eval exact matches, with no pass/fail reinterpretation;
- exact matches by item count and `template × item-count`;
- JSON validity, correct item count, expected/missing/extra item counts;
- equal-length Hamming/edit distance and first-error position;
- EOS, generated-length, generation-cap, and context-window diagnostics.

The 3000-step cell is a diagnostic counterfactual, not a candidate feasibility
configuration. Its result cannot be selected or silently substituted into the
master gate.

## Frozen metric semantics

Every Named row retains its operand-source and surface-source labels; every Array row
retains its comparison source, model size, steps, and seed. Every evaluated row also
retains its prompt, expected response, generated response or JSON `null`, full raw
token IDs, decoded generation error or JSON `null`, and exact-match flag. The
generation slice begins immediately after the exact encoded evaluation prefix. A row
has EOS only when that slice ends in exactly one valid EOS accepted by
`decode_generated_response`. It hits the generation cap only when no valid EOS
exists and the slice contains 64 tokens; it hits the context cap only when prefix
plus generation contains 256 tokens without a valid earlier EOS. Report generation
length both as slice token count including EOS when present and as decoded response
UTF-8 byte count excluding EOS; decoded byte count is JSON `null` on invalid decode.

Common response metrics use UTF-8 bytes:

- greedy exact means valid decoding followed by byte equality with the expected
  response;
- response Hamming distance is an integer only for valid decoded responses with the
  same byte length as expected, otherwise JSON `null`;
- response edit distance is byte-level Levenshtein distance for every valid decoded
  response, otherwise JSON `null`;
- first error is the zero-based byte index of the first mismatch, or the shorter
  length when one byte string is a strict prefix; it is JSON `null` for exact or
  invalid decoded responses.

Teacher-forced metrics predict every response byte plus EOS selected by the existing
response-only labels. Sequence exact requires every selected label, including EOS,
to equal the argmax token. Token accuracy is micro-averaged as total correct selected
tokens divided by total selected tokens. Loss is summed cross-entropy over all
selected tokens divided by that same token count. Record- or batch-macro alternatives
are forbidden.

Named row metrics parse the decoded response with `json.loads`. `valid_json_string`
requires exactly one JSON string; `valid_named_grammar` additionally requires the
existing fixed Named full-match grammar. `target_prefix` requires a parsed string
starting with the requested `<target>-`, even if its suffix length is wrong. Exact
target/distractor counts compare the entire parsed string against the four roster
values; distractors use the other three roster entries. `occurs_in_prompt` is literal
whole parsed-string occurrence in the prompt.

Named suffix positional accuracy is conditional on parsed target-prefix rows. The
expected suffix has four ASCII bytes; each of its four positions is correct only
when the generated suffix has the same byte at that position, with missing positions
counted incorrect and extra positions ignored. Report the numerator, denominator
`4 * target_prefix_rows`, and rate; if the denominator is zero, the rate is JSON
`null`. Suffix Hamming is defined only when the generated suffix is exactly four
bytes; suffix edit distance is defined for every parsed target-prefix suffix. Other
rows receive JSON `null` for suffix distances.

Array row metrics require `json.loads` to produce a list containing only strings.
Correct item count compares list lengths. Positional exact items compare equal items
at the same index. Missing and extra items use `collections.Counter` multiset
subtraction, retaining duplicates. For invalid arrays, correct-count is false and
positional/missing/extra item metrics are JSON `null`; JSON syntax and schema failures
are also counted separately. Full-response Hamming/edit/first-error retain the common
definitions above.

Every count or rate record must store its explicit numerator and denominator. A mean
stores its observation count and is JSON `null` when that count is zero. Empty or
undefined values must be JSON `null`, never `NaN`, infinity, or a fabricated zero.
Aggregate results are sums of row-level numerators and denominators, never unweighted
means of seed, record, template, or batch rates. Edge-case fixtures and a separately
implemented test oracle must reconstruct every metric exactly.

## Fixed output and provenance

The initial output root is:

```text
artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

Diagnostic roots use the exact basename `feasibility_diagnostic_NNN`, starting at
`001` with continuous numbering. The runner must refuse overwrite and use a
same-parent `.tmp` root followed by an atomic rename. Preflight refusal occurs before
creating either root. After the temporary root exists, a catchable execution error
must finalize a partial summary and checksum-bound manifest, publish `FAILED.json`,
and atomically rename the root. A source change, host/process death, or other failure
that prevents trustworthy finalization leaves the temporary root incomplete, creates
no manufactured terminal, blocks reuse of that number, and returns to `main`.

A completed diagnostic publishes `DONE.json` only when all 12 Named evaluation
cells, all three new Array runs, all six reused Array baseline bindings, and all
declared reports are complete. Scores never affect `DONE`; it is not a feasibility
pass. A `FAILED` root records the exact completed and partial scope and may omit cells
that were never reached.

A retry is permitted only after every earlier diagnostic root is finalized `FAILED`.
It uses the next continuous number, preserves all earlier roots, and passes every
earlier path in ascending order as repeated
`--predecessor-diagnostic-root <path>` arguments. Each predecessor binding includes
its terminal state, terminal SHA-256, manifest SHA-256, source/configuration, and
complete transitive diagnostic lineage. A prior `DONE` forbids every later diagnostic
root. The retry command is otherwise byte-for-byte the initial command with only the
new output root and required predecessor arguments added.

Every finalized failure is classified as `transient_infrastructure` or
`diagnostic_implementation_defect`; diagnostic scores are never a failure class. A
transient retry must use the identical accepted source and configuration. An
implementation-defect retry requires a new committed and independently reviewed
repair, uses the next root, binds the superseded source/root, and may change only the
diagnostic implementation/tests allowed here—not the frozen matrix or metrics.

The manifest and terminal must bind:

- exact diagnostic implementation commit and clean-source snapshot;
- this handoff path and Git binding;
- exact authorized command and output root;
- `feasibility_004` manifest, summary, and terminal paths/checksums plus its complete
  predecessor closure;
- every earlier diagnostic-root terminal/manifest binding and complete transitive
  diagnostic lineage;
- for `FAILED`, its failure classification and, where applicable, the exact
  superseded/repaired source binding;
- frozen record identities and checksums for every reused/train/eval diagnostic set;
- for `DONE`, all four Named cells for three seeds, all three new Array cells, and all
  six reused Array baseline bindings; for `FAILED`, the exact completed/partial scope;
- unchanged tokenizer/model/optimizer constants and the explicit diagnostic-only
  `3000`-step value;
- Python, PyTorch, CUDA, GPU, driver, deterministic flags, and wall time;
- every retained generation, checkpoint, metric table, and summary path, size, role,
  and SHA-256, excluding only the manifest and later terminal marker.

Source cleanliness may exclude only the exact checksum-bound feasibility roots `001`
through `004` supplied through the accepted input lineage, every explicitly supplied
inventory/checksum-bound earlier diagnostic root, and the active temporary output
root. Set `PYTHONDONTWRITEBYTECODE=1`; ignored executable inputs remain a hard
failure. Run new training only in an escalated environment whose complete environment
dictionary matches `feasibility_004`; CPU fallback is forbidden.

The following fields must occur with these exact values in `manifest.json`,
`summary.json`, and the terminal record, and in the metadata of each newly trained
3000-step checkpoint. Reused 004 checkpoints remain immutable and are bound by path
and checksum instead:

```text
artifact_class: non_evidence_feasibility_diagnostic
feasibility_selection_eligible: false
task_010d_authorized: false
```

No function in this task may write a feasibility selection or be accepted by the
selection validator as a feasibility root.

## Interpretation contract

Supported conclusions are limited to the frozen contrasts:

- Named surface effects are supported only by paired surface changes within the
  same operand source.
- Named operand-source differences remain descriptive because seen and held-out
  operands are different examples.
- Array continued-update sensitivity is supported only by small/1500 versus
  small/3000 under identical records, seeds, environments, core blobs, and successful
  step-1500 tensor equality.
- Array model-size sensitivity is supported only by small/1500 versus medium/1500;
  the frozen model-size package changes several architectural dimensions together.

Do not claim that a diagnostic factor is the unique cause, that the model-size
package identifies capacity or optimization as a cause, that a 3000-step result
authorizes a formal budget change, or that any result predicts Phase 8 scientific
task behavior. Do not choose a new task, model, budget, or threshold inside the
diagnostic runner.

## Allowed implementation paths

```text
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

Do not change existing feasibility record generation, model/training primitives,
selection validation, or formal Phase 8 code. The implementation should add the
smallest parameterized diagnostic command and pure analysis helpers needed by this
handoff.

## Required tests

Targeted tests must prove:

1. exact four-cell Named construction, 64 rows per cell, paired surface rendering,
   source identity metadata, byte-length equality, and `template × target-key`
   balance;
2. no train/eval semantic, operand, or record relabeling while constructing the
   Named matrix;
3. exactly three new Array runs, all `small/3000`, with seeds `0,1,2`, and no new
   Hex, Boolean, Named, medium, or 1500-step training;
4. one continuous 3000-step optimizer/batch stream, step-1500 state equality before
   step 1501, mismatch refusal, and snapshotting that leaves RNG/optimizer state
   unchanged;
5. exact frozen core blob IDs, canonical record hashes, complete environment match,
   and checksum reuse of the six `feasibility_004` Array baselines and three Named
   medium checkpoints;
6. row-level metric edge cases and independent reconstruction, including invalid
   decode/JSON, missing EOS, unequal/empty values, duplicate Array items, conditional
   denominators, EOS-inclusive teacher forcing, and JSON `null` handling;
7. refusal on input hash, source, terminal, inventory, lineage, record, frozen
   configuration, CUDA-device, ignored-input, output-root, command, or artifact-role
   mismatch;
8. preflight refusal without a root; complete `DONE`; partial catchable `FAILED`;
   incomplete source/host-failure blocking; continuous retry numbering; exact prior
   diagnostic lineage; and prohibition after a prior `DONE`;
9. exact inventory plus non-selection/non-authorization fields in each required
   root-level record and new checkpoint metadata;
10. existing feasibility selection validation rejects diagnostic roots.

Run targeted tests first, then:

```text
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_lm_bridge.py
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
git diff --check
git status --short
```

Do not launch the diagnostic from an uncommitted or unreviewed implementation.

## Stop boundary and return

Implementation stops after tests and returns to `main` for commit/review. Execution
stops after publishing and validating the immutable diagnostic root. Neither stage
may repair the feasibility protocol, create a selection, or enter Task 010D.

Return using the common format in `phase8/README.md`, including the exact input and
output hashes, cells executed or reused, verification results, resource observation,
protocol deviations, and the next decision reserved for `main`.
