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

## Objective

Separate the remaining failures without tuning on a Phase 8 scientific outcome:

1. For `named_value_json` medium, distinguish sensitivity to held-out prompt
   surfaces from failure to copy held-out operands after target-key selection.
2. For `array_json` small, distinguish a fixed-budget optimization limitation from
   a model-capacity/generalization limitation while retaining the same records,
   templates, item-count balance, tokenizer, scoring, seeds, and optimizer.

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

## Diagnostic B — Array budget × capacity comparison

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

The `small/1500` versus `small/3000` comparison changes only training steps. The
`small/1500` versus `medium/1500` comparison changes only the frozen model-size
package. Do not add an intermediate model, another step count, another seed,
early-stopping selection, or a threshold search after seeing results.

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

## Fixed output and provenance

The initial output root is:

```text
artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

The runner must refuse overwrite and use a same-parent temporary root followed by
an atomic rename. It must publish exactly one `DONE.json` or `FAILED.json` only after
finalizing a checksum-bound manifest. A completed diagnostic uses `DONE` when all
declared cells and reports are present, regardless of their scores. `DONE` is not a
feasibility pass. Any uncaught error produces `FAILED`; a retry uses the next
diagnostic root and preserves the failed root.

The manifest and terminal must bind:

- exact diagnostic implementation commit and clean-source snapshot;
- this handoff path and Git binding;
- exact authorized command and output root;
- `feasibility_004` manifest, summary, and terminal paths/checksums plus its complete
  predecessor closure;
- frozen record identities and checksums for every reused/train/eval diagnostic set;
- all four Named cells for three seeds and all three new Array cells;
- unchanged tokenizer/model/optimizer constants and the explicit diagnostic-only
  `3000`-step value;
- Python, PyTorch, CUDA, GPU, driver, deterministic flags, and wall time;
- every retained generation, checkpoint, metric table, and summary path, size, role,
  and SHA-256, excluding only the manifest and later terminal marker.

Source cleanliness may exclude only the exact checksum-bound roots `001` through
`004` supplied through the accepted input lineage and the active diagnostic root.
Set `PYTHONDONTWRITEBYTECODE=1`; ignored executable inputs remain a hard failure. Run
new training only in an escalated environment where PyTorch reports the same A800
CUDA device class as `feasibility_004`; CPU fallback is forbidden.

Every output must state:

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
- Array step sensitivity is supported only by small/1500 versus small/3000 under
  identical records and seeds.
- Array model-size sensitivity is supported only by small/1500 versus medium/1500;
  the frozen model-size package changes several architectural dimensions together.

Do not claim that a diagnostic factor is the unique cause, that a 3000-step result
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
   record identity, byte-length equality, and `template × target-key` balance;
2. no train/eval semantic, operand, or record relabeling while constructing the
   Named matrix;
3. exactly three new Array runs, all `small/3000`, with seeds `0,1,2`, and no new
   Hex, Boolean, Named, medium, or 1500-step training;
4. exact reuse and checksum binding of the six `feasibility_004` Array baselines and
   three Named medium checkpoints;
5. metric reconstruction from retained raw generations/checkpoints;
6. refusal on input hash, source, terminal, inventory, lineage, record, frozen
   configuration, CUDA-device, ignored-input, output-root, or artifact-role mismatch;
7. atomic `DONE`/`FAILED`, exact inventory, immutable retry numbering, and explicit
   non-selection/non-authorization fields;
8. existing feasibility selection validation rejects diagnostic roots.

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
