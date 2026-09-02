# Capability Certificate Task Handoff 011

## Phase 9 — Decomposed Learned-Behavior Bridge

## 1. Status and authority

This document is a milestone-level research protocol proposal. Its exact baseline is:

```text
main commit: 89c488bd8273f46f064672256293e5ed049089c0
Phase 8 status: independently accepted FAILED/stopped milestone
Phase 9 implementation authority: none before exact-commit review and main acceptance
Phase 9 experiment authority: none
Phase 9 artifact authority: none
```

The proposal must be committed and reviewed by a fresh-context `gpt-5.6-sol`
strict read-only reviewer with `reasoning_effort=high` before it can become the
current Phase 9 scientific contract. Acceptance authorizes only the later creation
and review of bounded subtask handoffs. It does not authorize code changes, data
generation, model selection, training, GPU use, or artifacts.

Phase 2–7 accepted evidence and the accepted Phase 8 failure package remain frozen
historical evidence. Phase 9 is a new milestone, not `feasibility_006`, a retry of
Phase 8, or a continuation of Task 010D.

## 2. Decision and rationale

Phase 8 asked whether a complete learned-model population could proceed directly
from generic sequence-transduction feasibility into primitive, composition, and
certificate experiments. Its required feasibility gate failed: `feasibility_005`
passed `11/24` cells, exactly:

```text
Hex Copy:        5/6
Named-value JSON: 0/6
Boolean JSON:    6/6
Array JSON:      0/6
```

The historical evidence used for this decision is bound exactly as follows:

| Package | Source commit | `manifest.json` SHA-256 | `summary.json` SHA-256 | Terminal SHA-256 |
| --- | --- | --- | --- | --- |
| `feasibility_005` | `2dc8c50fab9ece27a07eb0dc04021b78de52895c` | `51b43e3f3eba98b31c3574a9720c0df3f8e2a72c94d22730d70eb8682e8f516b` | `bfea56b52dfd1ffe094feb0774ce7895f4dd703770451aeda4d899aafdfef785` | `FAILED.json`: `6ac97f081586ca7fb8d7d8297dae4e0c7a3d0314a0567bbc493fce1b5dcc4c08` |
| `feasibility_postmortem_001` | `e3f99fcabbce812408793b62a1cdad536b8b303b` | `1cf55f231cc064ba1e059c084dac38b26888d59907d5469c803cb061dd61d503` | `d8e9d78aa32147d697accdfb92068751724f0de87a0d969ce69f89278201a1cd` | `DONE.json`: `f566a21e2ac2a1dd22dad90c77aaef2ee2331dd8532d824e0083e6f1382caec4` |

The accepted failure proves only that the frozen learner/configuration did not meet
the frozen prerequisite. It does not prove that learned-behavior certificates are
impossible. The completed D3 postmortem is non-evidence, but it may guide engineering
planning. It suggests that the medium learner nearly fit Named-value and Array
training sequences while generalizing poorly, and that teacher-forced and retained
greedy exact counts agreed. The next protocol therefore must separate:

1. whether the learner generalizes each elementary transformation;
2. whether corpus-only intervention creates the intended behavioral states; and
3. whether a learned behavioral population preserves a known certificate structure.

Phase 9 continues the bridge once, through this smaller ladder. It does not increase
the old training budget, enlarge the old model, relax `52/64`, or reuse the exposed
Phase 8 evaluation pack as a new final test merely to obtain a pass.

## 3. Scientific objective

Determine, in a small controlled setting, whether a single frozen learner and
semantic response contract can:

1. generalize factorized text transformations across held-out operands and held-out
   templates;
2. realize a declared population of capability states through training-corpus
   differences alone; and
3. preserve the exact fixed and adaptive certificate structure of a small accepted
   Phase 7 block world when observations come from trained checkpoints rather than a
   deterministic simulator.

The central Phase 9 claim, if all gates pass, is limited to:

> Under the frozen synthetic tasks, learner, training protocol, state intervention,
> semantic evaluator, and seed set, learned checkpoint behavior preserved one known
> finite capability-state and certificate structure.

Phase 9 does not test held-out composition. That becomes a separately proposed later
milestone only if Phase 9 succeeds.

## 4. Questions deliberately separated

### 4.1 Learnability

Can the selected learner execute the required elementary transformations on new
values, new surface forms, and their combination?

### 4.2 State realization

Can state-specific training corpora create separately trained checkpoints whose
primitive success/refusal behavior matches the declared state matrix without a state
token or inference-time side channel?

### 4.3 Certificate preservation

When a declared block world has a known compressed certificate, does the thresholded
checkpoint response matrix reproduce the oracle matrix and therefore its fixed and
adaptive certificate metrics?

Failure at an earlier question stops the milestone. Later analysis must never be used
to disguise a failed prerequisite.

## 5. Non-goals

Do not add or run:

- Phase 8 `feasibility_006`, Task 010D, or any Phase 8 formal shard;
- post-outcome changes to Phase 8 artifacts, reports, thresholds, or interpretation;
- open or pretrained language models;
- large-scale pretraining, real-world benchmarks, or automatic graph discovery;
- an architecture or hyperparameter sweep against the sealed final evaluation pack;
- p-values or confidence intervals treating prompts, states, cells, or three seeds as
  independent draws from a real model population;
- hidden state/persona tokens or a single prompt-conditioned model standing in for a
  population of checkpoints;
- held-out composition or randomized composition controls;
- a new adversarial in-process verifier, source-byte hash hierarchy, plugin system,
  generalized training framework, distributed runner, or experiment tracker.

## 6. Units and claim boundary

- One separately trained checkpoint is one evaluated system.
- One seed is one stability replication, not a sample from a deployment population.
- Sixty-four prompts define one task response cell; they are repeated measurements,
  not 64 model replicates.
- Every matrix and certificate is computed within one complete seed-specific model
  population. Rows from different seeds must never be pooled.
- The declared capability state is a controlled training intervention. It is not a
  claim that an unconstrained model has a naturally discrete latent state.
- Exact agreement with a declared matrix validates this experimental bridge only; it
  does not prove that real LLM capabilities have the same structure.

## 7. Ordered stages and gates

```text
011A factorized semantics and split contract
  -> independent review
011B development-only learner qualification
  -> independent artifact review and main decision
  -> at most one reviewed learner-revision proposal if required
011C sealed generalization gate
  -> independent artifact review and main decision
011D primitive-state realization
  -> independent artifact review and main decision
011E block-world certificate preservation
  -> final scientific review
```

No stage inherits execution authority from this document or from a previous stage.
Each stage needs a separate exact handoff, accepted implementation commit, explicit
main execution authorization, immutable output root, and independent review
proportional to its risk.

## 8. Stage 011A — Factorized semantics and splits

011A defines data and evaluators only. It must not train a model.

### 8.1 Factorized task families

Use exactly six task families:

1. `copy_one`: copy one supplied string without JSON serialization;
2. `select_named`: select a named field and return its raw string value;
3. `serialize_string`: JSON-quote and escape one explicitly supplied string, with no
   field selection;
4. `array_fixed_two`: serialize exactly two supplied strings in their given order;
5. `array_variable`: serialize one to four supplied strings in their given order;
6. `compare_bool`: compare two explicitly supplied bounded non-negative decimal
   integers and return canonical JSON `true` exactly when the left integer is smaller
   than the right integer, otherwise canonical JSON `false`.

These families isolate copying, selection, string serialization, multi-item copying,
length/termination control, and a simple comparison/Boolean-serialization control
rather than an opaque label lookup. The `compare_bool` operands form an
execution-relevant domain large enough for disjoint operand and joint holdouts. A
family must not silently add another transformation through its templates or
evaluator. 011A must freeze the integer bounds and representation for `compare_bool`,
cover `<`, `=`, and `>` cases, balance true/false outcomes in every split, and prevent
leading-zero or formatting shortcuts unless the exact frozen representation
explicitly permits them.

### 8.2 Three generalization axes

011A must assign two separate template identities. `template_family_id` identifies a
structural/surface-form family; `template_id` identifies one exact concrete template
definition. These identities are frozen in 011A. Every task family then has three
evaluation axes:

1. `operand_holdout`: reuse exact training `template_id` values and use operand
   tuples absent from training and every other evaluation axis;
2. `template_holdout`: use `template_family_id` and exact `template_id` values absent
   from training, while intentionally reusing operand tuples from the training
   operand pool so only the template factor is held out; the complete records and
   rendered prompts must nevertheless be new; and
3. `joint_holdout`: use `template_family_id` and exact `template_id` values absent
   from training and from the template-only axis, plus operand tuples absent from
   training and every other evaluation axis.

For both `template_holdout` and `joint_holdout`, the development and sealed-final
`template_family_id` sets are pairwise disjoint, and their exact `template_id` sets
are pairwise disjoint. Both roles remain disjoint from training as required above.
Their new operand sets are also mutually disjoint wherever operands are a held-out
factor. All payload IDs, complete records, and rendered prompt bytes are globally
disjoint across the train, development, and sealed-final splits. Every role uses the
same declared alphabet and grammar and the same per-task-family length distribution.
The old Phase 8 train/evaluation records may be inspected as development history but
must not occur in a Phase 9 final pack.

### 8.3 Semantic and canonical evaluation

For each generated response, retain two separate results:

- `semantic_success`: the output parses under the family grammar and its decoded
  value exactly equals the oracle value;
- `canonical_exact`: the response bytes exactly equal the canonical target bytes.

For raw-string families, semantic equality is exact string equality. For JSON
families, semantic equality is typed JSON-value equality; malformed JSON fails.
Parsing must not coerce booleans, numbers, strings, or arrays across types, ignore
extra fields/items, reorder arrays, repair malformed output, or accept trailing
non-whitespace content.

The Phase 9 behavioral response bit is based on `semantic_success`. Canonical exact
is a required secondary metric and defines the separate serialization capability; it
must not replace or retroactively reinterpret the Phase 8 byte-exact gate.

### 8.4 Independent oracles

The production generator/evaluator and the test oracle must not call each other.
011A must include literal edge cases for JSON escaping, Unicode/UTF-8 bytes, empty and
maximum-length allowed strings, repeated array values, one-to-four item lengths,
named distractors, bounded integer comparisons covering `<`, `=`, and `>`, balanced
true/false outcomes, canonical Boolean serialization, malformed output, early/late
EOS, and trailing content.

## 9. Stage 011B — Development-only learner qualification

011B decides whether the current model family is a usable measuring instrument. It
does not produce scientific evidence about certificates.

### 9.1 Baseline

Start with the accepted Phase 8 medium tied byte-level causal Transformer and its
tokenizer, response-only loss, optimizer semantics, and deterministic backend. Use a
new Phase 9 training set of `512` records per family, `1500` training steps, batch
size `64`, and seeds `0,1,2` as the initial baseline. Train one checkpoint per
factorized task family and seed: exactly `6 * 3 = 18` baseline runs.

Development evaluation uses all three axes and retains raw outputs, semantic counts,
canonical counts, token metrics, EOS/length metrics, loss, configuration, and timing.
It may use no future 011C record.

### 9.2 One-revision limit

If every baseline development cell passes, the baseline becomes the selected learner
and no alternative may run. If any cell fails, main has exactly two choices:

1. stop the Toy LM route; or
2. authorize one top-level, independently reviewed learner-revision proposal tied to
   the diagnosed development failure.

Only one revised learner/configuration may be executed, again for exactly 18 runs.
The revision may change the learner architecture, data volume, or training budget,
but must freeze one complete configuration before execution and must not change task
semantics, split axes, the `52/64` threshold, or any final record. It is not a grid,
sweep, or best-of-seed selection. If the revised configuration fails any development
cell, Phase 9 stops and the current from-scratch Toy LM route is rejected.

Choosing or designing the revision is top-level architecture work and therefore
requires the user to switch the main session manually to `gpt-5.6-sol` with
`reasoning_effort=xhigh` before the proposal is written.

### 9.3 Development gate

For every family, axis, and seed independently:

```text
semantic_success_count >= 52 of 64
```

Do not average across axes, families, model seeds, or prompts to hide a failing cell.
Canonical exact counts are reported but are not the 011B gate.

## 10. Stage 011C — Sealed generalization gate

011C is a one-shot qualification of the single selected configuration.

- Freeze the selected source commit, model/training configuration, and generator
  contract before generating or reading any final record.
- Generate a new immutable final pack with the axis-specific template and operand
  reuse/novelty required by sections 8.2 and 14, globally new payload IDs, complete
  records, and rendered prompt bytes. It has `64` records for every family/axis cell.
- Run exactly the selected `6 * 3 = 18` checkpoints/configurations. Do not compare
  another size or configuration.
- Require `semantic_success_count >= 52/64` for every family/axis/seed cell.
- Report canonical exact, parsing, content, length, EOS, and token metrics without
  using them to replace the primary gate.
- A failed cell stops Phase 9 before state realization. There is no 011C retry, new
  seed, threshold change, or final-pack revision.

Passing 011C proves only that the selected learner is adequate for these elementary
synthetic transformations. It is not evidence of a capability certificate.

## 11. Stage 011D — Primitive-state realization

011D tests whether training-corpus intervention creates the declared behavioral
population before any compression claim.

### 11.1 Declared population

Use four executable primitive capabilities and their complete 16-state powerset.
Their exact programs, contexts, templates, positive answers, and `unable` response
must be frozen in the 011D handoff and independently checked against the accepted DSL
executor. Use one separately instantiated checkpoint for every `(seed,state)` pair:

```text
3 seeds * 16 states = 48 checkpoints
```

Within each seed-specific population, all state checkpoints use byte-identical
initial model weights; the same model and optimizer configuration and optimizer
schedule; the same training record keys, prompt multiset, per-task record counts,
logical record order, and minibatch-index schedule; and the same non-target record
fields. For a fixed logical record, the only state-dependent bytes are the response
target: the canonical task answer when the corresponding declared capability is
present, or exact canonical `unable` when it is absent. A seed may change the frozen
initialization and schedule; state may not. A state ID, persona, capability list,
filename fragment, or other state signal must not appear in a prompt or inference
input.

### 11.2 Positive and negative behavior

Each checkpoint receives the same immutable evaluation pack with `64` prompts per
primitive, frozen and isolated under the shared state-stage rule in section 14. For a
primitive present in the declared state, require at least `52/64` semantic successes.
For an absent primitive, require at least `52/64` exact canonical `unable` responses.
Retain all semantic-success, canonical-refusal, malformed, and unrelated-answer
counts; do not convert malformed or unrelated answers into refusals.

For every seed independently, the thresholded `16 x 4` behavioral matrix must equal
the declared matrix exactly and pass an independent full-information identifiability
audit. Any mismatch or collision stops Phase 9 before 011E.

The expected exact fixed certificate contains all four primitives. This is a state-
realization check, not a compression result.

## 12. Stage 011E — Block-world certificate preservation

011E tests the first learned certificate structure with a known compressed oracle.

### 12.1 Frozen worlds

Use exactly one accepted Phase 7 parameter cell for each family:

```text
independent block: B=3, uniform block size s=2
prefix block:      B=3, uniform block size s=2
```

The future 011E handoff must choose exactly one task contract from each of the six
accepted 011A families and group the resulting six tasks into three two-task blocks.
All six must be genuinely distinct model-facing probes: they have distinct executable
transformation/evaluator contracts, pairwise-disjoint template sets and rendered
evaluation prompt bytes, and no alias or byte-identical program/rendering is accepted
as a separate task. At least one frozen, independently checked witness per two-task
block must demonstrate that its two executable task mappings are semantically
distinct. The two distinct tasks in a block share one declared capability toggle and
therefore one statewise oracle response column. The text-task mapping must preserve
the accepted block closure exactly and must be verified by independent exhaustive
enumeration before training.

Within each seed-specific population and each block-world family, all state
checkpoints use byte-identical initial model weights; the same model and optimizer
configuration and optimizer schedule; the same training record keys, prompt
multiset, per-task record counts, logical record order, and minibatch-index schedule;
and the same non-target record fields. For a fixed logical record, the only
state-dependent bytes are the response target: the canonical task answer when its
block capability is present, or exact canonical `unable` when it is absent. Seed and
block-world family may change the frozen initialization and schedule; state may not.
No state ID or other inference-time side channel is allowed.

Train a separate checkpoint for every declared state and seed using the 011C-selected
learner:

```text
independent block: 8 states * 3 seeds
prefix block:      4 states * 3 seeds
total:             36 checkpoints
```

No new learner/configuration choice is allowed.

### 12.2 Required results

Each family uses its immutable evaluation pack frozen and isolated under the shared
state-stage rule in section 14.

For each family and seed independently:

1. retain the complete `state x task x 64 prompts` raw output tensor;
2. require positive success and negative refusal gates identical to 011D;
3. compare the complete behavioral matrix with the independent oracle matrix;
4. audit identifiability before solving any certificate;
5. run the exact fixed solver and independent validator;
6. run the canonical exact-integer entropy adaptive solver and independent tree
   validator;
7. reconstruct fixed separation and adaptive leaf/depth metrics independently from
   the serialized outputs.

The oracle expectations are:

| Family | States | Tasks | Exact fixed size | Entropy average depth | Entropy worst depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent block | 8 | 6 | 3 | 3 | 3 |
| prefix block | 4 | 6 | 3 | 2 | 2 |

All three seed-specific populations must reproduce the oracle matrix and metrics.
No seed pooling, majority vote, nearest-state mapping, alternative threshold, or
certificate selected for better agreement is allowed.

## 13. Statistical and reporting semantics

- Report every family/axis/seed cell, not only aggregates.
- Report mean/min/max across the three fixed seeds only as descriptive stability
  summaries after displaying each seed.
- Do not compute inferential p-values or confidence intervals from prompt rows.
- A pass is a deterministic conjunction of all predeclared cells, not a statistical
  significance claim.
- Training and development results may motivate the one permitted learner revision,
  but they are not certificate evidence.
- 011C is learner qualification; 011D is state-realization evidence; 011E is the only
  Phase 9 certificate-preservation evidence.
- A failed gate is a valid negative result when implementation and provenance pass.

## 14. Leakage and split rules

Every stage must validate, before training, the exact required and forbidden overlaps
for its split contract. For 011A, 011B, and 011C this includes:

- globally disjoint train/development/sealed-final payload IDs, complete records, and
  rendered prompt bytes;
- exact training `template_id` reuse on `operand_holdout`, with operand tuples absent
  from training and every other evaluation axis;
- training-operand-pool reuse on `template_holdout`, with both `template_family_id`
  and exact `template_id` absent from training;
- on `joint_holdout`, both template identities absent from training and the
  template-only axis, and operand tuples absent from training and every other
  evaluation axis;
- for both `template_holdout` and `joint_holdout`, pairwise-disjoint development and
  sealed-final `template_family_id` sets and pairwise-disjoint development and
  sealed-final exact `template_id` sets, with both roles disjoint from training;
- mutually disjoint development and sealed-final new operand sets wherever operands
  are held out;
- the same declared grammar, alphabet, and per-task-family length distribution
  across roles;
- no state, seed, split, answer, task-availability, or target marker in prompts;
- no final-pack generation, loading, metric computation, or inspection before the
  selected configuration is frozen;
- no Phase 8 evaluation record in the Phase 9 final pack;
- exact oracle agreement between stored targets and executable task semantics.

The split validator must return concrete colliding records on failure. Empty splits,
silent filtering, truncation, fallback tokenization, or replacement sampling fail
closed.

For both 011D and 011E, before generating any state-training corpus or training any
state checkpoint, freeze one immutable held-out evaluation pack for that stage and
bind it in the stage handoff and artifact manifest. Use the same model-facing prompt
pack for every state and seed in the stage; if the two 011E families require separate
packs, use the same family-specific pack for every state and seed within that family.
Evaluation `template_family_id` values, exact `template_id` values, payload IDs,
execution-relevant operand tuples, complete records, and rendered prompt bytes must
be disjoint from every state-training corpus in that stage. Stored task semantics may
supply the positive oracle answer; the declared state determines whether scoring
expects that answer or exact canonical `unable`. No state signal enters the prompt.
Neither an evaluation pack nor any state corpus or configuration may be revised,
replaced, filtered, or regenerated after any checkpoint output or metric from that
stage is inspected. Such a failure is terminal for the stage.

## 15. Minimal implementation and provenance policy

Phase 9 must not extend the monolithic Phase 8 feasibility runner. Use a small
`phase9/` handoff directory and narrowly scoped modules only after this protocol is
accepted. Prefer one data/evaluator module, one bounded runner, and one independent
validator per active stage; do not retain executable writers for completed stages.

Git is the source of truth for versioned code, documents, templates, configurations,
and launchers. Experiment manifests bind the exact clean source commit, exact command,
configuration, split/artifact checksums, runtime versions, device, seeds, and complete
file inventory. Do not maintain parallel hashes for tracked source files.

The threat model is reproducible local research, not a hostile process with permission
to rewrite Git or Python objects during execution. Do not add inline verifier source,
transitive object-identity snapshots, or source-byte authentication beyond the exact
Git commit. Retain fail-closed schema validation, no-overwrite output roots, temporary
sibling publication, terminal markers, and evidence checksums.

Formal runs require a clean committed tree, a fixed new output root under
`artifacts/phase9_decomposed_bridge/`, explicit main authorization, bounded log/status
inspection, and independent artifact review. Never overwrite or select from a failed
root. New scientific semantics require a new reviewed protocol, not an implementation
patch.

## 16. Model routing and mutation lease

Follow repository `AGENTS.md`:

- task execution, documentation, repair, and experiment operation use a fresh-context
  `gpt-5.6-sol` subagent with `reasoning_effort=medium` and `fork_turns=none`;
- independent strict read-only review uses a separate fresh-context `gpt-5.6-sol`
  subagent with `reasoning_effort=high` and `fork_turns=none`;
- top-level architecture design requires the user to switch the main session manually
  to `gpt-5.6-sol` with `reasoning_effort=xhigh` before design work begins.

Main owns scientific choices, stage authorization, acceptance, review resolution,
and commits. Only one delegated writer or experiment operator may hold the mutation
lease. Main stays read-only while that lease is active.

## 17. Stop conditions and route decisions

### 17.1 Stop the current Toy LM route when

- the baseline and sole allowed revised learner both fail 011B;
- the selected learner fails any one-shot 011C cell;
- 011D cannot realize every declared state independently for all seeds;
- leakage, oracle, provenance, or exact-output retention cannot be validated;
- a pass would require lowering a threshold, changing a final pack, pooling seeds,
  adding configurations, or reusing Phase 8 final records.

After such a stop, do not add another from-scratch byte-level learner retry. A later
proposal may compare a pretrained small model with state-specific adapters, or return
to non-neural graph/certificate research, but neither path inherits Phase 9 authority.

### 17.2 Proceed beyond Phase 9 only when

011C, 011D, and 011E artifacts and reports each pass independent review and the final
Phase 9 scientific review accepts the complete package. Only then may main propose a
new held-out-composition milestone. Phase 9 never directly authorizes it.

## 18. Milestone acceptance criteria

Phase 9 is scientifically complete when either:

1. a predeclared stop gate fails and a reviewed report records the valid negative
   result without later-stage execution; or
2. all stages pass and independent final review accepts the limited learned-certificate
   preservation claim.

For a positive Phase 9 result, all of the following are mandatory:

1. independently reviewed task semantics and split validators;
2. one frozen learner selected without final-pack outcomes;
3. all 54 sealed-generalization cells pass (`6 families * 3 axes * 3 seeds`);
4. all three primitive-state matrices exactly equal the 16-state oracle;
5. all six block-world matrices (`2 families * 3 seeds`) exactly equal their oracles;
6. every matrix is independently identifiable;
7. exact fixed and entropy-adaptive certificate outputs match the independently
   reconstructed Phase 7 expectations;
8. manifests, raw outputs, checkpoints, terminal records, inventories, and checksums
   bind the exact clean source and contain no missing or overwritten evidence;
9. reports preserve the claim boundary and seed/prompt semantics; and
10. a fresh-context final scientific review returns `ACCEPT` for exact commits and
    artifact hashes.

## 19. Immediate authorized next step after proposal acceptance

If this exact proposal commit is independently accepted, main may create only:

```text
phase9/README.md
phase9/Task_011A_Factorized_Semantics_and_Splits.md
```

Those documents must freeze 011A's exact programs, grammars, templates, generator
domains, split IDs, semantic evaluators, independent oracles, allowed paths, tests,
and return format. Proposal acceptance does not authorize their implementation or
any data generation.

## 20. Common return format

Every later Phase 9 subtask returns:

```text
Scope completed:
Prerequisite commit used:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests or validators run and exact results:
New artifacts and hashes:
Gate result:
Unresolved decisions:
Protocol deviations:
Resource observations:
Independent review required:
Recommended next authorized step:
```

`Protocol deviations` must be `none` for acceptance. A blocked stage returns the
smallest concrete problem, reproduction, evidence, likely cause, classification,
decision required from main, and work explicitly not attempted.
