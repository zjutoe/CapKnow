# Capability Certificate Task Handoff 010: Phase 8 Toy Language Model Bridge

## 1. Milestone Boundary

Branch:

```text
milestone/phase8-toy-lm-bridge
```

Accepted baseline:

```text
main commit: 6d7012a905f8814e0788095bb3eee54a83616cc3
```

Phase 2-7 is accepted evidence. Do not edit or regenerate its accepted result
artifacts, manifests, phase-specific reports, revalidation scripts, or experiment
scripts in this milestone. `Capability_Certificate_Research_Progress_Summary.md` is
the living cross-milestone summary and may be updated only after Phase 8 evidence is
accepted, while retaining accurate Phase 2-7 conclusions and provenance.

The earlier untracked draft named
`Capability_Certificate_Task_Handoff_007_Phase7_Toy_Language_Model_Bridge.md` was
renumbered because Phase 7 Structural Compressibility is already complete and owns
`phase7_report.md`. This milestone owns `phase8_report.md` and a distinct artifact
root.

No Phase 8 implementation, model training, or experiment is authorized until this
handoff passes a fresh-context milestone-start scientific protocol review. Ordinary
read-only inspection of accepted APIs and the local runtime is allowed.

## 2. Scientific Objective

Determine, in a small controlled setting, whether separately trained causal
language models can acquire behavioral response structure from natural-language
corpora generated from executable DSL programs, and whether the resulting model
population admits a capability certificate corresponding to the ground-truth DSL
world.

The milestone separates four questions:

1. Can a toy model learn held-out primitive task instances from state-specific text
   demonstrations?
2. Can it execute a held-out composition when its training corpus is consistent with
   the required primitive capabilities?
3. Does the binary behavioral response matrix remain identifiable, and if so how do
   its fixed and adaptive certificates compare with the ground-truth certificate?
4. Does coherent composition supervision outperform an aggregate-token-histogram-
   matched randomized composition control?

The unit being certified is a separately trained checkpoint. A contextual agent
ID or state token is forbidden: a single model conditioned on multiple personas
would certify prompt-conditioned personas rather than model behavior.

## 3. Non-goals

Do not add or run:

- open or pretrained language models;
- large-scale pretraining or real-world benchmarks;
- architecture search or post-outcome hyperparameter tuning;
- automatic capability-graph discovery;
- graph neural networks;
- unequal state priors or unequal probe costs;
- causal claims about structured language in general;
- p-values or confidence intervals based on probe items, states, grid cells, or the
  three training seeds as if they were independent samples from a real population;
- changes to accepted DSL, knowledge-space, identifiability, certificate, adaptive,
  or probabilistic semantics merely to make a Phase 8 result pass.

No positive scientific hypothesis is an engineering acceptance gate. Failure to
learn a composition, loss of identifiability, or absence of a behavioral certificate
is a valid Phase 8 outcome when the protocol and implementation gates pass.

## 4. Evaluated-System and Response Contract

### 4.1 One checkpoint per declared state

Use exactly four primitive capabilities in this order:

```text
MEMORY
SEARCH
FILTER
CONDITION
```

The declared state population is the complete powerset of these primitives, in
canonical bit-mask order from `0` through `15`, with bit positions assigned by the
primitive order above (`MEMORY` is bit 0 and `CONDITION` is bit 3). Each state is
assigned its own corpus and separately instantiated and trained checkpoint. All 16
checkpoints within a training seed intentionally use the same initial weights; they
are separate evaluated systems, not independently initialized statistical
replicates. For every experiment condition and training seed, train exactly 16
checkpoints, one per state.

The state must not be supplied to the model as a token, embedding, prompt, filename
fragment inside a prompt, or inference-time side channel. It affects the model only
through which task outcomes appear in that checkpoint's training corpus.

For a condition `c`, seed `s`, declared state `K`, and probe task `q`, let the trained
checkpoint be `M[c,s,K]`. One complete `(c,s)` population contains all 16 checkpoints
and produces one behavioral response matrix. Never pool rows from different seeds
to manufacture a larger knowledge space.

### 4.2 Ground-truth success

For a DSL program `p_q`, input context `x`, and primitive state `K`, define

```text
y_gt(K, q, x) = 1
```

exactly when the accepted DSL executor completes `p_q` with every primitive in `K`
marked available. `MissingCapabilityError` gives `0`; malformed programs or contexts
are protocol failures and must propagate rather than being converted to failure
responses.

The counterfactual correct answer for a probe is obtained by executing the same
program and input with every primitive required by that program available. All
generated contexts must be prevalidated so that this full-capability execution
succeeds and serializes to one canonical answer.

Apply exactly one task-specific answer projection before canonical compact-JSON
serialization: the primitive `MEMORY` task returns the `value` field added to its
execution-result mapping; every other primitive or composite task returns the full
`ExecutionResult.value`. The projection is part of the frozen task contract and must
not inspect a model state. Every projected evaluation answer must encode to at most
63 tokenizer tokens so EOS fits within the 64-token generation limit.

### 4.3 Model task success

At evaluation time, a generated response counts as task success when, after removing
only the generated EOS token, it exactly equals the canonical counterfactual answer.
No case folding, whitespace normalization, substring matching, numeric tolerance, or
manual adjudication is allowed.

This definition is independent of the assigned state. A model assigned a state that
lacks a required primitive can still produce a behavioral false positive if it gives
the correct answer. Producing the training refusal response is never counted as task
success.

Each probe task has exactly 64 evaluation instances. Its behavioral response bit is

```text
y_lm(c, s, K, q) = 1 if exact_success_count >= 52, else 0
```

which is the integer definition of accuracy at least `0.8125`. The threshold is fixed
before training and must not be calibrated on Phase 8 outcomes. Raw counts and
accuracies must always accompany the bit. As a predeclared sensitivity analysis,
also reconstruct response matrices at thresholds `32/64`, `48/64`, and `64/64`;
these matrices are descriptive and cannot replace the primary `52/64` matrix.

### 4.4 Evaluator isolation

The behavioral evaluator receives only:

- a checkpoint and its architecture/tokenizer configuration;
- rendered natural-language prompts;
- opaque probe keys;
- canonical answer strings;
- decoding limits.

It must not import the capability graph, state object, program object, primitive IDs,
or training metadata. It emits immutable per-prompt generations and exact-success
bits. Only after those outputs are frozen may `certificate_eval.py` join opaque probe
keys to the separately stored task order and ground-truth matrix.

## 5. Frozen DSL World

### 5.1 Probe task universe

Use these eight internal task IDs in exactly this order:

| task ID | DSL program | required primitives | role |
| --- | --- | --- | --- |
| `MEMORY` | primitive memory lookup | `MEMORY` | primitive train/probe |
| `SEARCH` | primitive membership search | `SEARCH` | primitive train/probe |
| `FILTER` | primitive equality filter | `FILTER` | primitive train/probe |
| `CONDITION` | primitive Boolean condition | `CONDITION` | primitive train/probe |
| `MEMORY_FILTER` | `MEMORY -> FILTER` | `MEMORY,FILTER` | seen composition |
| `FILTER_CONDITION` | `FILTER -> CONDITION` | `FILTER,CONDITION` | seen composition |
| `SEARCH_CONDITION` | `SEARCH -> CONDITION` | `SEARCH,CONDITION` | seen composition |
| `MEMORY_SEARCH` | `MEMORY -> SEARCH` | `MEMORY,SEARCH` | held-out composition |

Programs must be built from the accepted `PrimitiveNode` and `SequenceNode` APIs.
Every program/context/output contract must have independent exact-output tests. Do
not add a synthetic composition label to a model's state; composite success is
defined only by executable primitive requirements.

Freeze these task-specific context schemas:

- primitive `MEMORY`: `{"memory": {k: v}, "key": k}`; projection returns `v`;
- primitive `SEARCH` and `FILTER`:
  `{"items": [item, ...], "target": t}`;
- primitive `CONDITION`: `{"condition": b}` where `b` is Boolean;
- `MEMORY_FILTER` and `MEMORY_SEARCH`:
  `{"memory": {k: v}, "key": k, "items": [item, ...]}`;
- `FILTER_CONDITION` and `SEARCH_CONDITION`:
  `{"items": [item, ...], "target": t}`.

Here `k`, `v`, `t`, and every item are strings, while `b` is Boolean. These are exact
top-level key sets, `memory` contains exactly the one shown entry, and extra fields
are forbidden.

The sequence schemas are semantically binding. Because accepted `MEMORY` adds
`value` and sets `target` only when it is absent, forbidding an initial `target` and
`value` ensures the downstream `FILTER` or `SEARCH` consumes the retrieved memory
value. For both memory sequences, dependency tests must keep every other context
field fixed and change only `memory[key]`; the final projected answer must change.
One test value must be present in `items` and the other absent. A context that already
contains `target` or `value` must be rejected by the Phase 8 generator rather than
accepted as another template variant.

### 5.2 Ground-truth certificate oracle

The 16 primitive states and eight program probes form the ground-truth world. Before
training, independently gate all of the following:

- full-information identifiability;
- exact certificate validity;
- exact fixed certificate size `4`;
- the unique minimum fixed certificate is
  `MEMORY, SEARCH, FILTER, CONDITION`;
- each primitive has the empty-state/singleton-state single-coordinate witness;
- no composite program appears in a singleton witness for a missing primitive;
- the accepted exact solver result agrees with an independent enumeration of all
  `2^8` task subsets.

The theorem-backed ground-truth result is a protocol gate, not a neural result.

## 6. DSL-to-Text and Split Contract

### 6.1 Rendered records

Every training record is a decoder-only prompt/response sequence with explicit BOS,
prompt/response separator, and EOS tokens. The loss is computed only over response
tokens. Every encoded record must fit the 256-token context without truncation; an
oversized record is a generator error. The model-facing text may describe operation
semantics naturally, but it must not contain:

- any internal task ID as an uppercase symbolic token;
- a serialized DSL node or capability-state bit vector;
- graph edges, prerequisite notation, rule tables, or expressions such as
  `A+B->C`;
- the words `primitive`, `capability`, `certificate`, or `knowledge state` as
  structural metadata;
- hidden state IDs, model IDs, or seed IDs.

Metadata may contain internal IDs but must be stored separately from model-facing
text. The leakage oracle scans decoded text, token sequences, and prompt/response
fields before training.

For every evaluation prompt, the encoded prefix consisting of BOS, rendered prompt,
and SEP must satisfy

```text
encoded_evaluation_prefix_length + 64 <= 256
```

so the full frozen generation allowance fits the positional embedding table. Both
training and evaluation reject overflow before model execution. Truncation, left
truncation, rolling windows, and sliding-window generation are forbidden.

Answers use canonical compact JSON values encoded as UTF-8. When a program is not
executable in the assigned state, its training response is exactly:

```text
unable
```

This response teaches state-specific absence but is never a successful evaluation
answer.

### 6.2 Templates and payloads

Each primitive and seen-composition task must have disjoint train and evaluation
template sets. Templates are source-controlled and assigned to one split; formatting
or paraphrase identity may not cross splits.

Payloads are generated from disjoint integer seed namespaces:

```text
training payload seeds:   100000..199999
evaluation payload seeds: 300000..399999
```

For task index `i` in the frozen task order and zero-based record index `j`, use
training payload seed `100000 + 1000*i + j` and evaluation payload seed
`300000 + 1000*i + j`. Base corpora use `j=0..127`, large corpora use `j=0..511`,
and evaluation uses `j=0..63`. The base corpus is therefore an exact prefix of the
large corpus for the same task and state.

The generator must store canonical program dictionaries, template IDs, payload IDs,
canonical compact-JSON input contexts, normalized payload-value tuples, rendered
prompts, answers, and split names in metadata. Payload seeds are metadata only and
must never be rendered. Derive each string operand as the first 16 lowercase
hexadecimal characters of

```text
sha256("phase8-payload|" + task_id + "|" + payload_seed + "|" + field_name)
```

where each component is encoded as its canonical ASCII decimal or text form. This
gives train and evaluation operands the same surface format without exposing their
split-partitioned seed ranges. Hash collisions are not silently resampled: the split
oracle below must reject them.

Normalize payload values without relying on mapping insertion order as follows:

```text
MEMORY:                    (key, memory[key])
SEARCH or FILTER:          (tuple(items), target)
CONDITION:                 (condition,)
MEMORY_FILTER or
MEMORY_SEARCH:             (key, memory[key], tuple(items))
FILTER_CONDITION or
SEARCH_CONDITION:          (tuple(items), target)
```

These tuples contain only operands read by the accepted executor. Every listed value
must also be rendered in the natural-language prompt. Mappings are serialized with
sorted keys and compact JSON separators for byte-level context comparison.

For every task except primitive `CONDITION`, gate empty intersections between
training and evaluation sets of canonical input-context bytes and normalized
execution-relevant payload tuples, in addition to exact non-overlap of prompt text,
template ID, payload ID, and the tuple `(program_dict, template_id, payload_id)`. Any
collision is a generator error; do not resample it silently. Tests must inject a
duplicate canonical context and a duplicate execution-relevant payload tuple and
require both to be rejected.

Primitive `CONDITION` has only the two execution-relevant inputs `true` and `false`
under the accepted DSL and both are required in train and evaluation. Semantic
payload overlap is therefore unavoidable and explicitly allowed only for that task;
its evaluation measures held-out template rendering, not held-out operand
generalization. Gate that both splits have the exact semantic set `{false,true}` and
that their template IDs and complete rendered prompts remain disjoint. Do not add an
ignored nonce and call it semantic separation.

All Boolean-valued task families must have exactly 32 `true` and 32 `false`
counterfactual answers in the 64 evaluation instances. Non-Boolean answer families
must have 64 distinct counterfactual answers. These are evaluator difficulty and
guessing diagnostics, not evidence of natural-language competence.

Generate exactly one immutable evaluation probe pack containing the same ordered
`8 * 64 = 512` prompts, opaque probe keys, and canonical answers for every condition,
state, seed, model size, corpus size, and checkpoint. Generate and checksum it once;
all evaluation records must reference that exact checksum. Per-condition or
per-state evaluation rendering is forbidden.

Primitive tasks use one shared source-controlled neutral evaluation-template set.
For each of the four composition tasks, evaluation indices `0..31` use held-out
explicit-stepwise paraphrases and indices `32..63` use held-out indirect end-to-end
paraphrases. Both evaluation style sets are disjoint from every A and B training
template. Within each Boolean composition task and each 32-instance style block,
exactly 16 answers are `true` and 16 are `false`. The non-Boolean
`MEMORY_FILTER` task has 32 distinct answers per style block. Report composition
mastery separately for the explicit and indirect evaluation halves as well as for
their fixed equal-weight combination; the combined value is the primary value.

### 6.3 Held-out composition

`MEMORY_SEARCH` is absent from every training corpus in every state, condition, seed,
and scale cell. Neither its program dictionary nor any template assigned to it may
occur in training. It appears only in the evaluation probe pack.

Primitive `MEMORY` and `SEARCH` examples remain present in every training corpus;
their outcome is the correct answer or `unable` according to the assigned state.
This split tests behavioral composition from primitive demonstrations and other seen
composition formats, not zero-shot recovery of unseen primitive semantics.

## 7. Corpus Conditions

All conditions use the same state population, task payloads, number of records,
training steps, batch schedule, tokenizer, optimizer, and model initialization seed
for paired `(seed,state)` runs. Record ordering is deterministically derived from the
training seed after a condition's records are built.

All conditions also use the single byte-identical evaluation probe pack frozen in
Section 6.2. Evaluation style is not a condition-specific choice.

Curriculum is fixed rather than treated as another intervention: all conditions use
the same task-stratified record layout followed by the same deterministic shuffle
schedule. This milestone varies corpus surface/composition consistency, corpus size,
and model size only.

### 7.1 Condition A: explicit structured composition

Primitive records use natural single-operation prompts. The three seen-composition
tasks use explicit stepwise natural-language prompts that describe the two operations
in execution order without naming either capability or stating a graph rule. Outcomes
are exactly consistent with the assigned primitive state and the DSL oracle.

### 7.2 Condition B: indirect structured composition

The outcome matrix is identical to Condition A. Primitive records are identical.
Seen-composition prompts describe only the end-to-end request and final outcome; they
do not include explicit step boundaries or an intermediate result. Training schedule
and record counts match Condition A, but surface token statistics are not claimed to
match because removal of stepwise language is the intended intervention.

### 7.3 Condition C: randomized composition control

Condition C starts from the exact Condition A primitive and composite prompts.
Primitive records remain byte-identical, including outcomes, for every state.

Let rows be the 16 states and columns be all seen-composition training records. Let
`L[r,c]` be the structured executable/non-executable outcome bit. Randomize `L` using
degree-preserving `2 x 2` switches. Initialize one `random.Random(700000 + seed)`.
Repeatedly sample two distinct rows and two distinct columns. Accept a proposal only
when the selected submatrix is one of

```text
1 0      0 1
0 1  or  1 0
```

and replace it with the other orientation. Stop after exactly `10 * L.size` accepted
switches; fail loudly if this is not reached within `1000 * L.size` proposals.

This preserves every state's number of positive composite demonstrations and every
exact prompt instance's positive count across states. Render a randomized `1` with
that prompt's canonical answer and a `0` with `unable`.

The control is valid only if:

- primitive records are byte-identical to Condition A per state;
- record counts and prompt bytes are identical to Condition A;
- the aggregate UTF-8 byte histogram and tokenizer-token histogram over all 16 state
  corpora are exactly identical to Condition A;
- every state has the same positive composite record count as in Condition A;
- every composite record has the same positive state count as in Condition A;
- every seen-composition family has at least one changed label;
- at least `15%` of composite label cells differ from `L`;
- generation is deterministic for the control seed.

The randomized control tests coherent composition outcomes while holding primitive
supervision fixed. It does not identify a causal effect of natural-language
structure and does not match per-state answer-token identities.

## 8. Tokenizer and Model Contract

### 8.1 Tokenizer

Implement a fixed byte-level tokenizer with IDs for all 256 byte values plus exactly
four special tokens: PAD, BOS, SEP, and EOS. The tokenizer has no training phase and
no unknown token. It must round-trip every generated UTF-8 string byte-for-byte and
reject malformed special-token placement.

Do not add `transformers`, `tokenizers`, or another tokenizer/model dependency.

### 8.2 Decoder-only Transformer

Implement the model with PyTorch `2.9.1` public APIs:

- learned token embeddings;
- learned positional embeddings;
- pre-norm causal Transformer blocks;
- GELU feed-forward layers;
- a final normalization and linear language-model head;
- no state embedding, task embedding, graph input, or external retrieval;
- dropout `0.0`;
- maximum sequence length `256` bytes including special tokens;
- greedy autoregressive decoding with a maximum of `64` generated tokens.

The small configuration is:

```text
d_model=64
n_heads=4
n_layers=2
d_ff=256
```

The medium configuration is:

```text
d_model=128
n_heads=4
n_layers=4
d_ff=512
```

Parameter counts must be computed from the constructed modules and recorded; do not
claim an approximate size as evidence.

### 8.3 Training

Use response-only cross-entropy, AdamW, and exactly:

```text
optimizer=AdamW
learning_rate=0.0003
betas=(0.9,0.95)
eps=1e-8
weight_decay=0.01
gradient_clip_norm=1.0
batch_size=64
training_steps=1500
```

Base corpora contain exactly 128 records for each of the four primitive and three
seen-composition task families: `896` records per state. The large-corpus scale cell
uses exactly 512 records per family: `3584` records per state. Sampling cycles through
a deterministically shuffled corpus without replacement within each cycle; the final
partial cycle is allowed. Do not silently replace the frozen batch schedule.

Evaluate all eight probe tasks at checkpoints after steps
`0, 100, 300, 750, 1500`. Step `0` is the initialized model before any optimizer
update. Save raw evaluation output and training loss at every checkpoint step. Save
model weights at steps `0` and `1500`; intermediate metrics are sufficient unless a
failed run requires a separate, non-evidence diagnostic rerun.

Use training seeds exactly `0, 1, 2`. For paired condition runs, `(seed,state)` must
produce identical initial weights. Before constructing every model, reset Python and
PyTorch CPU/CUDA RNGs to `900000 + seed`; all 16 states within a seed therefore also
begin from the same weights. Initialize one corpus-order `random.Random` per trained
checkpoint with seed `800000 + 100*seed + state_mask` and reuse its stream for every
successive shuffle cycle. Condition is deliberately absent from both seed formulas
so paired A/B/C runs share initialization and record-index order. Request
deterministic algorithms, disable TF32, disable cuDNN benchmarking, and set
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. If the frozen environment cannot satisfy a
deterministic operation, fail rather than changing algorithms silently.

## 9. Frozen Experiment Grid

### 9.1 Core comparison

Run small-model, base-corpus populations for:

```text
Condition A x seeds 0,1,2 x 16 states
Condition B x seeds 0,1,2 x 16 states
Condition C x seeds 0,1,2 x 16 states
```

This is `144` unique training runs.

### 9.2 Scale study

Run Condition A for the complete Cartesian grid:

```text
model size:  small, medium
corpus size: base, large
seed:        0,1,2
state:       all 16 states
```

The small/base cell is shared with the core comparison and must not be retrained.
The full milestone therefore contains `288` unique training runs, not `336`.

The grid is frozen before Phase 8 scientific outcomes are observed. It is a bounded
descriptive scale study, not architecture search and not a scaling-law experiment.
Because both corpus sizes use the same `1500` steps and batch size `64`, the
base-versus-large comparison is a fixed-compute corpus-diversity comparison: the
large cell exposes more unique records but fewer average repetitions per record. Do
not describe it as a data-scaling or increased-training-token result.

### 9.3 Software overfit smoke control

Before the formal grid, run one source-controlled, non-evidence correctness test that
checks batch construction and shifted response-only labels. A test-local model must
overfit a four-record byte-copy fixture on CPU to exact training accuracy `1.0`
within a test-local maximum of `500` optimizer steps. This fixture contains no Phase
8 DSL task or payload and is a software test, not a hyperparameter-selection pilot or
scientific artifact.

Failure of this test blocks formal execution. Passing it does not guarantee that the
formal models will learn primitives or compositions.

### 9.4 Held-out sequence-transduction feasibility control

Before any Phase 8 scientific checkpoint is trained, run a separate non-scientific
feasibility suite that contains no Phase 8 task ID, DSL program, template, operand,
payload, or state. It uses the Phase 8 byte tokenizer, small/medium model
implementations, optimizer family, response-only loss, checkpoint path, and greedy
generation path.

Use exactly four feasibility families:

1. copy one held-out 16-hex string from a natural-language prompt;
2. extract one named value from held-out key/value fields and emit its compact JSON
   string;
3. map a held-out Boolean statement to canonical JSON `true` or `false`;
4. emit a compact JSON array from held-out random string operands.

For each family use `512` training records and `64` evaluation records with disjoint
source-controlled templates and operands. Train both the small and medium
configurations for exactly `1500` steps with batch size `64` under seeds `0,1,2`.
Each configuration/seed/family must reach at least `52/64` held-out exact matches.
Retain per-record generations and counts; aggregate success cannot hide a failing
family or seed.

The suite is feasibility evidence, not Phase 8 scientific evidence. A failure blocks
formal training and returns to `main`. Architecture, training budget, tokenizer, or
generation changes are allowed only in a new source-controlled protocol commit based
solely on feasibility evidence, followed by fresh-context review; no Phase 8
scientific corpus or checkpoint may be generated first. Passing the suite establishes
only basic sequence-transduction capacity and does not guarantee primitive,
composition, identifiability, or certificate outcomes.

The initial fixed feasibility root is:

```text
artifacts/phase8_toy_lm_bridge/feasibility_001
```

It must be generated from clean, committed, independently reviewed source, refuse
overwrite, and bind its exact configuration, software/hardware environment, raw
results, retained checkpoints, file inventory, and checksums.

### 9.5 Non-scientific resource benchmark and authorization

After the complete training/evaluation/shard implementation is accepted but before
formal execution, run two source-controlled dummy-data benchmark fixtures:

```text
benchmark_small_base
benchmark_medium_large
```

Each fixture must exercise the same 16-checkpoint population shape, batch size,
sequence-length distribution, 1500 optimizer steps per checkpoint, five 512-prompt
evaluation points, checkpoint publication, status files, raw-generation writing, and
manifest path as its representative formal shard. Dummy inputs must not reuse or
reveal Phase 8 scientific outcomes.

Record training and evaluation wall time, CPU time, peak RSS, peak VRAM, checkpoint
bytes, raw-generation bytes, final artifact bytes, and per-stage breakdown. Report
measured shard cost and an explicitly parameterized whole-grid projection. A
proposed safety factor such as `1.25` is not authoritative until `main` freezes it in
the resource authorization.

Before formal execution, `main` must explicitly bind concurrency and ceilings for
wall time, GPU-hours, peak VRAM, peak RSS, temporary disk, and final disk in the
benchmark authorization artifact. A ceiling breach is an operational failure, not a
scientific outcome. The initial benchmark root is:

```text
artifacts/phase8_toy_lm_bridge/benchmark_001
```

## 10. Behavioral Certificate Analysis

For every condition/scale cell, seed, and checkpoint step, assemble one `16 x 8`
primary behavioral response matrix using the `52/64` threshold.

### 10.1 Identifiability first

Run the accepted full-information identifiability audit with a response callback
backed only by the frozen matrix. If any two rows collide, record the collision
groups and set the behavioral fixed/adaptive certificate fields to JSON `null` with
reason `not_identifiable`. Do not choose extra prompts, lower a threshold, combine
seeds, or use model logits to repair a collision.

### 10.2 Fixed and adaptive certificates

When identifiable:

- run the accepted exact fixed solver and independent validator;
- independently enumerate all `2^8` task subsets to recover every minimum fixed
  behavioral certificate;
- run entropy and balanced adaptive solvers and their independent validator;
- reconstruct leaf identities, per-state depths, average depth, and worst-case depth
  from each serialized tree and require agreement with solver metrics.

### 10.3 Metrics

For every state/task cell retain `exact_success_count`, `exact_unable_count`, and all
64 raw generations. Define

```text
a[K,q] = exact_success_count[K,q] / 64
```

and the task families `P` (four primitives), `S` (three seen compositions), and `H`
(the held-out composition). For each family `F`, report:

```text
mastery[F] = mean a[K,q] over q in F and y_gt(K,q)=1
false_positive_answer_rate[F] = mean a[K,q] over q in F and y_gt(K,q)=0
correct_refusal_rate[F] = mean exact_unable_count[K,q]/64
                          over q in F and y_gt(K,q)=0
all_state_answer_success[F] = sum exact_success_count[K,q]
                              / (64 * 16 * |F|)
```

Every `(K,q)` cell is equally weighted in these macro means. The exact positive and
negative cell denominators are respectively `32/32` for `P`, `12/36` for `S`, and
`4/12` for `H`. `all_state_answer_success` mixes mastery with capability prevalence
and must never be called mastery or general accuracy; an oracle-consistent population
would yield `0.5` for `P` and `0.25` for both composition families.

On the primary binary matrix, record false-positive count and rate over the 80 cells
with `y_gt=0`, false-negative count and rate over the 48 cells with `y_gt=1`, and
`behavioral_regret`, defined as total Hamming disagreement divided by all `16 * 8 =
128` cells. Also record family-stratified bit false-positive and false-negative rates
using the cell denominators above.

Record the following population-level metrics with these exact denominators:

- ground-truth-certificate state identification: assigned-state matches divided by
  `16`;
- exact full-signature state matching: assigned-state complete-signature matches
  divided by `16`; another state's signature and a non-ground-truth signature are
  both incorrect rather than mapped to a nearest state;
- exact behavioral certificate size `m`, `fixed_task_ratio = m/8`, and
  `fixed_task_savings = 8-m`;
- canonical selected-task Jaccard overlap with the unique ground-truth certificate;
- minimum, arithmetic mean, and maximum Jaccard overlap across all minimum behavioral
  certificates;
- entropy and balanced adaptive average and worst-case query depths under equal state
  weighting;
- the earliest frozen checkpoint step at which the primary behavioral matrix is
  identifiable, if any.

One certificate or adaptive-tree query means one 64-prompt task-family battery, not
one model generation. Report both family-query depths and their raw-generation costs
obtained by multiplying by `64`; do not compare a family-query count directly with an
individual-generation budget.

The canonical behavioral certificate is the accepted exact solver's deterministic
selection. Family-wide overlap statistics prevent conclusions from depending only
on its tie order.

State-identification and full-signature matching must be recomputed independently
from raw matrix rows. Behavioral-certificate self-identification is tautological and
must not be reported as evidence of ground-truth recovery.

For Conditions A and C, record response-target length diagnostics by state and task
family: total UTF-8 bytes and tokenizer tokens; UTF-8 byte and tokenizer-token
minimum, median, arithmetic mean, and maximum over all targets; the same statistics
over positive-answer targets; the canonical `unable` length; and the
unavailable-record ratio. Also record paired A-minus-C differences for every
diagnostic. These values diagnose a known unmatched per-state nuisance; they are not
acceptance gates or permission for post-hoc reweighting or corpus repair.

In addition to the primary `52/64` matrix, reconstruct fixed sensitivity matrices at
thresholds `32/64`, `48/64`, and `64/64`. For every sensitivity matrix report its
Hamming distance from the primary matrix, row-collision groups, identifiability,
fixed-certificate existence or `null` reason, minimum fixed-certificate size, all
minimum fixed certificates, canonical-set Jaccard overlap with the primary matrix's
canonical minimum certificate, and minimum/mean/maximum Jaccard over the Cartesian
product of both all-minimum-certificate families when both are defined. Sensitivity
thresholds may diagnose fragility but may not replace the primary threshold or be
selected by outcome.

### 10.4 Predeclared interpretation hierarchy

Interpret results in the following order. These are reporting constraints, not
acceptance gates:

1. **Primitive readiness.** Always report primitive mastery, primitive false-positive
   answer rate, and primitive correct-refusal rate, including separate `MEMORY` and
   `SEARCH` cells. For a held-out-positive state `K`, define
   `parent_ready(K)` to mean that the primary matrix has both
   `y_lm(K,MEMORY)=1` and `y_lm(K,SEARCH)=1`. Report held-out-composition outcomes
   separately for parent-ready and non-parent-ready states. If any held-out-positive
   state is not parent-ready, a full-population held-out failure is not clean evidence
   of compositional failure.
2. **Seen composition.** Always report the exact twelve positive seen-composition
   cell bits and the declared family metrics. Any broad claim that learned
   composition was established must be supported by those cells rather than inferred
   from a pooled score.
3. **Held-out composition.** Report `MEMORY_SEARCH` only alongside primitive and seen
   composition results, split by parent readiness and by explicit versus indirect
   evaluation style. No single held-out score is sufficient on its own.
4. **Identifiability.** A row collision means only that this frozen probe family does
   not uniquely identify all sixteen assigned states at that checkpoint. It does not
   establish absence of latent or alternative structure.
5. **Certificate comparison.** Compare behavioral and ground-truth certificates only
   for identifiable matrices. A mismatch is a difference under the frozen task,
   prompt, threshold, and state population, not evidence about general language-model
   capability structure.

The unique size-four ground-truth certificate is forced by the full four-primitive
powerset and the inclusion of primitive probes: each primitive coordinate must be
distinguished directly. It is a structure-preservation baseline, not a certificate
compression created by composition and not evidence that composition queries reduce
the ground-truth certificate.

## 11. Statistical Semantics

- One complete 16-checkpoint `(condition, scale cell, seed, step)` population is the
  unit of certificate analysis.
- States are equally weighted because the declared population is the full powerset;
  this is not a deployment prior.
- The 64 probes per task are repeated measurements used to define one response bit,
  not 64 independent model replicates.
- Report every seed separately, then equal-weighted mean, median, minimum, and maximum
  across the three seeds when an aggregate is useful.
- A certificate metric that is undefined because one or more seeds are not
  identifiable remains undefined for those seeds. The across-seed aggregate for that
  certificate metric is JSON `null` unless all three seed values are defined; always
  retain the three per-seed values and never average invalid certificates away.
- For every scalar metric, report paired differences `A(seed)-C(seed)` and
  `A(seed)-B(seed)` as ordered three-element seed vectors. Report their arithmetic
  mean, median, minimum, and maximum only when all three pairs are defined; otherwise
  set the paired aggregate to JSON `null` while retaining defined per-seed
  differences. Do not substitute zero or drop a seed.
- Do not pool checkpoints, model sizes, corpus sizes, task families, or threshold
  sensitivity matrices into pseudo-replicates.
- Do not report p-values, confidence intervals, or claims of statistical
  significance from this bounded grid.

## 12. Required Implementation

### 12.1 Mandatory engineering decomposition

This protocol must not be implemented as one monolithic executor task. The frozen
subtask handoffs are stored under `phase8/`:

```text
phase8/Task_010A_DSL_Oracle_and_Probe_Pack.md
phase8/Task_010B_Corpus_and_Control.md
phase8/Task_010C_Tokenizer_Model_and_Feasibility.md
phase8/Task_010D_Training_Checkpoint_and_Evaluator.md
phase8/Task_010E_Behavioral_Certificate_Integration.md
phase8/Task_010F_Metrics_and_Aggregation.md
phase8/Task_010G_Sharded_Runner_and_Provenance.md
phase8/Task_010H_Resource_Benchmark_and_Formal_Authorization.md
```

Every subtask uses a fresh-context Spark execution role with model override
`gpt-5.5` and `fork_turns=none`. The local routing contract names the available model
identifier exactly; do not invent a separate `gpt-5.5-spark` model slug. Before each
launch, `main` supplies the accepted prerequisite commit, working directory, allowed
paths, acceptance criteria, verification commands, and return format from the
corresponding handoff.

Only one delegated writer may hold the mutation lease. A delegate must stop at its
handoff boundary, must not implement a later subtask, and must return protocol
ambiguities rather than making scientific choices. `main` freezes and commits each
stage, resolves review findings, controls authorization, and launches the next stage
only from accepted prerequisites. Independent review is required in proportion to
risk and is never replaced by executor self-review.

### 12.2 Expected implementation paths

```text
capability_certificate_lab/lm_bridge/__init__.py
capability_certificate_lab/lm_bridge/corpus_generator.py
capability_certificate_lab/lm_bridge/tokenizer.py
capability_certificate_lab/lm_bridge/model.py
capability_certificate_lab/lm_bridge/train.py
capability_certificate_lab/lm_bridge/evaluator.py
capability_certificate_lab/lm_bridge/certificate_eval.py
scripts/phase8_sequence_feasibility.py
scripts/phase8_resource_benchmark.py
scripts/phase8_toy_lm_bridge.py
tests/test_lm_bridge.py
```

Use only the Python standard library, existing project code, and PyTorch. Do not add
a generalized training framework, plugin system, registry hierarchy, distributed
training layer, or third-party experiment tracker.

The implementation must expose pure functions for corpus generation, control
validation, tokenization, probe scoring, response-matrix construction, and
certificate metric reconstruction so tests do not need to launch the formal grid.

## 13. Required Tests

Targeted tests must cover at least:

1. exact four-primitive state order and eight-task program order;
2. exact DSL output, required-primitive, task-specific context-schema, and memory
   dependency-perturbation contracts for every program;
3. ground-truth `16 x 8` matrix, identifiability, unique size-four certificate, and
   independent subset-enumeration agreement;
4. deterministic corpus generation and record ordering for a seed;
5. byte tokenizer round-trip, special-token validation, padding, training-record and
   evaluation-prefix length gates, truncation refusal, and response-only loss mask;
6. literal capability/graph/state leakage rejection in model-facing text;
7. absence of rendered seed IDs; exact prompt/template/ID and non-`CONDITION`
   execution-relevant context/payload split non-overlap; deliberate semantic-collision
   rejection; the explicit `CONDITION` overlap contract; and complete absence of
   `MEMORY_SEARCH` from training;
8. one byte-identical 512-prompt evaluation-pack checksum across every run, frozen
   explicit/indirect style allocation, within-style Boolean balance, and distinct
   non-Boolean evaluation-answer gates;
9. A/B structured outcome agreement and primitive-record identity;
10. A/C primitive byte identity, degree preservation, aggregate byte/token histogram
    equality, changed-label threshold, and deterministic randomization;
11. deliberate malformed or insufficiently randomized controls are rejected;
12. causal masking and absence of future-token access;
13. deterministic initialization, optimizer stepping, and greedy decoding;
14. save/load round-trip with exact state-dict tensors and logits;
15. four-record CPU overfit smoke control;
16. evaluator isolation and exact-answer behavior, including refusal not counting as
    task success;
17. primary and sensitivity threshold boundaries, all raw/family-stratified metric
    denominators, paired null aggregation, fixed ratio/savings, and query-unit
    conversion;
18. behavioral identifiability, collision, non-ground-truth signature, and
    no-certificate paths;
19. all-minimum-certificate enumeration and overlap statistics;
20. independent adaptive-tree reconstruction agreement;
21. feasibility-family train/evaluation disjointness, exact per-cell pass/fail
    handling, retained raw generations, and rejection of every Phase 8 scientific
    task, template, operand, payload, or state marker;
22. target-length diagnostics and exact `32/64`, `48/64`, `52/64`, and `64/64`
    response-matrix sensitivity reconstruction;
23. fixed shard registry, per-attempt atomic status transitions, retry selection,
    binding consistency, and refusal to aggregate missing, failed, mixed-source, or
    mixed-configuration shards;
24. benchmark resource accounting, whole-grid projection, and rejection of missing
    or exceeded authorization ceilings.

Run targeted tests first, then:

```text
python -m pytest -q tests/test_lm_bridge.py
python -m pytest -q
git diff --check
```

Record exact test counts, Python version, PyTorch version, CUDA runtime, GPU, and
driver. Historical Phase 2-7 test counts are not Phase 8 verification.

## 14. Formal Execution and Provenance

The initial fixed output root is:

```text
artifacts/phase8_toy_lm_bridge/formal_001
```

The formal root contains exactly eighteen registered shards:

```text
core_A_small_base_seed0   core_B_small_base_seed0   core_C_small_base_seed0
core_A_small_base_seed1   core_B_small_base_seed1   core_C_small_base_seed1
core_A_small_base_seed2   core_B_small_base_seed2   core_C_small_base_seed2
scale_A_medium_base_seed0   scale_A_medium_large_seed0   scale_A_small_large_seed0
scale_A_medium_base_seed1   scale_A_medium_large_seed1   scale_A_small_large_seed1
scale_A_medium_base_seed2   scale_A_medium_large_seed2   scale_A_small_large_seed2
```

Each shard contains all sixteen states and every one of the five evaluation
checkpoints for its declared condition, model size, corpus size, and seed. The
`core_A_small_base_seed*` shard is the shared small/base Condition A cell; it must
not be retrained in the scale study. The fixed layout is:

```text
formal_001/
  manifest.json
  shards_manifest.json
  shards/<shard-id>/attempt001/
  aggregate/
```

Use only the following parameterized command surface:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_toy_lm_bridge.py prepare --device cuda:0 --output-root artifacts/phase8_toy_lm_bridge/formal_001
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_toy_lm_bridge.py run-shard --device cuda:0 --output-root artifacts/phase8_toy_lm_bridge/formal_001 --shard-id <registered-id> --attempt 001
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_toy_lm_bridge.py aggregate --output-root artifacts/phase8_toy_lm_bridge/formal_001
```

`prepare` is permitted only after the software smoke control and all feasibility
cells pass, the resource benchmark and explicit authorization are accepted, the
implementation and tests are committed, an independent strict read-only review
accepts that exact commit, and an unexcluded `git status --short` confirms a clean
worktree. The fixed output root must not exist and `prepare` must refuse overwrite.
It atomically locks the source commit, configuration, evaluation-pack checksum,
eighteen-shard registry, resource authorization, and output root before any shard
trains.

After generation, the source-cleanliness check may exclude exactly the newly created
fixed output root and nothing else. No pre-run cleanliness check may use an
exclusion.

Every shard attempt must write long-running output to job logs and atomically
maintain small `status.json` and `progress.json` files. It must end with exactly one
of `DONE.json` or `FAILED.json`. Attempt publication is atomic, existing attempts are
never overwritten, and monitoring must use the small status files rather than
continuous stdout inspection.

A transient infrastructure failure may be retried only with the identical accepted
source and locked configuration, after explicit `main` authorization, in the next
attempt directory such as `attempt002`; retain the failed attempt and declare the
selected successful attempt in `shards_manifest.json`. A source, configuration, data,
evaluation, or semantic defect invalidates every affected formal result. Preserve the
old root as rejected evidence, repair and review a new source-controlled commit, and
start a new top-level root such as `formal_002`. Never combine attempts with different
source commits or configurations.

`aggregate` must accept exactly one declared `DONE` attempt for each of the eighteen
registered shards and independently verify their source, configuration,
evaluation-pack, and resource-authorization bindings. Missing, failed, extra,
mixed-source, or mixed-configuration shards block final aggregation; completed
shards remain preserved.

The manifest must bind:

- exact source commit and dirty-tree checks;
- exact command, launcher path, output root, and device;
- Python, PyTorch, CUDA, GPU, and driver versions;
- deterministic-algorithm and backend flags;
- frozen task/state order, conditions, grid, seeds, thresholds, and checkpoint steps;
- template and payload inventories and the single shared evaluation-pack checksum;
- every corpus and split checksum;
- every retained model checkpoint checksum;
- every raw generation, response matrix, tree, result, and log checksum;
- a complete exact output-file inventory with file role and byte size, excluding only
  the manifest itself to avoid an impossible self-hash.

Versioned source, tests, protocol, report, summary JSON, manifest, and compact raw
behavior/certificate results are bound by Git. Corpora, logs, and model weights are
external evidence and require path and SHA-256 binding in the committed manifest.
Checksums establish byte identity only; they do not replace tests or scientific
review.

The top-level manifest must bind the selected shard attempts and a complete output
inventory, excluding only the manifest itself. Sharding is an execution boundary and
does not change any scientific unit or create additional replicates.

## 15. Experiment and Acceptance Gates

The following are implementation/protocol gates and block acceptance on mismatch:

- ground-truth DSL program, matrix, identifiability, and unique-certificate oracles;
- DSL context-schema and sequential dependency-perturbation oracles;
- corpus reproducibility, raw-seed leakage, task-wise execution-relevant payload
  split with the explicit `CONDITION` exception, single shared evaluation-pack
  identity/style balance, answer-distribution, evaluation-prefix length, and held-out
  gates;
- randomized-control degree, token-statistic, determinism, and changed-relation
  gates;
- tokenizer, causal model, training, save/load, evaluator, and threshold tests;
- the software smoke control and every frozen feasibility family/model/seed cell;
- accepted resource benchmark evidence and explicit concurrency, time, memory, and
  disk ceilings before `prepare`;
- complete raw output for all 288 unique runs and all frozen evaluation steps, unless
  a preserved failed run stops the milestone for an explicit decision;
- exactly eighteen registered, binding-consistent successful shards and one accepted
  aggregate, with retry or invalidation semantics followed exactly;
- response matrices reconstructed from raw generations;
- exact/adaptive validator and independent reconstruction agreement;
- correct seed/state/checkpoint denominators and undefined-metric handling;
- clean committed source, fixed command/root, atomic status publication, and complete
  artifact/checksum provenance.

The following are empirical outcomes, not acceptance gates:

- whether primitive answer-success counts exceed the mastery threshold;
- whether A outperforms B or C;
- whether `MEMORY_SEARCH` generalizes;
- whether the behavioral matrix is identifiable;
- whether a behavioral certificate exists or overlaps the ground truth;
- whether larger corpora or models improve any metric;
- whether certificate appearance is monotonic over checkpoints.

Do not change the world, templates, randomization, grid, seeds, thresholds, training
steps, model sizes, or claims after observing formal outcomes.

## 16. Deliverables and Completion Criteria

Expected accepted evidence paths:

```text
artifacts/phase8_toy_lm_bridge/feasibility_001/summary.json
artifacts/phase8_toy_lm_bridge/feasibility_001/manifest.json
artifacts/phase8_toy_lm_bridge/benchmark_001/summary.json
artifacts/phase8_toy_lm_bridge/benchmark_001/manifest.json
artifacts/phase8_toy_lm_bridge/benchmark_001/authorization.json
artifacts/phase8_toy_lm_bridge/formal_001/shards_manifest.json
artifacts/phase8_toy_lm_bridge/formal_001/aggregate/summary.json
artifacts/phase8_toy_lm_bridge/formal_001/manifest.json
phase8_report.md
Capability_Certificate_Research_Progress_Summary.md
```

Phase 8 is complete only when:

1. The software smoke control and every held-out sequence-transduction feasibility
   cell pass from accepted, clean, committed source.
2. A resource benchmark is complete and `main` has frozen explicit concurrency,
   time, memory, and disk ceilings.
3. DSL programs deterministically generate leak-checked A/B/C training corpora and a
   held-out probe pack.
4. Toy causal models train and evaluate reproducibly under the frozen configuration.
5. Primitive, seen-composition, and held-out-composition probes retain raw outputs.
6. Every valid behavioral matrix is analyzed with independent fixed/adaptive
   certificate checks, while non-identifiable matrices are reported without a fake
   certificate.
7. All eighteen shards and the binding-consistent aggregate are complete for every
   frozen condition and scale cell, or the milestone stops with preserved failure
   evidence and an explicit decision.
8. The report distinguishes protocol gates, empirical outcomes, unsupported
   hypotheses, and limitations.
9. An independent fresh-context final scientific review finds no unresolved
   correctness, leakage, control-validity, provenance, statistical, or overclaiming
   finding.

After completion, stop. Do not enter open-model evaluation, large-scale pretraining,
real-world benchmarks, or learned graph discovery.

## 17. Required Report Boundaries

The report must state at least:

- this is a synthetic, templated, byte-level language-model experiment;
- each row is a separately trained checkpoint from a state-specific corpus, not a
  naturally occurring model population;
- checkpoints within a seed share initial weights and are neither independently
  initialized nor independent statistical replicates;
- refusal demonstrations explicitly supervise missing capabilities;
- natural task semantics remain visible even though symbolic capability IDs and graph
  rules are hidden;
- executor-relevant train/evaluation operands are disjoint except for primitive
  `CONDITION`, whose Boolean domain necessarily overlaps and therefore tests only
  template generalization;
- every condition and checkpoint uses one byte-identical evaluation pack with equal
  explicit and indirect composition-template allocation;
- the random control preserves declared aggregate statistics but does not identify a
  general causal effect of language structure;
- Condition A versus C does not match response-target identities or lengths within
  every state; the frozen target-length diagnostics expose rather than remove this
  limitation;
- the base-versus-large comparison holds optimizer steps and batch size fixed and is
  a corpus-diversity comparison, not a data-scaling or increased-training-token
  result;
- the ground-truth size-four certificate is forced by the full primitive powerset
  and primitive probes; it is not composition-induced certificate compression;
- held-out composition is interpreted only after primitive readiness and seen
  composition, with parent-ready and evaluation-style strata reported separately;
- the primary response matrix depends on a predeclared mastery threshold;
- sensitivity thresholds diagnose threshold fragility but never replace the primary
  `52/64` result;
- mastery and false-positive answer rates are stratified by `y_gt`; pooled all-state
  answer-success values reflect capability prevalence and are not called accuracy;
- one certificate query is a 64-prompt task-family battery rather than one model
  generation;
- three seeds and repeated probes do not justify population-level significance;
- feasibility and resource benchmarks are non-scientific controls, and sharding does
  not create scientific replicates;
- a recovered certificate describes only this frozen probe family and state
  population;
- success does not establish real-LLM capability structure, and failure does not show
  that larger models cannot learn such structure.

## 18. Milestone-start Scientific Review Gate

Freeze this handoff in a commit and start a fresh-context `gpt-5.6-sol` strict
read-only reviewer. Provide only:

- this handoff path and exact commit SHA;
- accepted baseline `6d7012a905f8814e0788095bb3eee54a83616cc3`;
- the accepted Phase 2-7 summary;
- current public DSL, identifiability, exact, adaptive, and validation APIs;
- explicit instruction not to implement, edit, or run training/experiments.

The reviewer must audit:

- whether one-checkpoint-per-state makes the certificate object well defined;
- DSL program/type compatibility and the ground-truth certificate proof;
- task-specific context schemas, sequential information dependence, evaluation
  length bounds, raw-seed leakage, execution-relevant payload separation with the
  `CONDITION` exception, and held-out split semantics;
- identity and explicit/indirect balance of the single evaluation probe pack across
  conditions, states, seeds, scale cells, and checkpoints;
- feasibility and fairness of the degree-preserving random control;
- tokenizer/model/training sufficiency without architecture search, including the
  held-out feasibility suite and its frozen repair boundary;
- response threshold, stratified raw rates, error, regret, overlap,
  state-identification, paired-null aggregation, target-length diagnostics,
  sensitivity matrices, interpretation hierarchy, and query-unit metrics;
- seed, state, probe, checkpoint, and scale denominators;
- whether any empirical hypothesis has leaked into an acceptance gate;
- eighteen-shard completeness, retry versus invalidation semantics, resource
  authorization, artifact sufficiency, and Git/checksum provenance;
- the forced ground-truth-certificate scope and absence of composition-compression
  overclaiming;
- scope discipline and scientific overclaiming.

Verdict must be `ACCEPT` or `REJECT`. Resolve every confirmed finding in a new commit
and re-review the repaired handoff before implementation begins.

## 19. Return Format

Return to `main` at each later gate:

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

A blocked stage returns:

```text
Problem:
Minimal reproduction:
Evidence:
Likely cause:
Scientific or engineering classification:
Decision required from main:
Work explicitly not attempted:
```
