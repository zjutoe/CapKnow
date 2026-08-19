# Task 010C-D2-I — Weight-Tying Protocol Implementation

## Launch contract

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
mutation lease: this executor only
commit authority: none; main owns inspection, commit, review, and experiment gates
```

`main` supplies the exact handoff commit in the launch packet. The accepted protocol
decision is immutable commit:

```text
9a767c6708c7c69f5ba98848250afcf50c8c5a6f
```

That commit's `phase8/Task_010C_D2_Feasibility_Protocol_Amendment_Proposal.md`
received a fresh-context `gpt-5.6-sol`, `xhigh`, strict read-only verdict of `ACCEPT`
with no remaining findings. Read that proposal completely before editing. Also read
master Sections 8.2–9.4, Task 010C, Task 010C-D1, and this handoff.

## Authority and objective

Implement exactly the accepted D2 input/output weight-tying amendment and the
minimum provenance/legacy-validation changes required to make the single future
`feasibility_005` run executable. Update the controlling documentation to the new
protocol revision.

This implementation does not authorize or run `feasibility_005`, create a selection,
change a threshold, continue to Task 010D, or write under
`artifacts/phase8_toy_lm_bridge/`. It must return a reviewable source diff to `main`.

## Allowed paths

```text
Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md
phase8/README.md
phase8/Task_010C_Tokenizer_Model_and_Feasibility.md
capability_certificate_lab/lm_bridge/model.py
capability_certificate_lab/lm_bridge/train.py
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

Do not modify this handoff or the accepted D2 proposal. Do not touch corpus modules,
tokenizer code, other tests, evidence roots, dependencies, or configuration files.
If a necessary change falls outside the list, stop and return the exact need to
`main`.

## Sole scientific change

Both frozen model sizes use:

```text
embedding_weight_tying = true
model_protocol_revision = "phase8_tied_io_v1"
```

The two fields are required, with no defaults, in every new `TransformerConfig`.
`transformer_config("small")`, `transformer_config("medium")`, and `build_model`
return only this tied revision.

Retain the existing construction order exactly:

```text
token_embedding
position_embedding
blocks
final_norm
independent bias-free lm_head
lm_head.weight = token_embedding.weight
```

The independent head initialization is consumed and discarded. Do not reinitialize
the token embedding or assign the independently initialized head tensor in the other
direction. The final objects must satisfy:

```text
model.lm_head.weight is model.token_embedding.weight
small parameter_count == 133120
medium parameter_count == 859392
```

`model.parameters()` and the AdamW optimizer must contain the shared parameter once.
For sizes small/medium and seeds `0,1,2`, a test-only explicit historical constructor
must prove that the tied tensor is byte-equal to the old constructor's initialized
token embedding and that post-construction PyTorch CPU and all CUDA RNG states equal
the old constructor's states.

All other model, tokenizer, optimizer, loss, batch schedule, initialization seed,
backend, training, checkpoint, decoding, and scoring semantics remain unchanged.

## Checkpoint contract

Every newly saved checkpoint records both exact revision fields in its configuration.
Its state dict retains both canonical keys:

```text
token_embedding.weight
lm_head.weight
```

They must be tensor-equal when saved. Before loading a tied checkpoint, reject a
missing key, non-tensor value, dtype/shape mismatch, unequal bytes, missing revision
field, false tying flag, or revision other than `phase8_tied_io_v1`. Construct the
tied model first and preserve Python object identity after `load_state_dict`; never
accept PyTorch's last-key-wins behavior for divergent duplicate entries.

`load_model_from_checkpoint` is a new-revision loader and must fail closed on an old
or ambiguous checkpoint. Historical loading is a separate explicit internal path,
used only after the exact artifact allowlist below has been validated. It must not be
a silent fallback from the current loader.

## Exact historical boundary

The retained roots remain untied with counts small `149760` and medium `892672`.
Accept missing revision fields only for these exact path/source/checksum tuples:

| Root | Source commit | Manifest SHA-256 | Terminal SHA-256 |
| --- | --- | --- | --- |
| `feasibility_001` | `949683d7fd20461da97aa439915f432d18da7680` | `422f31c5110794892412499233406669edea82d271538c77f58e1ce493d8dd88` | `47187b0e2a2d0d151fd1ea8a19c0809bf7ca9eff5d5d9b401996cff763eb64ab` |
| `feasibility_002` | `ef893716373b9c83a33e0bfe71e64e1aa93f55bf` | `408f5737e0c574d355f589ce3180ee436a5a85bfdf74e448ae1575e162fb6151` | `2d3b15f074c5bf8ccc14ae0b00a3d4a97b58baa705f90c9eade24d9294da2eda` |
| `feasibility_003` | `530ea0bf96c9eab5fbc94bf3951f5bf712315a20` | `399aa93cb43a8d086ba2f26a87af25955e782d2d045eadafcce84c763de63c7b` | `82c5f88e4a4159e767fbeac1e802b9bb1db5f998c79dc937288ce378c471253e` |
| `feasibility_004` | `3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea` | `19cf2db4df6f6928d0afa7ab8aef8d891e944152e480541f5aaf8702565e8d00` | `6b3c81d3be78c8e3deeaf991dad4a1e2df41171187ba4e20533799162a46f3ae` |

The terminal is `FAILED.json` in every row. Validate path, complete source commit,
manifest checksum, terminal checksum, manifest-bound terminal checksum, old exact
configuration schema, old state schema, and old parameter count before constructing
an untied model. A different path containing copied bytes is not allowlisted.

The only additional historical untied checkpoints are the three D1-owned files below:

| D1 inventory path | SHA-256 |
| --- | --- |
| `array_json__small__3000__seed0/checkpoint_step3000.pt` | `8311fb311fc9f7b707be4bb6fe82eed19b99c024c31341e9013052903f00365b` |
| `array_json__small__3000__seed1/checkpoint_step3000.pt` | `b2684d53fde02136053d89090ef1595bb781a749c511f4c0a52a8c1c318d9d0a` |
| `array_json__small__3000__seed2/checkpoint_step3000.pt` | `709d8c628d4835dfc0fe5ed8f382c80919539207c7ab63135f4c592f6d04ad2f` |

They qualify only after the canonical diagnostic path, source commit,
manifest/summary/DONE checksums, complete file inventory, and the individual inventory
path/size/checksum have all matched the D1 binding below. D1 references to reused
1500-step checkpoints still route through the exact `feasibility_004` allowance.
No other diagnostic root, D1 file, checkpoint, or copied path qualifies for historical
model loading.

Use an explicit historical configuration/construction helper. Its in-memory marker
may be `embedding_weight_tying=false` and
`model_protocol_revision="phase8_untied_legacy_v1"`, but those marker fields are not
pretended to exist in old serialized checkpoint dictionaries. Old checkpoint config
dictionaries must match the exact pre-D2 key set and values. Do not rewrite old
files, migrate their dictionaries, or recompute expected hashes from their contents.

Keep the D1 validator semantically historical. Any D1 code that reconstructs a model,
configuration, parameter count, state schema, fingerprint, checkpoint, or replay must
use the explicit untied revision and the original `149760`/`892672` counts. Do not
reinterpret the completed diagnostic through the tied constructor. Future formal and
current feasibility paths use only tied models.

## Frozen record bytes

Make all eight hashes executable preflight invariants, computed with the existing
canonical record serialization before model construction:

| Record set | SHA-256 |
| --- | --- |
| hex train512 | `8798d57c3bdde6693d8046e3a7525687f06976074dc0cceffaf1c674a0f8c390` |
| hex eval64 | `be97a3a2877fa5f8e6c2e85a4213f975ccc89de6b3c61338dff100d1d1f21868` |
| named train512 | `9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8` |
| named eval64 | `6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20` |
| Boolean train512 | `52b5d74bec65bc903938b8b1993c1bf6df489d9c13659ad62aca76336c2d77f1` |
| Boolean eval64 | `3c2ce21e86dd88ce85c6d2927c8ff09eb8f87cecb223bdeb1009debff68dff3b` |
| array train512 | `6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94` |
| array eval64 | `8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a` |

Do not change records, order, IDs, prompts, answers, templates, operands, split
semantics, or canonical serialization.

## D1 decision-provenance binding

The future `run` CLI requires exactly one:

```text
--decision-diagnostic-root artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

It is distinct from predecessor roots/selections. Bind and validate:

```text
path: artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
source commit: cb49ebdf577df78e97b7748aadc48f8547a70f6a
artifact_class: non_evidence_feasibility_diagnostic
feasibility_selection_eligible: false
task_010d_authorized: false
manifest SHA-256: 5589ac3a1215ea4cf451bd14f75ab64ab53b183548d1d42134e3f52b6ad12241
summary SHA-256: 91582eda4629f29436289896c9f3802fe243c2b86a8f622c98c6c97e579a34ac
DONE SHA-256: 39f3893627d24162d94f541217f4f754c2e193fe4a61361dacc6156b84826f92
accepted proposal commit: 9a767c6708c7c69f5ba98848250afcf50c8c5a6f
independent review verdict: ACCEPT
```

Deep-validate its complete inventory, input/checkpoint lineage, common fields, and D1
semantics with the historical model path. The new feasibility manifest, summary, and
terminal bind it under a dedicated `decision_diagnostic` object. It cannot appear as
a predecessor root, selection, pass cell, selected root, or source of a new metric.

Source cleanliness may exclude only the exact inventory-bound files under all four
predecessor roots and this diagnostic. Do not whitelist the parent artifact tree.

Implement this as two ordered validation phases. First, perform only shallow canonical
path, terminal, manifest, and complete inventory size/checksum validation sufficient
to derive the exact allowed Git-status paths. This phase must not construct a model,
build a training schedule, load a checkpoint tensor for semantic validation, perform
a forward/replay, or create an output/temp root. Second, validate Git tracked/staged,
untracked, and ignored source cleanliness against only those paths. Deep checkpoint,
D1, lineage-semantic, and replay validation may begin only after cleanliness passes.

## Current feasibility schema

The new `frozen_configuration()` must include the exact tying flag, revision literal,
device `cuda:0`, and all eight record hashes in addition to the unchanged constants.
Every new manifest and summary stores that exact object. Every terminal stores and
binds that exact configuration, the decision diagnostic, and cells. Each new cell
records the tied flag, revision, and tied parameter count. Current validators reject
missing/additional/divergent fields; historical schemas are accepted only through the
allowlist above.

`feasibility_005` must explicitly bind all four predecessor roots in numerical order
and no predecessor selection. Preserve the existing complete/continuous lineage
rules. Selection validation must accept a future passing tied root only when its
current schema, exact decision diagnostic, and all historical predecessor bindings
validate; it must never accept the D1 root itself.

## Exact device, environment, and command gate

The `run` CLI requires `--device cuda:0`, rejects programmatic `main(argv=...)`, and
verifies the real process argv. It forbids CPU fallback, implicit `cuda`, another
device index, additional flags, reordered/missing roots, a predecessor selection, or
another output root.

Before creating `feasibility_005.tmp`, constructing any model, or generating any
training schedule, require:

```json
{
  "cuda": "13.0",
  "cuda_available": true,
  "gpu": "NVIDIA A800 80GB PCIe",
  "gpu_driver": "590.48.01",
  "platform": "Linux-6.12.0-184.el10.x86_64-x86_64-with-glibc2.39",
  "python": "3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]",
  "torch": "2.9.1+cu130"
}
```

Also require exactly:

```text
PYTHONDONTWRITEBYTECODE=1
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONPATH=.
```

Configure and verify deterministic algorithms, CUDA/cuDNN TF32 disabled, cuDNN
benchmarking disabled, and cuDNN determinism enabled. Retain the frozen construction
semantics: initialize every production model on CPU with the accepted CPU RNG stream,
complete the weight alias there, and only then move the completed model to
`torch.device("cuda:0")`. Every production training forward, evaluation, generation,
publication replay, and current-root semantic replay uses exactly `cuda:0`. Hashing,
checkpoint-schema inspection, shallow cleanliness authorization, and other non-forward
schema checks may use CPU. The test-local four-record smoke remains an explicitly
non-evidence CPU test. An environment/device mismatch fails preflight without creating
the output or temp root.

The only future process command is exactly:

```text
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_sequence_feasibility.py run --device cuda:0 --root artifacts/phase8_toy_lm_bridge/feasibility_005 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_001 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_002 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_003 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_004 --decision-diagnostic-root artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

The implementation must encode this command contract, but neither executor nor tests
may launch it. Tests use pure/preflight seams and temporary roots outside the
artifact tree.

## Run and terminal behavior

The future run always executes all 24 family/model/seed cells, even after an earlier
cell is below threshold. It trains only tied models for exactly 1500 steps and retains
the unchanged full outputs. It publishes `DONE` only if all 24 cells are at least
`52/64`; otherwise it preserves a complete `FAILED` root. No pilot, early stop,
model-size selection, threshold relaxation, step/data increase, surface change,
same-root retry after semantic execution, or `feasibility_006` fallback exists.

A preflight refusal that creates no output/temp root and no model/corpus/metric state
returns to `main`; only `main` may classify whether an operational correction can
reuse the still-absent `feasibility_005` root. The implementation must not encode an
automatic retry.

## Documentation changes

Update master Sections 8.2 and 9.4 to record the tied revision, exact counts,
historical lineage/decision-diagnostic boundary, device/environment gate, one full
`feasibility_005`, and stop rule. Update Task 010C and the Phase 8 index to point to
this accepted amendment and keep `010D` unauthorized pending a reviewed pass and
selection. Do not weaken or remove earlier scientific safeguards.

## Required tests

Add or update focused tests for:

1. exact config fields, alias identity, counts, construction-direction oracle, CPU and
   CUDA RNG-state equality, one optimizer parameter, and deterministic replay for both
   sizes/seeds;
2. tied checkpoint save/load, state-key equality, post-load alias identity, and every
   divergent/missing/malformed revision/state rejection;
3. exact legacy allowlist, including only the three checksum-bound D1-owned
   checkpoints after full D1-root authorization, untied config/count/state
   reconstruction, copied-path and checksum rejection, plus successful deep validation
   of roots 001–004 and D1;
4. all eight record hashes and mutation rejection before model construction;
5. exact D1 binding, complete inventory, tamper/substitution rejection, and proof it
   cannot enter selection or cells;
6. exact current manifest/summary/terminal/cell schemas and current-vs-historical
   validator routing;
7. exact real argv, environment, CPU construction followed by `cuda:0` transfer,
   backend flags, ordered roots, absent selection, fresh root, and no-output preflight
   rejection;
8. complete 24-cell execution/aggregation and all-cells pass semantics using test-local
   fakes only, never a real feasibility run;
9. regression coverage for tokenizer, response-only loss, overfit smoke, deterministic
   batching, selections, historical feasibility roots, and D1 diagnostic validation.

Use independently written expected dictionaries/hashes in tests; do not assert a
function against itself. Tests may monkeypatch GPU/model work only after separately
testing that production paths fail closed.

Add a call-order sentinel test in which source cleanliness fails after shallow
inventory authorization. It must prove that no model constructor, checkpoint tensor
load, training schedule, forward/replay, output-root creation, or temp-root creation
was reached. Add a separate passing-order test showing deep historical/D1 validation
begins only after cleanliness succeeds.

## Verification

Run at minimum:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q tests/test_lm_bridge.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -m py_compile capability_certificate_lab/lm_bridge/model.py capability_certificate_lab/lm_bridge/train.py scripts/phase8_sequence_feasibility.py tests/test_lm_bridge.py
git diff --check
git status --short
```

The full suite can be returned to `main` if execution time prevents completion, but
targeted model/checkpoint/schema/preflight tests must pass before handoff. Do not
delete or modify ignored/untracked evidence to make cleanliness checks pass.

## Stop boundary and return

Stop immediately on a protocol ambiguity, required out-of-scope path, evidence
checksum mismatch, or need to change a frozen constant. Do not commit. Do not run
training beyond test-local smoke fixtures. Return:

```text
Scope completed:
Prerequisite handoff commit used:
Accepted proposal commit used:
Changed paths:
Scientific contract changes:
Engineering-only changes:
Tests run and exact results:
Tests not run and why:
Evidence inspected and checksum results:
New artifacts:
Unresolved decisions:
Protocol deviations:
Independent review required:
Recommended next authorized step:
```

`Protocol deviations` must be `none` for acceptance. The only recommended next step
is `main` inspection, verification, commit, and fresh independent strict read-only
review of the exact implementation commit. `feasibility_005` remains unauthorized.
