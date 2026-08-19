# Task 010C-D2 — Feasibility Protocol Amendment Proposal

## Status and authority

```text
status: proposal pending independent protocol review
artifact class: source-controlled protocol proposal
execution authority: none
implementation authority: none until this proposal is accepted at an exact commit
feasibility_005 authority: none until a later implementation commit is independently accepted
Task 010D authority: false
```

This proposal is the bounded `main` decision after the accepted `010C-D1` failure
diagnostic. It proposes one falsifiable model repair and freezes the permitted
implementation and evaluation boundary. It does not amend the master handoff by
itself, select a feasibility root, authorize an experiment, or permit work on
`010D`.

If an independent review rejects this proposal, do not implement it. If the proposal
is accepted, `main` may freeze a separate implementation handoff and delegate that
implementation to a fresh-context `gpt-5.5` executor. The implementation must then
be committed and independently reviewed before any run is authorized.

## Frozen decision evidence

The proposal uses only the four immutable feasibility failures and the accepted
non-selection diagnostic below. Phase 8 scientific data, checkpoints, or outcomes do
not exist and must not be consulted.

| Root | Source commit | Manifest SHA-256 | Terminal SHA-256 |
| --- | --- | --- | --- |
| `feasibility_001` | `949683d7fd20461da97aa439915f432d18da7680` | `422f31c5110794892412499233406669edea82d271538c77f58e1ce493d8dd88` | `47187b0e2a2d0d151fd1ea8a19c0809bf7ca9eff5d5d9b401996cff763eb64ab` |
| `feasibility_002` | `ef893716373b9c83a33e0bfe71e64e1aa93f55bf` | `408f5737e0c574d355f589ce3180ee436a5a85bfdf74e448ae1575e162fb6151` | `2d3b15f074c5bf8ccc14ae0b00a3d4a97b58baa705f90c9eade24d9294da2eda` |
| `feasibility_003` | `530ea0bf96c9eab5fbc94bf3951f5bf712315a20` | `399aa93cb43a8d086ba2f26a87af25955e782d2d045eadafcce84c763de63c7b` | `82c5f88e4a4159e767fbeac1e802b9bb1db5f998c79dc937288ce378c471253e` |
| `feasibility_004` | `3cb75ad550c4357562c0d4d9a9b098bfb2cf66ea` | `19cf2db4df6f6928d0afa7ab8aef8d891e944152e480541f5aaf8702565e8d00` | `6b3c81d3be78c8e3deeaf991dad4a1e2df41171187ba4e20533799162a46f3ae` |

The terminal object in every row is `FAILED.json`. The accepted diagnostic is:

```text
root: artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
source commit: cb49ebdf577df78e97b7748aadc48f8547a70f6a
manifest SHA-256: 5589ac3a1215ea4cf451bd14f75ab64ab53b183548d1d42134e3f52b6ad12241
summary SHA-256: 91582eda4629f29436289896c9f3802fe243c2b86a8f622c98c6c97e579a34ac
DONE SHA-256: 39f3893627d24162d94f541217f4f754c2e193fe4a61361dacc6156b84826f92
artifact_class: non_evidence_feasibility_diagnostic
feasibility_selection_eligible: false
task_010d_authorized: false
independent artifact review: ACCEPT, no findings
```

The independent review recomputed the diagnostic inventory, retained records,
aggregates, checkpoint fingerprints, training trajectories, and semantic replays.

## Accepted interpretation and repair hypothesis

The diagnostic supports these bounded observations:

- Named-value exact matches, ordered as seen surface/seen operand, held surface/seen
  operand, seen surface/held operand, held surface/held operand, were
  `64,64,6,6`, `64,58,20,18`, and `64,62,14,13` for seeds `0,1,2`. Surface shift is
  present but secondary; the principal observed failure is suffix copying for held
  operands after the target prefix has usually been selected correctly.
- Array exact matches for small/1500 were `17,17,15`; medium/1500 were `59,57,57`;
  and small/3000 were `33,19,21`. All generations emitted EOS and none hit the
  generation or context cap. More steps helped some metrics but did not produce a
  stable pass. Medium/1500 was materially better, but the diagnostic does not isolate
  capacity as the cause and cannot select the medium model.
- The evidence does not establish operand novelty as a unique cause because only the
  surface contrast is paired. It also does not justify a threshold change, a budget
  increase, model selection, corpus expansion, or surface alignment as a sufficient
  repair.

The single proposed repair is to tie the bias-free language-model output weight to
the token-embedding weight in both frozen model sizes. The hypothesis is that a
shared byte representation at input and output directly improves the observed
held-suffix and array-item copying failure at fixed data and compute. This is a
one-shot falsifiable mechanism proposal, not a claim that the diagnostic proved
weight tying will work.

## Proposed protocol amendment

After proposal acceptance, the implementation commit must amend master Section 8.2
and Task 010C so that both model sizes satisfy all of the following:

1. `token_embedding` remains a learned `nn.Embedding` and `lm_head` remains a
   bias-free `nn.Linear`.
2. `lm_head.weight` and `token_embedding.weight` are the same `nn.Parameter`, not
   merely equal tensors or periodically synchronized values. Construction retains
   the existing exact module order: initialize `token_embedding`,
   `position_embedding`, every block, `final_norm`, and the independent `lm_head` in
   that order; then assign the already initialized `token_embedding.weight` object to
   `lm_head.weight`. The independent head initialization is consumed and discarded.
   The shared parameter is never reinitialized, and the initialized head weight must
   never replace the initialized token embedding.
3. The explicit model/checkpoint configuration records
   `embedding_weight_tying=true` and
   `model_protocol_revision="phase8_tied_io_v1"`. Both exact fields are required in
   every new in-memory configuration, checkpoint configuration, feasibility manifest
   configuration, summary configuration, and terminal cell/configuration binding.
   Absence of either field means the historical untied revision only under the exact
   legacy rules below; neither field may silently default for a new object.
4. Parameter counts are computed from the constructed module after aliasing and are
   exactly `133120` for small and `859392` for medium. These values follow from
   removing one independent `260 x d_model` matrix from the accepted untied counts
   `149760` and `892672`.
5. The optimizer contains the shared parameter exactly once. Saving and loading a
   new checkpoint preserves pointer identity. A tied checkpoint whose serialized
   `token_embedding.weight` and `lm_head.weight` values differ is rejected before
   loading; last-key-wins behavior is forbidden.
6. Seed reset, deterministic backend, initialization, training, checkpoint,
   evaluation, and greedy-generation paths remain common to feasibility and future
   formal models. No feasibility-only model branch is allowed.

Initialization tests must use both sizes and seeds `0,1,2`. For each pair, reset the
accepted deterministic backend and construct an explicit historical untied reference,
then reset identically and construct the tied model. The tied shared tensor must be
bitwise equal to the reference model's initialized `token_embedding.weight`, and the
post-construction PyTorch CPU and CUDA RNG states must be exactly equal between the
two constructions. This reference oracle freezes both the source tensor and the
otherwise discarded head's RNG consumption without deriving a choice from outcomes.

The amendment changes model architecture and its exact parameter count. It does not
claim comparison of new and historical exact-match counts is a controlled causal
estimate; the new full feasibility matrix is only the prespecified acceptance test
of this repair.

## Frozen invariants

No other scientific or engineering constant may change in the D2 implementation:

- tokenizer: the existing 256-byte plus PAD/BOS/SEP/EOS tokenizer;
- model widths, heads, layers, feed-forward widths, positional embeddings, pre-norm
  blocks, GELU, dropout `0.0`, context `256`, and generation cap `64`;
- optimizer, learning rate, betas, epsilon, weight decay, gradient clipping, batch
  size `64`, training steps `1500`, and seeds `0,1,2`;
- all four feasibility families, exactly 512 train and 64 eval records per family,
  their order, template IDs, operand IDs, prompts, targets, and disjointness;
- response-only loss, deterministic record schedule, checkpoint timing, greedy decode,
  per-cell `52/64` threshold, and the requirement that all 24 cells pass;
- overwrite refusal, clean-source check, atomic terminal publication, retained
  checkpoints/generations, inventory checksums, and immutable lineage.

The only authorized execution device is exactly `cuda:0`; CPU fallback, ambient
default-device selection, another CUDA index, and a non-CUDA run are forbidden. Before
creating the temporary output root or constructing a model, preflight must require
the following complete runtime environment dictionary, which is identical in the
checksum-bound `feasibility_004` and accepted D1 manifests:

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

Preflight must also require `PYTHONDONTWRITEBYTECODE=1`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `PYTHONPATH=.` and must verify deterministic
algorithms enabled, CUDA/cuDNN TF32 disabled, cuDNN benchmarking disabled, and cuDNN
determinism enabled. Environment or device mismatch is a preflight refusal, not a
reason to change the protocol or select a fallback.

The canonical record hashes at prerequisite commit
`cb49ebdf577df78e97b7748aadc48f8547a70f6a` are frozen as follows:

| Family | Train-512 SHA-256 | Eval-64 SHA-256 |
| --- | --- | --- |
| hex copy | `8798d57c3bdde6693d8046e3a7525687f06976074dc0cceffaf1c674a0f8c390` | `be97a3a2877fa5f8e6c2e85a4213f975ccc89de6b3c61338dff100d1d1f21868` |
| named value JSON | `9b55084d281a9420e12a1b6a35c3eb199abd74f4b766a14a833e6241c59564b8` | `6cc73c98616430a12a357de99702e8cb4db95258225adab80fd11212aa203d20` |
| Boolean JSON | `52b5d74bec65bc903938b8b1993c1bf6df489d9c13659ad62aca76336c2d77f1` | `3c2ce21e86dd88ce85c6d2927c8ff09eb8f87cecb223bdeb1009debff68dff3b` |
| array JSON | `6bbcae6203dfcf443f9871f30b48c29580fa721317387ff26b757b4066a98e94` | `8d504dd63ad2538aedc8195a4f3c0729ed38ec95a078b9c9895fbf93cf8c173a` |

The implementation must make these hashes executable invariants and test them. A
mismatch fails before model construction.

## Historical compatibility and lineage

New code must validate retained untied artifacts without reinterpreting them as tied
models. Compatibility is narrow:

- the missing tying field is accepted only for the exact `feasibility_001` through
  `feasibility_004` source commits and immutable manifest/terminal bindings listed
  above, and for the accepted diagnostic that transitively binds `feasibility_004`;
- their checkpoints remain untied, with parameter counts `149760` and `892672`, and
  must be reconstructed with an explicit historical untied configuration when deep
  validation or semantic replay is required;
- a new root or arbitrary checkpoint missing either `embedding_weight_tying` or
  `model_protocol_revision`, or carrying a value other than `true` and
  `"phase8_tied_io_v1"`, is rejected;
- historical files are read-only. No migration, rewrite, replacement, or new hash is
  permitted.

`feasibility_005` must bind all four failed feasibility roots in numerical order. It
must also bind the accepted D1 diagnostic as decision provenance using a dedicated
`--decision-diagnostic-root` input and a distinct manifest object containing at least
its canonical path, source commit, artifact class, manifest/summary/DONE checksums,
and the accepted proposal commit that records `main`'s review decision. A diagnostic is never a predecessor
feasibility root or selection record and can never satisfy a pass cell.

Source cleanliness may exclude only files transitively enumerated and checksum-bound
by those five exact inputs. It must not whitelist their parent artifact directory.

## Bounded implementation package after proposal acceptance

The later implementation handoff may allow changes only to:

```text
Capability_Certificate_Task_Handoff_010_Phase8_Toy_Language_Model_Bridge.md
phase8/README.md
phase8/Task_010C_Tokenizer_Model_and_Feasibility.md
phase8/Task_010C_D2_Feasibility_Protocol_Amendment_Proposal.md
capability_certificate_lab/lm_bridge/model.py
capability_certificate_lab/lm_bridge/train.py
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

The proposal file may receive only status/binding updates needed to identify its
accepted review and implementation handoff; its hypothesis and frozen constants must
not be rewritten during implementation.

Required targeted tests include:

- shared-parameter identity, exact counts, exact construction-order initialization
  oracle/RNG-state equality, and single optimizer registration for both sizes and all
  three seeds;
- new tied checkpoint round trip plus rejection of divergent duplicate state entries;
- explicit reconstruction and validation of a historical untied fixture;
- exact eight record-set hashes and unchanged train/eval disjointness;
- exact configuration/manifest/summary/terminal schema for the new revision;
- exact ordered binding and tamper rejection for all four predecessor roots and the
  D1 decision diagnostic;
- rejection of diagnostic substitution, missing review acceptance, broad cleanliness
  exclusions, missing tying metadata, altered thresholds, and partial-cell success;
- existing tokenizer, response-only loss, smoke overfit, deterministic replay,
  selection, historical-root, and diagnostic regression tests.

Run targeted tests first, then the full repository suite because this changes the
model and historical validator. The implementation commit is high risk and requires
a fresh-context, same-family strict read-only review at an exact commit before an
experiment is considered.

## One authorized experiment after implementation acceptance

Only after a separate implementation commit is accepted may `main` authorize exactly
one full run at:

```text
artifacts/phase8_toy_lm_bridge/feasibility_005
```

The proposed command contract is:

```text
PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. python scripts/phase8_sequence_feasibility.py run --device cuda:0 --root artifacts/phase8_toy_lm_bridge/feasibility_005 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_001 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_002 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_003 --predecessor-root artifacts/phase8_toy_lm_bridge/feasibility_004 --decision-diagnostic-root artifacts/phase8_toy_lm_bridge/feasibility_diagnostic_001
```

The launcher must enforce the exact real process command, environment, CUDA device,
ordered inputs, fresh output root, and clean committed source before model
construction. `main` must separately bind the exact accepted implementation commit in
the launch packet; the manifest must record and check that current HEAD. A preflight
failure may be repaired only if it is
purely operational and creates no model/corpus/metric output; a semantic run gets no
retry at the same root.

All 24 family/model/seed cells run. Early stopping after a failing cell, pilot cells,
model selection, or a reduced run used to decide whether to launch the full matrix is
forbidden.

## Decision and stop rule

- The runner may publish `DONE` only when every cell reaches at least `52/64` and all
  artifacts and lineage pass its frozen validation. The root remains ineligible for
  selection until a fresh independent artifact review accepts it. Only then may
  `main` create and review a selection record and consider `010D`.
- If any cell is below `52/64`, preserve `FAILED.json` and all completed artifacts.
  Do not change the threshold, choose only the medium model, increase steps or data,
  change surfaces, or launch another feasibility root. The Phase 8 toy-language-model
  bridge stops without `010D` authorization.
- If this one repair cannot be implemented without an additional protocol choice,
  hyperparameter search, or evidence-dependent fallback, return the ambiguity to
  `main`; do not improvise a D3 proposal.

## Independent proposal-review gate

Review this proposal at an exact commit with a fresh-context `gpt-5.6-sol` agent at
`xhigh` reasoning. The review is read-only and must decide whether:

1. the repair is a single mechanism-targeted hypothesis supported enough by the D1
   observations to justify one prespecified falsification run;
2. all confounds other than weight tying are frozen and executable;
3. legacy validation cannot reinterpret or admit arbitrary artifacts;
4. diagnostic provenance cannot become selection evidence;
5. the implementation and run gates prevent search, partial selection, or silent
   protocol drift; and
6. the stop rule is unambiguous.

Return `ACCEPT` or `REJECT`, findings ordered by severity with file/line references,
and the smallest repair for each confirmed finding.
