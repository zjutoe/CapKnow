# Task 010C — Tokenizer, Model, and Feasibility Control

## Launch contract

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010B prerequisite commit. Read master Sections
8, 9.3–9.4, 12–13, and 15. The feasibility suite is isolated from all Phase 8
scientific tasks and cannot be tuned using future scientific outcomes.

Post-010C amendment: the independently accepted 010C-D2 proposal at commit
`9a767c6708c7c69f5ba98848250afcf50c8c5a6f` replaces the current model/checkpoint
revision with the tied `phase8_tied_io_v1` protocol for future feasibility and
formal paths. The retained `feasibility_001` through `feasibility_004` roots and the
accepted D1 diagnostic remain historical untied artifacts only under their exact
allowlists.

## Objective

Implement the fixed byte tokenizer, small/medium causal Transformer, response-only
loss and deterministic training primitives needed by the software smoke control and
the held-out sequence-transduction feasibility runner.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/tokenizer.py
capability_certificate_lab/lm_bridge/model.py
capability_certificate_lab/lm_bridge/train.py
scripts/phase8_sequence_feasibility.py
tests/test_lm_bridge.py
```

Do not modify corpus semantics or any other file.

## Required implementation

- Implement the exact 256-byte plus PAD/BOS/SEP/EOS tokenizer, strict special-token
  placement, byte-perfect UTF-8 round trip, padding, and hard sequence-length gates.
- Implement only the frozen small and medium decoder-only configurations using
  PyTorch public APIs, pre-norm causal blocks, learned position embeddings, GELU,
  dropout zero, input/output weight tying, explicit
  `embedding_weight_tying=true` and
  `model_protocol_revision="phase8_tied_io_v1"` fields, and parameter-count
  recording. The exact tied counts are `133120` for small and `859392` for medium.
- Implement causal response-only shifted labels, exact AdamW settings, gradient
  clipping, deterministic RNG/backend setup, and greedy decoding bounded by the
  master protocol.
- Add the test-local four-record CPU byte-copy overfit control. It contains no Phase
  8 task or payload and must not leak into runtime configuration.
- Implement the four exact feasibility families, with 512 train and 64 held-out eval
  records per family, disjoint templates and operands, small/medium models, seeds
  0–2, 1500 steps, and the independent `52/64` pass requirement for every cell.
- Make the feasibility launcher refuse overwrite, reject Phase 8 scientific markers,
  retain per-record generations/checkpoints, publish atomic terminal status, and build
  a checksum-bound manifest at the immutable numbered root supplied by `main`.
- Preserve failed roots and require the next numbered root for every retry or repair.
  Provide pure validation for a separate immutable selection record binding a passing
  root/manifest, exact source/configuration, per-cell counts, review verdict, and the
  terminal/manifest checksums of every predecessor root plus the transitive path and
  SHA-256 lineage of every predecessor selection record.

## Acceptance checks

Add targeted tests for master Section 13 items 5, 12–15, and 21. Tests may use tiny
test-local configurations except where exact frozen constants themselves are under
test.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

A local reduced smoke is allowed only when clearly marked non-evidence. Do not launch
an accepted feasibility evidence run: `main` may do that only after committing and
independently reviewing this stage from a clean worktree. An early accepted pass is
an engineering continuation decision; after later implementation stages, formal
execution requires a newly selected pass from the exact final source commit.

Under the accepted D2 amendment, the next current feasibility candidate is exactly
`artifacts/phase8_toy_lm_bridge/feasibility_005`, with all four historical
predecessor roots, the D1 decision diagnostic bound separately, exact `cuda:0`
argv/environment gates, all eight frozen record hashes, and all 24 cells executed.
This task text still does not authorize that run; it becomes runnable only after the
D2 implementation is committed and independently accepted.

## Stop boundary

Do not implement formal state-specific training, the 512-prompt Phase 8 evaluator,
certificate logic, resource benchmark, shard runner, or formal experiment. A
feasibility failure is not permission to alter architecture or budget; return it to
`main`. The accepted D2 implementation also does not authorize `010D`; `010D`
remains blocked until a reviewed passing `feasibility_005`, reviewed selection, and
explicit `main` decision exist. Do not commit unless explicitly authorized. Return
using the common format.
