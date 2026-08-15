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
  dropout zero, and parameter-count recording.
- Implement causal response-only shifted labels, exact AdamW settings, gradient
  clipping, deterministic RNG/backend setup, and greedy decoding bounded by the
  master protocol.
- Add the test-local four-record CPU byte-copy overfit control. It contains no Phase
  8 task or payload and must not leak into runtime configuration.
- Implement the four exact feasibility families, with 512 train and 64 held-out eval
  records per family, disjoint templates and operands, small/medium models, seeds
  0–2, 1500 steps, and the independent `52/64` pass requirement for every cell.
- Make the feasibility launcher refuse overwrite, reject Phase 8 scientific markers,
  retain per-record generations/checkpoints, publish atomic status, and build a
  checksum-bound manifest at the fixed root supplied by `main`.

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
the accepted `feasibility_001` evidence run: `main` may do that only after committing
and independently reviewing this stage from a clean worktree.

## Stop boundary

Do not implement formal state-specific training, the 512-prompt Phase 8 evaluator,
certificate logic, resource benchmark, shard runner, or formal experiment. A
feasibility failure is not permission to alter architecture or budget; return it to
`main`. Do not commit unless explicitly authorized. Return using the common format.
