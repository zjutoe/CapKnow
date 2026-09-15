# Task 010D — Training, Checkpoints, and Isolated Evaluator

## Launch contract

```text
executor: fresh-context Codex subagent
model override: gpt-5.5
fork_turns: none
working directory: /home/mye/src/llm/CapKnow
```

`main` supplies the exact accepted 010C prerequisite commit and proof that the
accepted feasibility decision passed. Read master Sections 4.2–4.4, 6.2, 8.3,
9.1–9.2, 10.3, and 12–15.

## Objective

Implement the formal per-state training/checkpoint path and an evaluation path that
scores only the immutable shared probe pack. Keep these components independent of
certificate construction and experiment orchestration.

## Allowed paths

```text
capability_certificate_lab/lm_bridge/train.py
capability_certificate_lab/lm_bridge/evaluator.py
tests/test_lm_bridge.py
```

Change model or corpus code only by stopping and returning a prerequisite defect to
`main`; do not expand the lease.

## Required implementation

- Implement the exact 1500-step, batch-64 cycling schedule, corpus-order RNG, paired
  initialization RNG, deterministic backend requirements, and failure behavior.
- Evaluate after steps `0,100,300,750,1500`; retain training loss and every raw
  generation; save exact weights only at steps 0 and 1500.
- Verify save/load tensor identity and logits before accepting a checkpoint path.
- Require every run to reference the one 512-prompt evaluation-pack checksum and
  reject per-condition or per-state probe rendering.
- Give the evaluator only a checkpoint and architecture/tokenizer configuration,
  rendered natural-language prompts, opaque probe keys, canonical answer strings,
  and decoding limits. Its callable surface must reject state, graph, DSL program,
  primitive/task identity, corpus, condition, seed, and training metadata, and the
  evaluator module must not import those objects or their defining packages.
- Score exact canonical success and exact `unable` separately. Refusal never counts
  as success for a positive cell, and an answer never counts as correct refusal for a
  negative cell.
- Atomically freeze immutable opaque-keyed prompts, generations, and exact-success/
  exact-`unable` bits before any later code may join probe keys to task order,
  programs, states, or the ground-truth matrix. No certificate or join code may run
  inside the evaluator.
- Record all 64 outputs per state/task cell and all fields needed for later
  sensitivity matrices and length diagnostics. Do not threshold or build
  certificates in this task.
- Keep callable training and evaluation primitives parameterized enough for the
  later fixed shard runner, but add no generic framework or registry hierarchy.

## Acceptance checks

Add targeted tests for master Section 13 items 13–16, the complete Section 4.4 input
allowlist and forbidden-import contract, freeze-before-join behavior, shared-pack
checksum isolation, the exact checkpoint schedule, response-only labels, and
raw-output completeness.

Run:

```text
python -m pytest -q tests/test_lm_bridge.py
git diff --check
git status --short
```

Do not launch a 16-state population or any formal scientific training.

## Stop boundary

Do not implement response matrices, certificates, aggregate metrics, shard/status
orchestration, benchmark projections, or reports. Do not commit unless explicitly
authorized. Return using the common format.
